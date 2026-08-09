#!/usr/bin/env python3
from __future__ import annotations

import json
import multiprocessing
import os
import shutil
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mla_recsys.artifacts import (  # noqa: E402
    atomic_output_path,
    atomic_write_json,
    fingerprint_file,
    make_cache_key,
    validate_output_cache,
    write_output_manifest,
)
from mla_recsys.command import load_stage_context, require_choice  # noqa: E402
from mla_recsys.config import config_fingerprint, to_plain_dict  # noqa: E402
from mla_recsys.data import read_request_parquet  # noqa: E402
from mla_recsys.counters import CounterLookup, validate_scope  # noqa: E402


_FEATURE_STATE: dict = {}


def _fingerprint_content(value: dict) -> tuple[object, ...]:
    return tuple(value.get(key) for key in ("exists", "size_bytes", "sha256"))


def _materialize_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_raw = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    os.close(fd)
    temporary = Path(temporary_raw)
    temporary.unlink()
    try:
        try:
            os.link(source, temporary)
        except OSError:
            shutil.copy2(source, temporary)
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _feature_config_without_reuse(cfg: object) -> dict:
    value = OmegaConf.to_container(cfg.features, resolve=True)
    assert isinstance(value, dict)
    value.pop("reuse_run", None)
    return value


def _table_stats(table: object) -> dict[str, int]:
    request_ids = table["request_id"].to_pylist()
    positive_flags = table["is_positive"].to_pylist()
    group_positive_flags = table["group_has_positive"].to_pylist()
    groups = len(set(request_ids))
    positives = int(sum(positive_flags))
    positive_group_ids = {
        request_id
        for request_id, flag in zip(request_ids, group_positive_flags)
        if flag
    }
    return {
        "rows": table.num_rows,
        "groups": groups,
        "positive_groups": len(positive_group_ids),
        "missed_positive_groups": max(0, groups - len(positive_group_ids))
        if positives >= 0
        else 0,
    }


def _try_reuse_feature_partition(
    *,
    partition: int,
    output: Path,
    inputs: list[dict],
    config_sha: str,
) -> dict[str, int] | None:
    from mla_recsys.feature_cache import feature_schema

    cfg = _FEATURE_STATE["cfg"]
    donor_raw = cfg.features.get("reuse_run")
    if not donor_raw:
        return None
    donor = Path(str(donor_raw))
    donor_config_path = donor / "config.yaml"
    donor_output = donor / "features" / _FEATURE_STATE["split"] / output.name
    donor_manifest_path = donor_output.with_name(donor_output.name + ".manifest.json")
    if not donor_config_path.is_file() or not donor_manifest_path.is_file():
        return None
    donor_cfg = OmegaConf.load(donor_config_path)
    if _feature_config_without_reuse(donor_cfg) != _feature_config_without_reuse(cfg):
        return None
    donor_manifest = json.loads(donor_manifest_path.read_text(encoding="utf-8"))
    if str(donor_manifest.get("artifact_version")) != str(cfg.features.version):
        return None
    donor_inputs = donor_manifest.get("inputs")
    if not isinstance(donor_inputs, list) or len(donor_inputs) != len(inputs):
        return None
    if any(
        _fingerprint_content(current) != _fingerprint_content(previous)
        for current, previous in zip(inputs, donor_inputs)
    ):
        return None
    valid, _ = validate_output_cache(
        donor_output,
        expected_cache_key=str(donor_manifest.get("cache_key")),
        expected_schema=str(feature_schema(cfg)),
    )
    if not valid:
        return None
    _materialize_file(donor_output, output)
    table = pq.read_table(
        output,
        columns=["group_has_positive", "is_positive", "request_id"],
    )
    write_output_manifest(
        output,
        stage=f"build_features_{_FEATURE_STATE['split']}_{partition}",
        artifact_version=str(cfg.features.version),
        config_sha256=config_sha,
        inputs=inputs,
        rows=table.num_rows,
        schema=str(feature_schema(cfg)),
        scope=str(cfg.runtime.scope),
    )
    return _table_stats(table)


def _preinitialize_feature_reuse(
    *,
    cfg: object,
    run_path: Path,
    split: str,
    request_path: Path,
    banner_index_path: Path,
    partitions: int,
    config_sha: str,
    force: bool,
) -> list[tuple[int, dict[str, int]]] | None:
    """Reuse every feature part before loading multi-gigabyte lookup state."""
    if force or not cfg.features.get("reuse_run"):
        return None
    counter_inputs: list[dict] = []
    if str(cfg.features.version) != "feature_v1":
        counter_dir = run_path / "counters" / str(cfg.runtime.scope)
        counter_path = counter_dir / "click_events.parquet"
        scope_path = counter_dir / "scope.json"
        scope_manifest = json.loads(scope_path.read_text(encoding="utf-8"))
        validate_scope(str(cfg.runtime.scope), str(scope_manifest["scope"]))
        counter_inputs = [fingerprint_file(counter_path), fingerprint_file(scope_path)]
    _FEATURE_STATE.clear()
    _FEATURE_STATE.update(
        cfg=cfg,
        run_path=run_path,
        split=split,
        request_path=request_path,
        banner_index_path=banner_index_path,
        counter_inputs=counter_inputs,
    )
    reused_results: list[tuple[int, dict[str, int]]] = []
    for partition in range(partitions):
        merged_path = (
            run_path
            / "candidates"
            / split
            / "merged"
            / f"part-{partition:05d}.parquet"
        )
        output = run_path / "features" / split / f"part-{partition:05d}.parquet"
        inputs = [
            fingerprint_file(request_path),
            fingerprint_file(merged_path),
            fingerprint_file(banner_index_path),
            *counter_inputs,
        ]
        reused = _try_reuse_feature_partition(
            partition=partition,
            output=output,
            inputs=inputs,
            config_sha=config_sha,
        )
        if reused is None:
            return None
        reused_results.append((partition, {**reused, "reused": 1}))
    return reused_results


def _initialize_feature_worker(
    cfg_dict: dict,
    run_path_raw: str,
    split: str,
    request_path_raw: str,
    banner_index_path_raw: str,
) -> None:
    from mla_recsys.feature_cache import BannerIndex

    cfg = OmegaConf.create(cfg_dict)
    run_path = Path(run_path_raw)
    request_path = Path(request_path_raw)
    requests = {str(row["request_id"]): row for row in read_request_parquet(request_path)}
    banner_index_path = Path(banner_index_path_raw)
    banner_index = BannerIndex(banner_index_path)
    counter_lookup = None
    frozen_counter_cutoff = None
    counter_inputs: list[dict] = []
    if str(cfg.features.version) != "feature_v1":
        counter_dir = run_path / "counters" / str(cfg.runtime.scope)
        counter_path = counter_dir / "click_events.parquet"
        scope_path = counter_dir / "scope.json"
        scope_manifest = json.loads(scope_path.read_text(encoding="utf-8"))
        validate_scope(str(cfg.runtime.scope), str(scope_manifest["scope"]))
        frozen_counter_cutoff = int(scope_manifest["frozen_cutoff_ts"])
        counter_lookup = CounterLookup.from_parquet(
            counter_path,
            families=[str(value) for value in cfg.features.counter_families],
        )
        counter_inputs = [fingerprint_file(counter_path), fingerprint_file(scope_path)]
    _FEATURE_STATE.clear()
    _FEATURE_STATE.update(
        cfg=cfg,
        run_path=run_path,
        split=split,
        request_path=request_path,
        requests=requests,
        banner_index_path=banner_index_path,
        banner_index=banner_index,
        counter_lookup=counter_lookup,
        frozen_counter_cutoff=frozen_counter_cutoff,
        counter_inputs=counter_inputs,
    )


def _build_one_feature_partition(
    partition: int,
    config_sha: str,
    force: bool,
) -> tuple[int, dict[str, int]]:
    from mla_recsys.feature_cache import build_feature_partition, feature_schema

    cfg = _FEATURE_STATE["cfg"]
    run_path = _FEATURE_STATE["run_path"]
    split = _FEATURE_STATE["split"]
    merged_path = (
        run_path / "candidates" / split / "merged" / f"part-{partition:05d}.parquet"
    )
    output = run_path / "features" / split / f"part-{partition:05d}.parquet"
    inputs = [
        fingerprint_file(_FEATURE_STATE["request_path"]),
        fingerprint_file(merged_path),
        fingerprint_file(_FEATURE_STATE["banner_index_path"]),
        *_FEATURE_STATE["counter_inputs"],
    ]
    artifact_version = str(cfg.features.version)
    cache_key = make_cache_key(
        stage=f"build_features_{split}_{partition}",
        artifact_version=artifact_version,
        config_sha256=config_sha,
        inputs=inputs,
    )
    reused = None if force else _try_reuse_feature_partition(
        partition=partition,
        output=output,
        inputs=inputs,
        config_sha=config_sha,
    )
    if reused is not None:
        return partition, {**reused, "reused": 1}
    if not force and validate_output_cache(
        output,
        expected_cache_key=cache_key,
        expected_schema=str(feature_schema(cfg)),
    )[0]:
        table = pq.read_table(
            output,
            columns=["group_has_positive", "is_positive", "request_id"],
        )
        stats = _table_stats(table)
    else:
        table, stats = build_feature_partition(
            cfg=cfg,
            merged_path=merged_path,
            requests=_FEATURE_STATE["requests"],
            banner_index=_FEATURE_STATE["banner_index"],
            counter_lookup=_FEATURE_STATE["counter_lookup"],
            frozen_counter_cutoff=_FEATURE_STATE["frozen_counter_cutoff"],
        )
        with atomic_output_path(output) as temporary:
            pq.write_table(table, temporary, compression="zstd")
        write_output_manifest(
            output,
            stage=f"build_features_{split}_{partition}",
            artifact_version=artifact_version,
            config_sha256=config_sha,
            inputs=inputs,
            rows=table.num_rows,
            schema=str(feature_schema(cfg)),
            scope=str(cfg.runtime.scope),
        )
    return partition, {**stats, "reused": 0}


def main() -> int:
    context = load_stage_context(
        "Build baseline features from the frozen natural pool",
        extra_keys=("split", "force"),
    )
    cfg = context.cfg
    split = require_choice(context, "split", ("train", "holdout", "full_train", "test"))
    partitions = int(cfg.data.partition_count)
    request_path = context.store.path / "data" / f"{split}_requests.parquet"
    banner_index_path = Path(str(cfg.paths.banner_index))
    force = context.values.get("force", "false").lower() == "true"
    config_sha = config_fingerprint(cfg)
    totals = {
        "groups": 0,
        "positive_groups": 0,
        "missed_positive_groups": 0,
        "rows": 0,
        "reused": 0,
    }
    cfg_dict = to_plain_dict(cfg)
    workers = max(1, int(cfg.pipeline.get("feature_partition_workers", 1)))
    results = _preinitialize_feature_reuse(
        cfg=cfg,
        run_path=context.store.path,
        split=split,
        request_path=request_path,
        banner_index_path=banner_index_path,
        partitions=partitions,
        config_sha=config_sha,
        force=force,
    )
    initargs = (
        cfg_dict,
        str(context.store.path),
        split,
        str(request_path),
        str(banner_index_path),
    )
    if results is not None:
        pass
    elif workers == 1:
        _initialize_feature_worker(*initargs)
        results = [
            _build_one_feature_partition(partition, config_sha, force)
            for partition in range(partitions)
        ]
    elif sys.platform.startswith("linux"):
        # CounterLookup and BannerIndex are large immutable structures. Build
        # them once and fork after initialization so Linux shares their pages
        # copy-on-write instead of parsing the same parquet in every worker.
        _initialize_feature_worker(*initargs)
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("fork"),
        ) as executor:
            futures = [
                executor.submit(_build_one_feature_partition, partition, config_sha, force)
                for partition in range(partitions)
            ]
            results = [future.result() for future in futures]
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_initialize_feature_worker,
            initargs=initargs,
        ) as executor:
            futures = [
                executor.submit(_build_one_feature_partition, partition, config_sha, force)
                for partition in range(partitions)
            ]
            results = [future.result() for future in futures]
    stats_by_partition = dict(results)
    partition_rows = []
    for partition in range(partitions):
        stats = stats_by_partition[partition]
        for key in totals:
            totals[key] += int(stats[key])
        partition_rows.append(int(stats["rows"]))
    report = {
        "split": split,
        **totals,
        "reused_from": str(cfg.features.get("reuse_run") or "") or None,
        "partition_rows": partition_rows,
    }
    atomic_write_json(context.store.path / "metrics" / f"features_{split}.json", report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
