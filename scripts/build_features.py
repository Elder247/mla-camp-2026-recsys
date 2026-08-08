#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
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
        counter_lookup = CounterLookup.from_parquet(counter_path)
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
    if not force and validate_output_cache(
        output,
        expected_cache_key=cache_key,
        expected_schema=str(feature_schema(cfg)),
    )[0]:
        table = pq.read_table(
            output,
            columns=["group_has_positive", "is_positive", "request_id"],
        )
        groups = len(set(table["request_id"].to_pylist()))
        positives = int(sum(table["is_positive"].to_pylist()))
        positive_group_ids = {
            request_id
            for request_id, flag in zip(
                table["request_id"].to_pylist(),
                table["group_has_positive"].to_pylist(),
            )
            if flag
        }
        stats = {
            "rows": table.num_rows,
            "groups": groups,
            "positive_groups": len(positive_group_ids),
            "missed_positive_groups": max(0, groups - len(positive_group_ids))
            if positives >= 0
            else 0,
        }
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
    return partition, stats


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
    totals = {"groups": 0, "positive_groups": 0, "missed_positive_groups": 0, "rows": 0}
    cfg_dict = to_plain_dict(cfg)
    workers = max(1, int(cfg.pipeline.get("feature_partition_workers", 1)))
    initargs = (
        cfg_dict,
        str(context.store.path),
        split,
        str(request_path),
        str(banner_index_path),
    )
    if workers == 1:
        _initialize_feature_worker(*initargs)
        results = [
            _build_one_feature_partition(partition, config_sha, force)
            for partition in range(partitions)
        ]
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
    report = {"split": split, **totals, "partition_rows": partition_rows}
    atomic_write_json(context.store.path / "metrics" / f"features_{split}.json", report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
