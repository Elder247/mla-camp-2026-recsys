#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pyarrow.parquet as pq
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mla_recsys.artifacts import (  # noqa: E402
    atomic_write_json,
    fingerprint_file,
    make_cache_key,
    validate_output_cache,
)
from mla_recsys.candidate_cache import (  # noqa: E402
    SOURCE_SCHEMA,
    cache_is_valid,
    finalize_source_manifests,
    generate_source_candidates,
    load_source,
    source_part_path,
    source_input_fingerprints,
)
from mla_recsys.command import load_stage_context, require_choice  # noqa: E402
from mla_recsys.config import config_fingerprint  # noqa: E402
from mla_recsys.data import read_request_parquet  # noqa: E402
from mla_recsys.temporal_candidates import (  # noqa: E402
    generate_temporal_source_candidates,
    is_temporal_source,
    temporal_source_inputs,
)


def _same_content(left: Path, right: Path) -> bool:
    a = fingerprint_file(left)
    b = fingerprint_file(right)
    return all(a.get(key) == b.get(key) for key in ("exists", "size_bytes", "sha256"))


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


def try_reuse_candidates(
    *,
    cfg: object,
    run_path: Path,
    source: str,
    split: str,
    request_path: Path,
    partitions: int,
    artifact_version: str,
    config_sha: str,
    inputs: list[dict],
) -> dict | None:
    donor_raw = cfg.candidates.get("reuse_run")
    if not donor_raw:
        return None
    donor = Path(str(donor_raw))
    donor_request = donor / "data" / f"{split}_requests.parquet"
    donor_config = donor / "config.yaml"
    donor_metric = donor / "metrics" / f"generate_{split}_{source}.json"
    if not donor_request.is_file() or not donor_config.is_file() or not donor_metric.is_file():
        return None
    if not _same_content(request_path, donor_request):
        return None
    previous_cfg = OmegaConf.load(donor_config)
    if OmegaConf.to_container(
        previous_cfg.candidates.generators[source], resolve=True
    ) != OmegaConf.to_container(cfg.candidates.generators[source], resolve=True):
        return None
    rows: list[int] = []
    donor_parts: list[Path] = []
    for partition in range(partitions):
        path = source_part_path(donor, split, source, partition)
        manifest_path = path.with_name(path.name + ".manifest.json")
        if not manifest_path.is_file():
            return None
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        valid, _ = validate_output_cache(
            path,
            expected_cache_key=str(manifest.get("cache_key")),
            expected_schema=str(SOURCE_SCHEMA),
        )
        if not valid or pq.ParquetFile(path).schema_arrow != SOURCE_SCHEMA:
            return None
        donor_parts.append(path)
        rows.append(int(manifest.get("rows") or pq.ParquetFile(path).metadata.num_rows))
    for partition, donor_part in enumerate(donor_parts):
        _materialize_file(
            donor_part,
            source_part_path(run_path, split, source, partition),
        )
    finalize_source_manifests(
        run_path=run_path,
        split=split,
        source=source,
        partitions=partitions,
        rows=rows,
        artifact_version=artifact_version,
        config_sha256=config_sha,
        inputs=inputs,
        scope=str(cfg.runtime.scope),
    )
    report = json.loads(donor_metric.read_text(encoding="utf-8"))
    report.update(
        status="cross_run_cache_hit",
        wall_seconds=0.0,
        reused_from=str(donor),
        partition_rows=rows,
        rows=sum(rows),
    )
    return report


def main() -> int:
    context = load_stage_context(
        "Generate and cache one candidate source",
        extra_keys=("cg", "split", "force"),
    )
    cfg = context.cfg
    cg = require_choice(context, "cg", cfg.candidates.generators.keys())
    split = require_choice(context, "split", ("train", "holdout", "full_train", "test"))
    if not bool(cfg.candidates.generators[cg].get("enabled", False)):
        raise ValueError(f"Candidate generator is disabled: {cg}")
    request_path = context.store.path / "data" / f"{split}_requests.parquet"
    temporal = is_temporal_source(cfg, cg)
    spec = None if temporal else load_source(cfg, cg)
    inputs = (
        temporal_source_inputs(
            cfg=cfg,
            run_path=context.store.path,
            split=split,
            source=cg,
        )
        if temporal
        else source_input_fingerprints(spec, request_path)
    )
    config_sha = config_fingerprint(cfg)
    artifact_version = f"{cg}_candidates_v1"
    cache_key = make_cache_key(
        stage=f"generate_candidates_{cg}_{split}",
        artifact_version=artifact_version,
        config_sha256=config_sha,
        inputs=inputs,
    )
    partitions = int(cfg.data.partition_count)
    force = context.values.get("force", "false").lower() == "true"
    if not force and cache_is_valid(
        run_path=context.store.path,
        split=split,
        source=cg,
        partitions=partitions,
        cache_key=cache_key,
    ):
        print(json.dumps({"source": cg, "split": split, "status": "cache_hit"}))
        return 0
    if not force:
        reused = try_reuse_candidates(
            cfg=cfg,
            run_path=context.store.path,
            source=cg,
            split=split,
            request_path=request_path,
            partitions=partitions,
            artifact_version=artifact_version,
            config_sha=config_sha,
            inputs=inputs,
        )
        if reused is not None:
            metric_path = context.store.path / "metrics" / f"generate_{split}_{cg}.json"
            atomic_write_json(metric_path, reused)
            print(json.dumps(reused, indent=2))
            return 0
    requests = read_request_parquet(request_path)
    if temporal:
        report = generate_temporal_source_candidates(
            cfg=cfg,
            run_path=context.store.path,
            split=split,
            source=cg,
            requests=requests,
            partitions=partitions,
            buffer_rows=int(cfg.data.candidate_buffer_rows),
        )
    else:
        assert spec is not None
        report = generate_source_candidates(
            spec=spec,
            requests=requests,
            run_path=context.store.path,
            split=split,
            partitions=partitions,
            buffer_rows=int(cfg.data.candidate_buffer_rows),
        )
    finalize_source_manifests(
        run_path=context.store.path,
        split=split,
        source=cg,
        partitions=partitions,
        rows=report["partition_rows"],
        artifact_version=artifact_version,
        config_sha256=config_sha,
        inputs=inputs,
        scope=str(cfg.runtime.scope),
    )
    metric_path = context.store.path / "metrics" / f"generate_{split}_{cg}.json"
    atomic_write_json(metric_path, report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
