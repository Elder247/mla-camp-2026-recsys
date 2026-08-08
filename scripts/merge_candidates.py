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
from mla_recsys.candidate_cache import enabled_sources, source_part_path  # noqa: E402
from mla_recsys.command import load_stage_context, require_choice  # noqa: E402
from mla_recsys.config import config_fingerprint, to_plain_dict  # noqa: E402
from mla_recsys.data import read_request_parquet, stable_partition  # noqa: E402
from mla_recsys.merge import merge_partition, merged_schema  # noqa: E402


def _merge_one(
    cfg_dict: dict,
    run_path_raw: str,
    split: str,
    partition: int,
    requests: list[dict],
    requests_path_raw: str,
    config_sha: str,
    force: bool,
) -> tuple[int, int]:
    cfg = OmegaConf.create(cfg_dict)
    run_path = Path(run_path_raw)
    requests_path = Path(requests_path_raw)
    output = (
        run_path
        / "candidates"
        / split
        / "merged"
        / f"part-{partition:05d}.parquet"
    )
    inputs = [fingerprint_file(requests_path)] + [
        fingerprint_file(source_part_path(run_path, split, source, partition))
        for source in enabled_sources(cfg)
    ]
    artifact_version = "merged_rrf_natural_v2"
    cache_key = make_cache_key(
        stage=f"merge_candidates_{split}_{partition}",
        artifact_version=artifact_version,
        config_sha256=config_sha,
        inputs=inputs,
    )
    if not force and validate_output_cache(
        output,
        expected_cache_key=cache_key,
        expected_schema=str(merged_schema(cfg)),
    )[0]:
        rows = pq.ParquetFile(output).metadata.num_rows
    else:
        table = merge_partition(
            cfg=cfg,
            run_path=run_path,
            split=split,
            partition=partition,
            requests=requests,
        )
        with atomic_output_path(output) as temporary:
            pq.write_table(table, temporary, compression="zstd")
        rows = table.num_rows
        write_output_manifest(
            output,
            stage=f"merge_candidates_{split}_{partition}",
            artifact_version=artifact_version,
            config_sha256=config_sha,
            inputs=inputs,
            rows=rows,
            schema=str(merged_schema(cfg)),
            scope=str(cfg.runtime.scope),
        )
    return partition, rows


def main() -> int:
    context = load_stage_context(
        "Merge cached sources with deterministic RRF",
        extra_keys=("split", "force"),
    )
    cfg = context.cfg
    split = require_choice(context, "split", ("train", "holdout", "full_train", "test"))
    partitions = int(cfg.data.partition_count)
    requests_path = context.store.path / "data" / f"{split}_requests.parquet"
    requests = read_request_parquet(requests_path)
    requests_by_partition = {index: [] for index in range(partitions)}
    for request in requests:
        requests_by_partition[stable_partition(request["request_id"], partitions)].append(request)
    config_sha = config_fingerprint(cfg)
    force = context.values.get("force", "false").lower() == "true"
    workers = max(1, int(cfg.pipeline.get("merge_partition_workers", 1)))
    args = [
        (
            to_plain_dict(cfg),
            str(context.store.path),
            split,
            partition,
            requests_by_partition[partition],
            str(requests_path),
            config_sha,
            force,
        )
        for partition in range(partitions)
    ]
    if workers == 1:
        results = [_merge_one(*values) for values in args]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_merge_one, *values) for values in args]
            results = [future.result() for future in futures]
    rows_by_partition = dict(results)
    part_rows = [rows_by_partition[index] for index in range(partitions)]
    total_rows = sum(part_rows)
    report = {"split": split, "requests": len(requests), "rows": total_rows, "partition_rows": part_rows}
    atomic_write_json(context.store.path / "metrics" / f"merge_{split}.json", report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
