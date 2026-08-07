#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mla_recsys.artifacts import atomic_write_json, fingerprint_file, write_output_manifest
from mla_recsys.command import load_stage_context, run_data_dir
from mla_recsys.config import config_fingerprint
from mla_recsys.data import (
    REQUEST_SCHEMA,
    load_test_requests,
    load_validation_requests,
    temporal_split,
    write_request_parquet,
)


def main() -> int:
    context = load_stage_context("Prepare request-level temporal datasets")
    cfg = context.cfg
    val_path = Path(str(cfg.paths.val_clicks))
    rows = load_validation_requests(val_path)
    train, holdout = temporal_split(rows, boundary=int(cfg.split.fit.end_exclusive))
    if len(train) != int(cfg.split.observed.fit_request_groups):
        raise RuntimeError(f"Unexpected fit request count: {len(train)}")
    if len(holdout) != int(cfg.split.observed.holdout_request_groups):
        raise RuntimeError(f"Unexpected holdout request count: {len(holdout)}")

    mode = str(cfg.runtime.mode)
    datasets = {"train": train, "holdout": holdout}
    if mode == "smoke":
        limit = int(cfg.data.smoke_requests_per_split)
        datasets = {name: values[:limit] for name, values in datasets.items()}
    if str(cfg.runtime.scope) == "full":
        datasets["full_train"] = train + holdout
        datasets["test"] = load_test_requests(Path(str(cfg.paths.test_clicks)))

    output_dir = run_data_dir(context)
    report = {"split_version": str(cfg.split.version), "datasets": {}}
    for name, values in datasets.items():
        path = output_dir / f"{name}_requests.parquet"
        table = write_request_parquet(path, values)
        inputs = [
            fingerprint_file(Path(str(cfg.paths.test_clicks if name == "test" else cfg.paths.val_clicks)))
        ]
        manifest = write_output_manifest(
            path,
            stage="prepare_data",
            artifact_version=str(cfg.data.request_schema_version),
            config_sha256=config_fingerprint(cfg),
            inputs=inputs,
            rows=table.num_rows,
            schema=str(REQUEST_SCHEMA),
            scope=str(cfg.runtime.scope),
        )
        report["datasets"][name] = {
            "path": str(path),
            "rows": table.num_rows,
            "cache_key": manifest["cache_key"],
        }
    metric_path = context.store.path / "metrics" / "data.json"
    atomic_write_json(metric_path, report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
