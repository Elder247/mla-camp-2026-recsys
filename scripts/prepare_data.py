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
    read_request_parquet,
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

    oof_path = None
    oof: list[dict] = []
    walk_forward = cfg.get("walk_forward_ranker", {})
    if bool(walk_forward.get("enabled", False)):
        path_key = str(walk_forward.oof_requests_path_key)
        oof_path = Path(str(cfg.paths[path_key]))
        if not oof_path.is_file():
            raise FileNotFoundError(f"Walk-forward OOF requests are missing: {oof_path}")
        oof = read_request_parquet(oof_path)
        if not oof:
            raise RuntimeError("Walk-forward OOF requests are empty")
        oof_ids = {str(row["request_id"]) for row in oof}
        validation_ids = {str(row["request_id"]) for row in rows}
        if len(oof_ids) != len(oof):
            raise RuntimeError("Walk-forward OOF request ids are not unique")
        if oof_ids & validation_ids:
            raise RuntimeError("Walk-forward OOF and validation request ids overlap")
        if max(int(row["show_time"]) for row in oof) >= min(
            int(row["show_time"]) for row in rows
        ):
            raise RuntimeError("Walk-forward OOF rows are not strictly before validation")
        train = sorted(
            [*oof, *train],
            key=lambda row: (int(row["show_time"]), str(row["request_id"])),
        )

    mode = str(cfg.runtime.mode)
    datasets = {"train": train, "holdout": holdout}
    if mode == "smoke":
        limit = int(cfg.data.smoke_requests_per_split)
        datasets = {name: values[:limit] for name, values in datasets.items()}
    if str(cfg.runtime.scope) == "full":
        datasets["full_train"] = sorted(
            [*train, *holdout],
            key=lambda row: (int(row["show_time"]), str(row["request_id"])),
        )
        datasets["test"] = load_test_requests(Path(str(cfg.paths.test_clicks)))

    output_dir = run_data_dir(context)
    report = {
        "split_version": str(cfg.split.version),
        "walk_forward_oof_rows": len(oof),
        "datasets": {},
    }
    for name, values in datasets.items():
        path = output_dir / f"{name}_requests.parquet"
        table = write_request_parquet(path, values)
        inputs = [
            fingerprint_file(Path(str(cfg.paths.test_clicks if name == "test" else cfg.paths.val_clicks)))
        ]
        if oof_path is not None and name in {"train", "full_train"}:
            inputs.append(fingerprint_file(oof_path))
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
