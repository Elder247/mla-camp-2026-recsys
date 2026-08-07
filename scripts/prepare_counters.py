#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pyarrow as pa
import pyarrow.parquet as pq

from mla_recsys.artifacts import (
    atomic_output_path,
    atomic_write_json,
    fingerprint_file,
    write_output_manifest,
)
from mla_recsys.command import load_stage_context
from mla_recsys.config import config_fingerprint


COUNTER_SCHEMA = pa.schema(
    [
        pa.field("entity_type", pa.string(), nullable=False),
        pa.field("entity_id", pa.string(), nullable=False),
        pa.field("week_start", pa.uint64(), nullable=False),
        pa.field("prior_shows", pa.uint64(), nullable=False),
        pa.field("prior_clicks", pa.uint64(), nullable=False),
        pa.field("prior_source_cost", pa.float64(), nullable=False),
    ]
)


def main() -> int:
    context = load_stage_context("Prepare scope-separated leakage-safe counter contract")
    cfg = context.cfg
    scope = str(cfg.runtime.scope)
    spec = cfg.split.counters[scope]
    output_dir = context.store.path / "counters" / scope
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "weekly_snapshots.parquet"
    # Iteration 0 baseline has no counter features. Persist an explicit empty,
    # typed artifact so a consumer cannot silently fall back to another scope.
    table = pa.Table.from_pylist([], schema=COUNTER_SCHEMA)
    with atomic_output_path(output) as temporary:
        pq.write_table(table, temporary, compression="zstd")
    inputs = [
        fingerprint_file(
            Path(str(cfg.paths.history_artifact)) / "history_candidates.parquet"
        )
    ]
    write_output_manifest(
        output,
        stage="prepare_counters",
        artifact_version="weekly_past_only_v1",
        config_sha256=config_fingerprint(cfg),
        inputs=inputs,
        rows=0,
        schema=str(COUNTER_SCHEMA),
        scope=scope,
    )
    scope_manifest = {
        "version": 1,
        "scope": scope,
        "cutoff_ts": int(spec.cutoff_ts),
        "source_tables": list(spec.sources),
        "semantics": "week(row) sees only weeks strictly before week(row)",
        "used_by_i0_features": False,
        "legacy_history_input": inputs[0],
    }
    atomic_write_json(output_dir / "scope.json", scope_manifest)
    print(json.dumps(scope_manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
