#!/usr/bin/env python3
"""Filter cached candidate rankings against an excluded banner universe."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(ROOT / "src"))

from mla_recsys.artifacts import (  # noqa: E402
    atomic_output_path,
    atomic_write_json,
    fingerprint_file,
)


def filter_and_rerank(table: pa.Table, excluded: set[int]) -> pa.Table:
    """Drop excluded banners and restore contiguous per-request source ranks."""

    next_rank: dict[int, int] = {}
    output = []
    for row in table.to_pylist():
        if int(row["banner_id"]) in excluded:
            continue
        hit_log_id = int(row["hit_log_id"])
        rank = next_rank.get(hit_log_id, 1)
        row["source_rank"] = rank
        next_rank[hit_log_id] = rank + 1
        output.append(row)
    return pa.Table.from_pylist(output, schema=table.schema)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--exclude-banner-index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    parts = sorted(args.input_dir.glob("part-*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No candidate partitions at {args.input_dir}")
    index = pq.read_table(args.exclude_banner_index, columns=["BannerID"])
    excluded = {int(value) for value in index["BannerID"].to_pylist()}

    started = time.monotonic()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_rows = 0
    output_rows = 0
    output_parts = []
    for source in parts:
        table = pq.read_table(source)
        filtered = filter_and_rerank(table, excluded)
        target = args.output_dir / source.name
        with atomic_output_path(target) as temporary:
            pq.write_table(filtered, temporary, compression="zstd")
        input_rows += table.num_rows
        output_rows += filtered.num_rows
        output_parts.append(fingerprint_file(target))

    report = {
        "status": "completed",
        "input_dir": str(args.input_dir),
        "exclude_banner_index": fingerprint_file(args.exclude_banner_index),
        "output_dir": str(args.output_dir),
        "partitions": len(parts),
        "input_rows": input_rows,
        "output_rows": output_rows,
        "excluded_rows": input_rows - output_rows,
        "output_parts": output_parts,
        "wall_seconds": time.monotonic() - started,
    }
    report_path = args.report or args.output_dir / "manifest.json"
    atomic_write_json(report_path, report)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
