#!/usr/bin/env python3
"""Upload a validated local BannerID parquet to a new temporary YT table."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--table", required=True)
    parser.add_argument("--expected-rows", type=int, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--step2-root",
        type=Path,
        default=Path("/home/astrofimuk/workspace/step2_ce"),
    )
    args = parser.parse_args()
    sys.path.insert(0, str(args.step2_root))
    import yt.wrapper as yt
    from common.yt_data import make_client

    if args.manifest.exists():
        raise FileExistsError(f"Refusing to overwrite manifest: {args.manifest}")
    values = np.asarray(
        pq.read_table(args.input, columns=["BannerID"])["BannerID"].to_numpy(),
        dtype=np.int64,
    )
    if values.size != args.expected_rows:
        raise ValueError(f"Unexpected rows: {values.size} != {args.expected_rows}")
    if np.any(values <= 0) or np.unique(values).size != values.size:
        raise ValueError("BannerID values must be positive and unique")
    client = make_client()
    if client.exists(args.table):
        raise FileExistsError(f"Refusing to overwrite YT table: {args.table}")
    started = time.perf_counter()
    client.create(
        "table",
        args.table,
        recursive=True,
        attributes={
            "schema": [{"name": "BannerID", "type": "int64", "required": True}],
            "expiration_timeout": 86_400_000,
        },
    )
    client.write_table(
        yt.TablePath(args.table, append=False),
        ({"BannerID": int(value)} for value in values),
    )
    row_count = int(client.get(args.table + "/@row_count"))
    if row_count != args.expected_rows:
        raise RuntimeError(f"Unexpected YT rows: {row_count} != {args.expected_rows}")
    report = {
        "version": 1,
        "kind": "temporary_banner_id_table",
        "input": str(args.input.resolve()),
        "table": args.table,
        "rows": row_count,
        "expiration_timeout_ms": 86_400_000,
        "seconds": time.perf_counter() - started,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.manifest.with_suffix(args.manifest.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.manifest)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
