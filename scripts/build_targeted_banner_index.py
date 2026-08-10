#!/usr/bin/env python3
"""Append a validated canonical extension to the frozen banner index."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-index", type=Path, required=True)
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-extension-rows", type=int, required=True)
    args = parser.parse_args()
    for path in (args.base_index, args.extension):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.output.exists() or args.output.with_suffix(args.output.suffix + ".tmp").exists():
        raise FileExistsError(f"Refusing to overwrite output: {args.output}")

    started = time.perf_counter()
    base_file = pq.ParquetFile(args.base_index)
    extension_file = pq.ParquetFile(args.extension)
    schema = base_file.schema_arrow
    missing = [name for name in schema.names if name not in extension_file.schema_arrow.names]
    if missing:
        raise ValueError(f"Extension is missing columns: {missing}")
    extension_rows = extension_file.metadata.num_rows
    if extension_rows != args.expected_extension_rows:
        raise ValueError(
            f"Unexpected extension rows: {extension_rows} != {args.expected_extension_rows}"
        )

    base_ids = np.asarray(
        pq.read_table(args.base_index, columns=["BannerID"])["BannerID"].to_numpy(),
        dtype=np.int64,
    )
    extension_ids = np.asarray(
        pq.read_table(args.extension, columns=["BannerID"])["BannerID"].to_numpy(),
        dtype=np.int64,
    )
    if np.any(base_ids <= 0) or np.unique(base_ids).size != base_ids.size:
        raise ValueError("Base BannerID values must be positive and unique")
    if np.any(extension_ids <= 0) or np.unique(extension_ids).size != extension_ids.size:
        raise ValueError("Extension BannerID values must be positive and unique")
    overlap = np.intersect1d(base_ids, extension_ids, assume_unique=True)
    if overlap.size:
        raise ValueError(f"Extension overlaps base index by {overlap.size} BannerID values")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    writer = pq.ParquetWriter(temporary, schema, compression="zstd")
    written = 0
    try:
        for source in (base_file, extension_file):
            for batch in source.iter_batches(batch_size=100_000, columns=schema.names):
                table = pa.Table.from_batches([batch]).cast(schema, safe=True)
                writer.write_table(table)
                written += table.num_rows
    finally:
        writer.close()
    expected = base_file.metadata.num_rows + extension_rows
    if written != expected:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Unexpected output rows: {written} != {expected}")
    os.replace(temporary, args.output)
    report = {
        "version": 1,
        "kind": "targeted_canonical_banner_index",
        "base_index": str(args.base_index.resolve()),
        "extension": str(args.extension.resolve()),
        "output": str(args.output.resolve()),
        "base_rows": int(base_file.metadata.num_rows),
        "extension_rows": int(extension_rows),
        "rows": int(written),
        "columns": schema.names,
        "output_bytes": args.output.stat().st_size,
        "output_sha256": file_sha256(args.output),
        "seconds": time.perf_counter() - started,
    }
    atomic_json(args.output.with_suffix(args.output.suffix + ".manifest.json"), report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
