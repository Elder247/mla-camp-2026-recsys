#!/usr/bin/env python3
"""Select the most frequent train-only banners outside an existing index."""
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--exclude-index", type=Path, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.count <= 0:
        parser.error("--count must be positive")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite output: {args.output}")
    started = time.perf_counter()
    prior = pq.read_table(args.prior, columns=["banner_id", "count"]).to_pydict()
    banner_ids = np.asarray(prior["banner_id"], dtype=np.int64)
    counts = np.asarray(prior["count"], dtype=np.int64)
    if np.any(banner_ids <= 0) or np.any(counts <= 0):
        raise ValueError("Prior ids and counts must be positive")
    if np.unique(banner_ids).size != banner_ids.size:
        raise ValueError("Prior BannerID values are not unique")
    excluded = np.sort(
        np.asarray(
            pq.read_table(args.exclude_index, columns=["BannerID"])["BannerID"].to_numpy(),
            dtype=np.int64,
        )
    )
    order = np.lexsort((banner_ids, -counts))
    ranked_ids = banner_ids[order]
    positions = np.searchsorted(excluded, ranked_ids)
    inside = positions < excluded.size
    inside[inside] &= excluded[positions[inside]] == ranked_ids[inside]
    selected = ranked_ids[~inside][: args.count]
    selected_counts = counts[order][~inside][: args.count]
    if selected.size != args.count or np.unique(selected).size != selected.size:
        raise RuntimeError(f"Unable to select {args.count} unique outside-index banners")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    pq.write_table(
        pa.table(
            {
                "BannerID": pa.array(selected, type=pa.int64()),
                "TrainCount": pa.array(selected_counts, type=pa.int64()),
            }
        ),
        temporary,
        compression="zstd",
    )
    os.replace(temporary, args.output)
    report = {
        "version": 1,
        "kind": "train_only_popularity_index_extension",
        "prior": str(args.prior.resolve()),
        "exclude_index": str(args.exclude_index.resolve()),
        "rows": int(selected.size),
        "maximum_train_count": int(selected_counts[0]),
        "minimum_train_count": int(selected_counts[-1]),
        "output": str(args.output.resolve()),
        "output_bytes": args.output.stat().st_size,
        "output_sha256": sha256(args.output),
        "seconds": time.perf_counter() - started,
    }
    manifest = args.output.with_suffix(args.output.suffix + ".manifest.json")
    temporary_manifest = manifest.with_suffix(manifest.suffix + ".tmp")
    temporary_manifest.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_manifest, manifest)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
