#!/usr/bin/env python3
"""Export a deterministic top-K CatBoost ranking from cached natural-pool features."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from catboost import CatBoostRanker

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mla_recsys.artifacts import (  # noqa: E402
    atomic_output_path,
    fingerprint_file,
    write_output_manifest,
)


def matrix(table: pa.Table, names: list[str]) -> np.ndarray:
    return np.column_stack(
        [table[name].combine_chunks().to_numpy(zero_copy_only=False) for name in names]
    ).astype(np.float32, copy=False)


def ordered_banner_ids(
    scores: np.ndarray,
    pre_ranks: np.ndarray,
    banner_ids: np.ndarray,
    *,
    top_k: int,
) -> list[int]:
    order = np.lexsort((banner_ids, pre_ranks, -scores))
    return [int(banner_ids[index]) for index in order[:top_k]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--split", choices=("holdout", "test"), required=True)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.top_k <= 0:
        parser.error("--top-k must be positive")

    metadata_path = args.run / "models" / "catboost.json"
    model_path = args.run / "models" / "catboost.cbm"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    names = list(metadata["feature_names"])
    model = CatBoostRanker()
    model.load_model(str(model_path))

    rows: list[dict[str, object]] = []
    feature_paths = sorted((args.run / "features" / args.split).glob("part-*.parquet"))
    if not feature_paths:
        raise FileNotFoundError(f"No feature parts for {args.split}: {args.run}")
    for path in feature_paths:
        columns = ["request_id", "hit_log_id", "banner_id", "pre_rank", *names]
        table = pq.read_table(path, columns=columns)
        if table.num_rows == 0:
            continue
        scores = np.asarray(model.predict(matrix(table, names)), dtype=np.float64)
        request_ids = np.asarray(table["request_id"].to_pylist(), dtype=object)
        hit_log_ids = table["hit_log_id"].combine_chunks().to_numpy(zero_copy_only=False)
        banner_ids = table["banner_id"].combine_chunks().to_numpy(zero_copy_only=False)
        pre_ranks = table["pre_rank"].combine_chunks().to_numpy(zero_copy_only=False)
        starts = np.r_[0, np.flatnonzero(request_ids[1:] != request_ids[:-1]) + 1]
        ends = np.r_[starts[1:], table.num_rows]
        for start, end in zip(starts, ends):
            rows.append(
                {
                    "HitLogID": int(hit_log_ids[start]),
                    "BannerID": ordered_banner_ids(
                        scores[start:end],
                        pre_ranks[start:end],
                        banner_ids[start:end],
                        top_k=args.top_k,
                    ),
                }
            )

    rows.sort(key=lambda row: int(row["HitLogID"]))
    schema = pa.schema(
        [
            pa.field("HitLogID", pa.uint64(), nullable=False),
            pa.field("BannerID", pa.list_(pa.uint64()), nullable=False),
        ]
    )
    output = pa.Table.from_pylist(rows, schema=schema)
    with atomic_output_path(args.output) as temporary:
        pq.write_table(output, temporary, compression="zstd")
    write_output_manifest(
        args.output,
        stage="export_ranker_ranking",
        artifact_version=f"catboost_top{args.top_k}_v1",
        config_sha256=str(metadata.get("config_sha256") or "unknown"),
        inputs=[
            fingerprint_file(model_path),
            fingerprint_file(metadata_path),
            *(fingerprint_file(path) for path in feature_paths),
        ],
        rows=output.num_rows,
        schema=str(schema),
        scope=str(metadata["scope"]),
    )
    report = {
        "run": str(args.run),
        "split": args.split,
        "top_k": args.top_k,
        "requests": output.num_rows,
        "min_items": min(len(row["BannerID"]) for row in rows),
        "max_items": max(len(row["BannerID"]) for row in rows),
        "output": str(args.output),
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
