#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from audit_exact_query_history import fill, prediction_map, request_rows
from audit_impression_exact_query import add_prior, blend_order, load_model_ranks, load_stats, order


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize exact-query impression submissions")
    parser.add_argument("--impressions", type=Path, required=True)
    parser.add_argument("--test-requests", type=Path, required=True)
    parser.add_argument("--full-train-requests", type=Path, required=True)
    parser.add_argument("--fallback", type=Path, required=True)
    parser.add_argument("--model-candidates", type=Path)
    parser.add_argument("--valid-index", type=Path, required=True)
    parser.add_argument(
        "--variant",
        action="append",
        required=True,
        help="NAME:MODE:MODEL_WEIGHT:PREFIX; may be repeated",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = arguments()
    requests = request_rows(args.test_requests)
    queries = {str(row["query"]) for row in requests}
    stats = load_stats(args.impressions, queries)
    full_train_rows = request_rows(args.full_train_requests)
    add_prior(stats, full_train_rows)
    fallback = prediction_map(args.fallback)
    model_ranks = load_model_ranks(args.model_candidates)
    valid_ids = {
        int(value)
        for value in pq.read_table(args.valid_index, columns=["BannerID"])["BannerID"].to_pylist()
    }
    provenance_ids = {
        int(banner_id)
        for row in full_train_rows
        for banner_id in row["clicked_banner_ids"]
    }
    impression_ids = {
        int(banner_id)
        for values in stats.values()
        for banner_id in values
    }
    expected = {int(row["hit_log_id"]) for row in requests}
    if set(fallback) != expected or len(expected) != 10_000:
        raise ValueError("Request/fallback HitLogID contract is not exactly 10,000 rows")

    variants = []
    for spec in args.variant:
        name, mode, model_weight, prefix = spec.split(":")
        variants.append((name, mode, float(model_weight), int(prefix)))
    modes = {mode for _, mode, _, _ in variants}
    ordered = {mode: {query: order(values, mode) for query, values in stats.items()} for mode in modes}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = []

    for name, mode, model_weight, prefix in variants:
        output_rows = []
        exact_lengths = []
        model_hits = []
        outside_trainseen = set()
        for row in requests:
            hit_log_id = int(row["hit_log_id"])
            exact_pool = ordered[mode].get(str(row["query"]), [])
            ranks = model_ranks.get(str(row["request_id"]), {})
            exact = blend_order(exact_pool, ranks, model_weight)[:prefix]
            outside_trainseen.update(set(exact) - valid_ids)
            missing = set(exact) - valid_ids - provenance_ids - impression_ids
            if missing:
                raise ValueError(f"Exact candidates outside valid train-seen universe: {len(missing)}")
            banner_ids = fill(exact, fallback[hit_log_id], 50)
            if len(banner_ids) != 50 or len(set(banner_ids)) != 50:
                raise ValueError(f"Invalid top-50 for HitLogID {hit_log_id}")
            exact_lengths.append(len(exact))
            model_hits.append(sum(banner_id in ranks for banner_id in exact))
            output_rows.append({"HitLogID": hit_log_id, "BannerID": banner_ids})

        output = args.output_dir / f"test_top50_{name}.parquet"
        table = pa.Table.from_pylist(
            output_rows,
            schema=pa.schema(
                [
                    pa.field("HitLogID", pa.uint64(), nullable=False),
                    pa.field("BannerID", pa.list_(pa.uint64()), nullable=False),
                ]
            ),
        )
        pq.write_table(table, output, compression="zstd")
        reread = pq.read_table(output)
        if reread.num_rows != 10_000 or len(set(reread["HitLogID"].to_pylist())) != 10_000:
            raise ValueError("Written submission does not satisfy the 10,000-row contract")
        reports.append(
            {
                "name": name,
                "mode": mode,
                "model_weight": model_weight,
                "prefix": prefix,
                "path": str(output),
                "sha256": sha256(output),
                "coverage": sum(value > 0 for value in exact_lengths) / len(exact_lengths),
                "mean_exact_length": statistics.fmean(exact_lengths),
                "median_exact_length": statistics.median(exact_lengths),
                "mean_model_hits": statistics.fmean(model_hits),
                "exact_candidate_ids_outside_trainseen_index": len(outside_trainseen),
                "outside_trainseen_provenance": "official train_100m impressions or full validation click labels",
            }
        )

    manifest = {
        "version": 1,
        "impressions": str(args.impressions),
        "test_requests": str(args.test_requests),
        "full_train_requests": str(args.full_train_requests),
        "fallback": str(args.fallback),
        "model_candidates": str(args.model_candidates) if args.model_candidates else None,
        "valid_index": str(args.valid_index),
        "variants": reports,
    }
    manifest_path = args.output_dir / "impression_exact_query_materialization_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
