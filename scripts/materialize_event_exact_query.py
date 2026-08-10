#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from audit_event_history import ordered, query_key, scan_events
from audit_exact_query_history import fill, order_rows, prediction_map, request_rows


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize normalized exact-query prefix submissions")
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--test-requests", type=Path, required=True)
    parser.add_argument("--fallback", type=Path, required=True)
    parser.add_argument("--valid-index", type=Path, required=True)
    parser.add_argument("--raw-history", type=Path, required=True)
    parser.add_argument("--full-train-requests", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def raw_rankings(history_path: Path, full_train_path: Path, test_rows: list[dict]) -> dict[str, list[tuple[int, list[float]]]]:
    target_queries = {str(row["query"]) for row in test_rows}
    stats: dict[str, dict[int, list[float]]] = defaultdict(dict)
    parquet = pq.ParquetFile(history_path)
    columns = ["key_type", "search_query", "banner_id", "click_count", "source_cost_sum", "last_show_time"]
    for batch in parquet.iter_batches(batch_size=131_072, columns=columns):
        values = batch.to_pydict()
        for key_type, query, banner, clicks, cost_sum, last_time in zip(
            values["key_type"], values["search_query"], values["banner_id"], values["click_count"], values["source_cost_sum"], values["last_show_time"]
        ):
            query = str(query or "")
            if key_type != "query" or query not in target_queries:
                continue
            stats[query][int(banner)] = [float(clicks or 0), float(cost_sum or 0.0), float(last_time or 0)]
    for row in request_rows(full_train_path):
        query = str(row["query"])
        if query not in target_queries:
            continue
        show_time = int(row["show_time"] or 0)
        for banner, cost in zip(row["clicked_banner_ids"], row["clicked_source_costs"]):
            values = stats[query].setdefault(int(banner), [0.0, 0.0, 0.0])
            values[0] += 1.0
            values[1] += float(cost)
            values[2] = max(values[2], show_time)
    return {query: order_rows(values, "rrf") for query, values in stats.items()}


def main() -> int:
    args = arguments()
    requests = request_rows(args.test_requests)
    target_queries = {query_key(row["query"]) for row in requests}
    query_stats, _, event_meta = scan_events(args.events, target_queries, set())
    query_orders = ordered(query_stats)
    raw_orders = raw_rankings(args.raw_history, args.full_train_requests, requests)
    fallback = prediction_map(args.fallback)
    valid_ids = {
        int(value)
        for value in pq.read_table(args.valid_index, columns=["BannerID"])["BannerID"].to_pylist()
    }
    fallback_ids = {banner_id for values in fallback.values() for banner_id in values}
    variants = (
        ("rrf_e20", "rrf", 20),
        ("rrf_e30", "rrf", 30),
        ("source_cost_e30", "source_cost_sum", 30),
        ("raw_then_norm_rrf_e30", "raw_then_norm_rrf", 30),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    expected_hit_logs = {int(row["hit_log_id"]) for row in requests}
    if set(fallback) != expected_hit_logs:
        raise ValueError("Fallback HitLogID set differs from test requests")

    for name, mode, exact_size in variants:
        output_rows = []
        exact_lengths = []
        new_ids = set()
        outside_trainseen = set()
        for row in requests:
            hit_log_id = int(row["hit_log_id"])
            normalized = [
                int(banner_id)
                for banner_id, _ in query_orders["rrf" if mode == "raw_then_norm_rrf" else mode].get(query_key(row["query"]), ())
            ]
            if mode == "raw_then_norm_rrf":
                literal = [int(banner_id) for banner_id, _ in raw_orders.get(str(row["query"]), ())]
                exact = fill(literal, normalized, exact_size)
            else:
                exact = normalized[:exact_size]
            exact_lengths.append(len(exact))
            new_ids.update(set(exact) - fallback_ids)
            banner_ids = fill(exact, fallback[hit_log_id], 50)
            if len(banner_ids) != 50 or len(set(banner_ids)) != 50:
                raise ValueError(f"Invalid row {hit_log_id}: {len(banner_ids)} / {len(set(banner_ids))}")
            missing_new = set(exact) - valid_ids
            outside_trainseen.update(missing_new)
            output_rows.append({"HitLogID": hit_log_id, "BannerID": banner_ids})

        output = args.output_dir / f"test_top50_exact_query_{name}_v27.parquet"
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
        if reread.num_rows != 10_000:
            raise ValueError(f"Expected 10,000 rows, got {reread.num_rows}")
        if len(set(int(value) for value in reread["HitLogID"].to_pylist())) != 10_000:
            raise ValueError("HitLogID values are not unique")
        reports.append(
            {
                "name": name,
                "mode": mode,
                "exact_size": exact_size,
                "path": str(output),
                "sha256": file_sha256(output),
                "rows": reread.num_rows,
                "coverage": sum(value > 0 for value in exact_lengths) / len(exact_lengths),
                "mean_exact_length": statistics.fmean(exact_lengths),
                "median_exact_length": statistics.median(exact_lengths),
                "new_exact_candidate_ids_vs_fallback_union": len(new_ids),
                "exact_candidate_ids_outside_trainseen_index": len(outside_trainseen),
                "outside_trainseen_provenance": "official train/validation click_events",
            }
        )

    manifest = {
        "version": 1,
        "events": str(args.events),
        "test_requests": str(args.test_requests),
        "fallback": str(args.fallback),
        "valid_index": str(args.valid_index),
        "event_scan": event_meta,
        "variants": reports,
    }
    manifest_path = args.output_dir / "exact_query_materialization_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
