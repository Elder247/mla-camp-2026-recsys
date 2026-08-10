#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from audit_exact_query_history import (
    compare,
    deduplicated_prefix,
    fill,
    metric,
    order_rows,
    prediction_map,
    project_normalize,
    request_rows,
)


def query_key(value: object) -> str:
    normalized = project_normalize(value)
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:20] if normalized else ""


def user_key(value: object) -> str:
    return "" if value is None or int(value) == 0 else str(int(value))


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit normalized query and user click history")
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--prior-requests", type=Path)
    parser.add_argument("--test-requests", type=Path)
    parser.add_argument("--fallback", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def update(by_entity: dict[str, dict[int, list[float]]], key: str, banner_id: int, cost: float, show_time: int) -> None:
    if not key:
        return
    stats = by_entity[key].setdefault(banner_id, [0.0, 0.0, 0.0])
    stats[0] += 1.0
    stats[1] += cost
    stats[2] = max(stats[2], show_time)


def scan_events(path: Path, query_keys: set[str], user_keys: set[str]) -> tuple[dict, dict, dict]:
    queries: dict[str, dict[int, list[float]]] = defaultdict(dict)
    users: dict[str, dict[int, list[float]]] = defaultdict(dict)
    rows = query_matches = user_matches = 0
    minimum_time = None
    maximum_time = 0
    parquet = pq.ParquetFile(path)
    columns = ["show_time", "banner_id", "query_key", "user_key", "source_cost"]
    for batch in parquet.iter_batches(batch_size=131_072, columns=columns):
        values = batch.to_pydict()
        for show_value, banner_value, query_value, user_value, cost_value in zip(
            values["show_time"], values["banner_id"], values["query_key"], values["user_key"], values["source_cost"]
        ):
            rows += 1
            show_time = int(show_value)
            minimum_time = show_time if minimum_time is None else min(minimum_time, show_time)
            maximum_time = max(maximum_time, show_time)
            banner_id = int(banner_value)
            cost = float(cost_value or 0.0)
            query_value = str(query_value or "")
            user_value = str(user_value or "")
            if query_value in query_keys:
                update(queries, query_value, banner_id, cost, show_time)
                query_matches += 1
            if user_value in user_keys and user_value != "0":
                update(users, user_value, banner_id, cost, show_time)
                user_matches += 1
    return queries, users, {
        "rows_scanned": rows,
        "query_event_matches": query_matches,
        "user_event_matches": user_matches,
        "minimum_show_time": minimum_time,
        "maximum_show_time": maximum_time,
    }


def add_requests(queries: dict, users: dict, rows: list[dict]) -> dict:
    maximum_time = 0
    clicks = 0
    for row in rows:
        show_time = int(row["show_time"] or 0)
        maximum_time = max(maximum_time, show_time)
        qkey = query_key(row["query"])
        ukey = user_key(row.get("crypta_id_v2"))
        for banner_value, cost_value in zip(row["clicked_banner_ids"], row["clicked_source_costs"]):
            banner_id = int(banner_value)
            cost = float(cost_value)
            update(queries, qkey, banner_id, cost, show_time)
            update(users, ukey, banner_id, cost, show_time)
            clicks += 1
    return {"requests": len(rows), "clicks": clicks, "maximum_show_time": maximum_time}


def ordered(stats: dict) -> dict[str, dict[str, list[tuple[int, list[float]]]]]:
    return {
        mode: {key: order_rows(values, mode) for key, values in stats.items()}
        for mode in ("source_cost_sum", "click_count", "recency", "rrf")
    }


def build_predictions(
    rows: list[dict],
    fallback: dict[int, list[int]],
    query_rankings: dict[str, list[tuple[int, list[float]]]],
    user_rankings: dict[str, list[tuple[int, list[float]]]],
    exact_size: int,
    user_size: int,
) -> tuple[dict[int, list[int]], list[int], list[int]]:
    result = {}
    exact_lengths = []
    user_lengths = []
    for row in rows:
        hit_log_id = int(row["hit_log_id"])
        cutoff = int(row["show_time"] or 0)
        exact = deduplicated_prefix(query_rankings.get(query_key(row["query"]), ()), exact_size, cutoff)
        user = deduplicated_prefix(user_rankings.get(user_key(row.get("crypta_id_v2")), ()), user_size, cutoff)
        exact_lengths.append(len(exact))
        user_lengths.append(len(user))
        result[hit_log_id] = fill(exact + user, fallback[hit_log_id])
    return result, exact_lengths, user_lengths


def main() -> int:
    args = arguments()
    requests = request_rows(args.requests)
    prior = request_rows(args.prior_requests) if args.prior_requests else []
    test = request_rows(args.test_requests) if args.test_requests else []
    target = requests + test
    target_queries = {query_key(row["query"]) for row in target}
    target_users = {user_key(row.get("crypta_id_v2")) for row in target} - {""}
    queries, users, event_meta = scan_events(args.events, target_queries, target_users)
    prior_meta = add_requests(queries, users, prior)
    query_orders = ordered(queries)
    user_orders = ordered(users)
    fallback = prediction_map(args.fallback)
    ordered_requests = sorted(requests, key=lambda row: int(row["show_time"] or 0))
    midpoint = len(ordered_requests) // 2
    splits = {"early": ordered_requests[:midpoint], "late": ordered_requests[midpoint:], "full": ordered_requests}
    baseline = {name: metric(rows, fallback, 50) for name, rows in splits.items()}

    exact_screens = []
    exact_predictions = {}
    for mode in ("source_cost_sum", "click_count", "rrf"):
        for size in (10, 20, 30, 40, 50):
            candidate, lengths, _ = build_predictions(
                requests, fallback, query_orders[mode], {}, size, 0
            )
            key = f"{mode}:e{size}"
            exact_predictions[key] = candidate
            exact_screens.append({
                "key": key,
                "mode": mode,
                "exact_size": size,
                "coverage": sum(value > 0 for value in lengths) / len(lengths),
                "mean_length": statistics.fmean(lengths),
                "median_length": statistics.median(lengths),
                "splits": {name: metric(rows, candidate, 50) for name, rows in splits.items()},
            })

    selected_exact = max(
        exact_screens, key=lambda row: row["splits"]["early"]["source_cost_recall"]
    )
    context_screens = []
    for user_mode in ("source_cost_sum", "click_count", "rrf"):
        for user_size in (5, 10, 15):
            candidate, exact_lengths, user_lengths = build_predictions(
                requests,
                fallback,
                query_orders[selected_exact["mode"]],
                user_orders[user_mode],
                int(selected_exact["exact_size"]),
                user_size,
            )
            context_screens.append({
                "exact": selected_exact["key"],
                "user_mode": user_mode,
                "user_size": user_size,
                "user_coverage": sum(value > 0 for value in user_lengths) / len(user_lengths),
                "mean_user_length": statistics.fmean(user_lengths),
                "splits": {name: metric(rows, candidate, 50) for name, rows in splits.items()},
                "comparison_to_fallback": compare(requests, candidate, fallback),
            })

    minimum_request_time = min(int(row["show_time"] or 0) for row in requests)
    report = {
        "inputs": {"events": str(args.events), "requests": str(args.requests), "fallback": str(args.fallback)},
        "events": event_meta,
        "prior_requests": prior_meta,
        "leakage_contract": {
            "maximum_available_history_time": max(event_meta["maximum_show_time"], prior_meta["maximum_show_time"]),
            "minimum_temporal_request_show_time": minimum_request_time,
            "all_available_history_precedes_temporal_requests": max(event_meta["maximum_show_time"], prior_meta["maximum_show_time"]) < minimum_request_time,
        },
        "coverage": {
            "temporal_query": sum(query_key(row["query"]) in queries for row in requests) / len(requests),
            "test_query": sum(query_key(row["query"]) in queries for row in test) / len(test) if test else 0.0,
            "temporal_user": sum(user_key(row.get("crypta_id_v2")) in users for row in requests) / len(requests),
            "test_user": sum(user_key(row.get("crypta_id_v2")) in users for row in test) / len(test) if test else 0.0,
        },
        "baseline": baseline,
        "exact_screens": exact_screens,
        "selected_exact_on_early": selected_exact,
        "context_screens": context_screens,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
