#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from audit_exact_query_history import compare, fill, metric, prediction_map, request_rows


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Temporal audit for exact-query impression history")
    parser.add_argument("--impressions", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--prior-requests", type=Path, required=True)
    parser.add_argument("--fallback", type=Path, required=True)
    parser.add_argument("--model-candidates", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_model_ranks(path: Path | None) -> dict[str, dict[int, int]]:
    if path is None:
        return {}
    files = sorted(path.rglob("*.parquet")) if path.is_dir() else [path]
    result: dict[str, dict[int, int]] = defaultdict(dict)
    for file_path in files:
        for batch in pq.ParquetFile(file_path).iter_batches(
            columns=["request_id", "banner_id", "source_rank"], batch_size=131_072
        ):
            for row in batch.to_pylist():
                result[str(row["request_id"])][int(row["banner_id"])] = int(row["source_rank"])
    return result


def blend_order(exact: list[int], model: dict[int, int], model_weight: float) -> list[int]:
    if not exact or model_weight <= 0.0:
        return exact
    denominator = float(max(len(exact), 1))
    exact_ranks = {banner_id: rank for rank, banner_id in enumerate(exact, start=1)}
    return sorted(
        exact,
        key=lambda banner_id: (
            (1.0 - model_weight) * (exact_ranks[banner_id] / denominator)
            + model_weight * (min(model.get(banner_id, 501), 501) / 501.0),
            model.get(banner_id, 501),
            exact_ranks[banner_id],
            banner_id,
        ),
    )


def load_stats(path: Path, target_queries: set[str]) -> dict[str, dict[int, dict[str, float]]]:
    result: dict[str, dict[int, dict[str, float]]] = defaultdict(dict)
    for batch in pq.ParquetFile(path).iter_batches(batch_size=131_072):
        for row in batch.to_pylist():
            query = str(row["search_query"])
            if query not in target_queries:
                continue
            result[query][int(row["banner_id"])] = {
                "shows": float(row["show_count"] or 0),
                "clicks": float(row["click_count"] or 0),
                "value": float(row["click_source_cost_sum"] or 0),
                "last": float(row["last_show_time"] or 0),
                "shows7": float(row["show_count_7d"] or 0),
                "clicks7": float(row["click_count_7d"] or 0),
                "shows42": float(row["show_count_42d"] or 0),
                "clicks42": float(row["click_count_42d"] or 0),
            }
    return result


def add_prior(stats: dict, rows: list[dict]) -> None:
    for row in rows:
        query = str(row["query"])
        show_time = int(row["show_time"] or 0)
        for banner, cost in zip(row["clicked_banner_ids"], row["clicked_source_costs"]):
            values = stats.setdefault(query, {}).setdefault(
                int(banner),
                {"shows": 0.0, "clicks": 0.0, "value": 0.0, "last": 0.0, "shows7": 0.0, "clicks7": 0.0, "shows42": 0.0, "clicks42": 0.0},
            )
            values["shows"] += 1.0
            values["clicks"] += 1.0
            values["value"] += float(cost)
            values["last"] = max(values["last"], show_time)
            values["shows7"] += 1.0
            values["clicks7"] += 1.0
            values["shows42"] += 1.0
            values["clicks42"] += 1.0


def order(values: dict[int, dict[str, float]], mode: str) -> list[int]:
    rows = list(values.items())
    if mode == "shows":
        key = lambda row: (-row[1]["shows"], -row[1]["clicks"], -row[1]["last"], row[0])
    elif mode == "clicks":
        key = lambda row: (-row[1]["clicks"], -row[1]["shows"], -row[1]["last"], row[0])
    elif mode == "ctr":
        key = lambda row: (-(row[1]["clicks"] / (row[1]["shows"] + 20.0)), -row[1]["clicks"], -row[1]["shows"], row[0])
    elif mode == "value":
        key = lambda row: (-row[1]["value"], -row[1]["clicks"], -row[1]["shows"], row[0])
    elif mode == "recency":
        key = lambda row: (-row[1]["last"], -row[1]["clicks"], -row[1]["shows"], row[0])
    elif mode in {"shows7", "clicks7", "shows42", "clicks42"}:
        secondary = "clicks7" if mode == "shows7" else "shows7" if mode == "clicks7" else "clicks42" if mode == "shows42" else "shows42"
        key = lambda row: (-row[1][mode], -row[1][secondary], -row[1]["last"], row[0])
    else:
        raise ValueError(mode)
    return [banner_id for banner_id, _ in sorted(rows, key=key)]


def main() -> int:
    args = arguments()
    requests = request_rows(args.requests)
    prior = request_rows(args.prior_requests)
    stats = load_stats(args.impressions, {str(row["query"]) for row in requests})
    add_prior(stats, prior)
    fallback = prediction_map(args.fallback)
    model_ranks = load_model_ranks(args.model_candidates)
    temporal = sorted(requests, key=lambda row: int(row["show_time"] or 0))
    midpoint = len(temporal) // 2
    splits = {"early": temporal[:midpoint], "late": temporal[midpoint:], "full": temporal}
    baseline = {name: metric(rows, fallback, 50) for name, rows in splits.items()}
    exact_pool_predictions = {}
    top50_oracle = {}
    for row in requests:
        hit_log_id = int(row["hit_log_id"])
        pool = list(stats.get(str(row["query"]), {}))
        exact_pool_predictions[hit_log_id] = pool
        pool_set = set(pool)
        truth = sorted(
            (
                (int(banner_id), float(cost))
                for banner_id, cost in zip(row["clicked_banner_ids"], row["clicked_source_costs"])
                if int(banner_id) in pool_set
            ),
            key=lambda value: (-value[1], value[0]),
        )
        top50_oracle[hit_log_id] = fill([banner_id for banner_id, _ in truth], pool, 50)
    modes = ("shows", "clicks", "ctr", "value", "recency", "shows7", "clicks7", "shows42", "clicks42")
    ordered = {mode: {query: order(values, mode) for query, values in stats.items()} for mode in modes}
    screens = []
    model_weights = (0.0, 0.25, 0.5, 0.75, 1.0) if model_ranks else (0.0,)
    for mode in modes:
        for model_weight in model_weights:
            for prefix in (10, 20, 30, 40, 50):
                candidate = {}
                lengths = []
                model_hits = []
                for row in requests:
                    hit_log_id = int(row["hit_log_id"])
                    exact_candidates = ordered[mode].get(str(row["query"]), [])
                    ranks = model_ranks.get(str(row["request_id"]), {})
                    exact = blend_order(exact_candidates, ranks, model_weight)[:prefix]
                    lengths.append(len(exact))
                    model_hits.append(sum(banner_id in ranks for banner_id in exact))
                    candidate[hit_log_id] = fill(exact, fallback[hit_log_id], 50)
                screens.append(
                    {
                        "mode": mode,
                        "model_weight": model_weight,
                        "prefix": prefix,
                        "coverage": sum(value > 0 for value in lengths) / len(lengths),
                        "mean_length": statistics.fmean(lengths),
                        "median_length": statistics.median(lengths),
                        "mean_model_hits": statistics.fmean(model_hits),
                        "splits": {name: metric(rows, candidate, 50) for name, rows in splits.items()},
                        "comparison_to_fallback": compare(requests, candidate, fallback),
                    }
                )
    selected = max(screens, key=lambda row: row["splits"]["early"]["source_cost_recall"])
    report = {
        "baseline": baseline,
        "exact_pool_membership": {name: metric(rows, exact_pool_predictions, 10_000) for name, rows in splits.items()},
        "exact_pool_top50_oracle": {name: metric(rows, top50_oracle, 50) for name, rows in splits.items()},
        "selected_on_early": selected,
        "screens": screens,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
