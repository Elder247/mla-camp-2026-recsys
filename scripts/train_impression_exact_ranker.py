#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from audit_exact_query_history import fill, metric, prediction_map, request_rows
from audit_impression_exact_query import add_prior, blend_order, load_model_ranks, load_stats


FEATURE_NAMES = (
    "log_shows",
    "log_clicks",
    "log_value",
    "log_shows7",
    "log_clicks7",
    "log_shows42",
    "log_clicks42",
    "ctr20",
    "mean_click_value",
    "age_days",
    "shows7_share",
    "clicks7_share",
    "shows42_share",
    "clicks42_share",
    "log_pool_size",
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a temporal exact-impression CatBoost ranker")
    parser.add_argument("--impressions", type=Path, required=True)
    parser.add_argument("--train-requests", type=Path, required=True)
    parser.add_argument("--holdout-requests", type=Path, required=True)
    parser.add_argument("--fallback", type=Path, required=True)
    parser.add_argument("--model-candidates", type=Path)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=400)
    return parser.parse_args()


def features(values: dict[str, float], show_time: int, pool_size: int) -> list[float]:
    shows = values["shows"]
    clicks = values["clicks"]
    shows7 = values["shows7"]
    clicks7 = values["clicks7"]
    shows42 = values["shows42"]
    clicks42 = values["clicks42"]
    return [
        math.log1p(shows),
        math.log1p(clicks),
        math.log1p(values["value"]),
        math.log1p(shows7),
        math.log1p(clicks7),
        math.log1p(shows42),
        math.log1p(clicks42),
        clicks / (shows + 20.0),
        values["value"] / max(clicks, 1.0),
        max(float(show_time) - values["last"], 0.0) / 86_400.0,
        shows7 / max(shows, 1.0),
        clicks7 / max(clicks, 1.0),
        shows42 / max(shows, 1.0),
        clicks42 / max(clicks, 1.0),
        math.log1p(pool_size),
    ]


def training_matrix(requests: list[dict], stats: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    matrix = []
    labels = []
    groups = []
    kept_groups = positive_groups = 0
    for row in requests:
        values = stats.get(str(row["query"]), {})
        if not values:
            continue
        truth = {
            int(banner_id): float(cost)
            for banner_id, cost in zip(row["clicked_banner_ids"], row["clicked_source_costs"])
        }
        if not set(values).intersection(truth):
            continue
        positive_groups += 1
        group_id = kept_groups
        kept_groups += 1
        for banner_id, candidate in values.items():
            matrix.append(features(candidate, int(row["show_time"] or 0), len(values)))
            labels.append(math.log1p(truth.get(int(banner_id), 0.0) / 1_000_000.0))
            groups.append(group_id)
    return (
        np.asarray(matrix, dtype=np.float32),
        np.asarray(labels, dtype=np.float32),
        np.asarray(groups, dtype=np.int64),
        {"requests": len(requests), "positive_pool_groups": positive_groups, "fit_groups": kept_groups, "rows": len(labels)},
    )


def ranked_predictions(model, requests: list[dict], stats: dict) -> dict[int, list[int]]:
    result = {}
    for row in requests:
        values = stats.get(str(row["query"]), {})
        banners = list(values)
        if not banners:
            result[int(row["hit_log_id"])] = []
            continue
        matrix = np.asarray(
            [features(values[banner_id], int(row["show_time"] or 0), len(values)) for banner_id in banners],
            dtype=np.float32,
        )
        scores = np.asarray(model.predict(matrix), dtype=np.float64)
        order = np.lexsort((np.asarray(banners, dtype=np.int64), -scores))
        result[int(row["hit_log_id"])] = [banners[index] for index in order]
    return result


def main() -> int:
    args = arguments()
    from catboost import CatBoostRanker, Pool

    train_rows = request_rows(args.train_requests)
    holdout_rows = request_rows(args.holdout_requests)
    all_queries = {str(row["query"]) for row in train_rows + holdout_rows}
    base_stats = load_stats(args.impressions, all_queries)
    x_train, y_train, group_train, train_meta = training_matrix(train_rows, base_stats)
    if not len(y_train):
        raise RuntimeError("No positive exact-query groups for training")
    pool = Pool(x_train, label=y_train, group_id=group_train, feature_names=list(FEATURE_NAMES))
    model = CatBoostRanker(
        loss_function="QueryRMSE",
        iterations=args.iterations,
        depth=8,
        learning_rate=0.07,
        l2_leaf_reg=5.0,
        random_seed=42,
        task_type="CPU",
        verbose=max(args.iterations // 10, 1),
        allow_writing_files=False,
    )
    model.fit(pool)
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(args.model_output))

    holdout_stats = load_stats(args.impressions, {str(row["query"]) for row in holdout_rows})
    add_prior(holdout_stats, train_rows)
    ranker_orders = ranked_predictions(model, holdout_rows, holdout_stats)
    fallback = prediction_map(args.fallback)
    model_ranks = load_model_ranks(args.model_candidates)
    temporal = sorted(holdout_rows, key=lambda row: int(row["show_time"] or 0))
    midpoint = len(temporal) // 2
    splits = {"early": temporal[:midpoint], "late": temporal[midpoint:], "full": temporal}
    baseline = {name: metric(rows, fallback, 50) for name, rows in splits.items()}
    screens = []
    for model_weight in ((0.0, 0.25, 0.5, 0.75) if model_ranks else (0.0,)):
        for prefix in (10, 20, 30, 40, 50):
            prediction = {}
            for row in holdout_rows:
                hit_log_id = int(row["hit_log_id"])
                exact = blend_order(
                    ranker_orders[hit_log_id],
                    model_ranks.get(str(row["request_id"]), {}),
                    model_weight,
                )[:prefix]
                prediction[hit_log_id] = fill(exact, fallback[hit_log_id], 50)
            screens.append(
                {
                    "model_weight": model_weight,
                    "prefix": prefix,
                    "splits": {name: metric(rows, prediction, 50) for name, rows in splits.items()},
                }
            )
    robust = [
        screen
        for screen in screens
        if all(
            screen["splits"][name]["source_cost_recall"] >= baseline[name]["source_cost_recall"]
            for name in ("early", "late")
        )
    ]
    selected = max(robust or screens, key=lambda row: row["splits"]["full"]["source_cost_recall"])
    report = {
        "version": 1,
        "feature_names": FEATURE_NAMES,
        "train": train_meta,
        "baseline": baseline,
        "selected_robust": selected,
        "screens": screens,
        "model": str(args.model_output),
    }
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
