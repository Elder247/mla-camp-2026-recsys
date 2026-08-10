#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from audit_exact_query_history import fill, metric, prediction_map, request_rows
from audit_impression_exact_query import add_prior, load_stats


FEATURES = (
    "h_present",
    "h_rank_reciprocal",
    "h_rank_fraction",
    "h_score",
    "exact_present",
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
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train hN + exact-impression mixed CatBoost ranker")
    parser.add_argument("--impressions", type=Path, required=True)
    parser.add_argument("--train-requests", type=Path, required=True)
    parser.add_argument("--holdout-requests", type=Path, required=True)
    parser.add_argument("--train-history", type=Path, required=True)
    parser.add_argument("--holdout-history", type=Path, required=True)
    parser.add_argument("--fallback", type=Path, required=True)
    parser.add_argument("--history-limit", type=int, default=500)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    return parser.parse_args()


def history_frame(path: Path, limit: int) -> pd.DataFrame:
    files = sorted(path.rglob("part-*.parquet"))
    frames = [
        pq.read_table(
            file_path,
            columns=["request_id", "hit_log_id", "banner_id", "source_rank", "source_score"],
        ).to_pandas()
        for file_path in files
    ]
    frame = pd.concat(frames, ignore_index=True)
    frame = frame[frame["source_rank"] <= limit].copy()
    return frame.rename(columns={"source_rank": "h_rank", "source_score": "h_score"})


def exact_frame(rows: list[dict], stats: dict) -> pd.DataFrame:
    records = []
    for row in rows:
        request_id = str(row["request_id"])
        for banner_id, values in stats.get(str(row["query"]), {}).items():
            records.append(
                {
                    "request_id": request_id,
                    "banner_id": int(banner_id),
                    **values,
                }
            )
    return pd.DataFrame.from_records(records)


def make_pool(
    rows: list[dict],
    stats: dict,
    history_path: Path,
    history_limit: int,
    *,
    positive_only: bool,
) -> tuple[pd.DataFrame, dict]:
    history = history_frame(history_path, history_limit)
    exact = exact_frame(rows, stats)
    pool = history.merge(exact, on=["request_id", "banner_id"], how="outer")
    meta = pd.DataFrame.from_records(
        [
            {
                "request_id": str(row["request_id"]),
                "hit_log_id_meta": int(row["hit_log_id"]),
                "show_time": int(row["show_time"] or 0),
                "group_order": index,
            }
            for index, row in enumerate(rows)
        ]
    )
    truth = pd.DataFrame.from_records(
        [
            {
                "request_id": str(row["request_id"]),
                "banner_id": int(banner_id),
                "raw_cost": float(cost),
            }
            for row in rows
            for banner_id, cost in zip(row["clicked_banner_ids"], row["clicked_source_costs"])
        ]
    )
    pool = pool.merge(meta, on="request_id", how="inner")
    pool = pool.merge(truth, on=["request_id", "banner_id"], how="left")
    pool["raw_cost"] = pool["raw_cost"].fillna(0.0)
    if positive_only:
        positive_requests = pool.loc[pool["raw_cost"] > 0, "request_id"].unique()
        pool = pool[pool["request_id"].isin(positive_requests)].copy()

    h_rank = pool["h_rank"].fillna(history_limit + 1).astype(np.float32)
    pool["h_present"] = pool["h_rank"].notna().astype(np.float32)
    pool["h_rank_reciprocal"] = pool["h_present"] / h_rank
    pool["h_rank_fraction"] = h_rank / float(history_limit + 1)
    pool["h_score"] = pool["h_score"].fillna(-10.0).astype(np.float32)
    pool["exact_present"] = pool["shows"].notna().astype(np.float32)
    for column in ("shows", "clicks", "value", "last", "shows7", "clicks7", "shows42", "clicks42"):
        pool[column] = pool[column].fillna(0.0).astype(np.float64)
    pool["log_shows"] = np.log1p(pool["shows"])
    pool["log_clicks"] = np.log1p(pool["clicks"])
    pool["log_value"] = np.log1p(pool["value"])
    pool["log_shows7"] = np.log1p(pool["shows7"])
    pool["log_clicks7"] = np.log1p(pool["clicks7"])
    pool["log_shows42"] = np.log1p(pool["shows42"])
    pool["log_clicks42"] = np.log1p(pool["clicks42"])
    pool["ctr20"] = pool["clicks"] / (pool["shows"] + 20.0)
    pool["mean_click_value"] = pool["value"] / pool["clicks"].clip(lower=1.0)
    pool["age_days"] = np.maximum(pool["show_time"] - pool["last"], 0.0) / 86_400.0
    pool["shows7_share"] = pool["shows7"] / pool["shows"].clip(lower=1.0)
    pool["clicks7_share"] = pool["clicks7"] / pool["clicks"].clip(lower=1.0)
    pool["shows42_share"] = pool["shows42"] / pool["shows"].clip(lower=1.0)
    pool["clicks42_share"] = pool["clicks42"] / pool["clicks"].clip(lower=1.0)
    pool["label"] = np.log1p(pool["raw_cost"] / 1_000_000.0)
    pool.sort_values(["group_order", "h_rank", "banner_id"], inplace=True, kind="mergesort")
    pool["group_id"] = pd.factorize(pool["request_id"], sort=False)[0]
    return pool, {
        "requests": len(rows),
        "groups": int(pool["group_id"].nunique()),
        "rows": len(pool),
        "history_rows": len(history),
        "exact_rows": len(exact),
    }


def prediction_map_from_pool(pool: pd.DataFrame, scores: np.ndarray) -> dict[int, list[int]]:
    ranked = pool[["hit_log_id_meta", "banner_id", "h_rank"]].copy()
    ranked["score"] = scores
    ranked.sort_values(
        ["hit_log_id_meta", "score", "h_rank", "banner_id"],
        ascending=[True, False, True, True],
        inplace=True,
        kind="mergesort",
    )
    return {
        int(hit_log_id): [int(value) for value in group["banner_id"].tolist()]
        for hit_log_id, group in ranked.groupby("hit_log_id_meta", sort=False)
    }


def main() -> int:
    args = arguments()
    from catboost import CatBoostRanker, Pool

    train_rows = request_rows(args.train_requests)
    holdout_rows = request_rows(args.holdout_requests)
    base_stats = load_stats(args.impressions, {str(row["query"]) for row in train_rows})
    train_pool, train_meta = make_pool(
        train_rows,
        base_stats,
        args.train_history,
        args.history_limit,
        positive_only=True,
    )
    cat_train = Pool(
        train_pool[list(FEATURES)].to_numpy(dtype=np.float32),
        label=train_pool["label"].to_numpy(dtype=np.float32),
        group_id=train_pool["group_id"].to_numpy(dtype=np.int64),
        feature_names=list(FEATURES),
    )
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
    model.fit(cat_train)
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(args.model_output))

    holdout_stats = load_stats(args.impressions, {str(row["query"]) for row in holdout_rows})
    add_prior(holdout_stats, train_rows)
    validation_pool, validation_meta = make_pool(
        holdout_rows,
        holdout_stats,
        args.holdout_history,
        args.history_limit,
        positive_only=False,
    )
    scores = np.asarray(
        model.predict(validation_pool[list(FEATURES)].to_numpy(dtype=np.float32)),
        dtype=np.float64,
    )
    model_order = prediction_map_from_pool(validation_pool, scores)
    fallback = prediction_map(args.fallback)
    temporal = sorted(holdout_rows, key=lambda row: int(row["show_time"] or 0))
    midpoint = len(temporal) // 2
    splits = {"early": temporal[:midpoint], "late": temporal[midpoint:], "full": temporal}
    baseline = {name: metric(rows, fallback, 50) for name, rows in splits.items()}
    screens = []
    for prefix in (10, 20, 30, 40, 50):
        prediction = {}
        for row in holdout_rows:
            hit_log_id = int(row["hit_log_id"])
            prediction[hit_log_id] = fill(model_order.get(hit_log_id, [])[:prefix], fallback[hit_log_id], 50)
        screens.append(
            {
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
        "features": FEATURES,
        "history_limit": args.history_limit,
        "iterations": args.iterations,
        "train": train_meta,
        "validation": validation_meta,
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
