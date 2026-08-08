#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from catboost import CatBoostRanker

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mla_recsys.data import read_request_parquet  # noqa: E402
from mla_recsys.metrics import MISS_RANK, recall_metrics, truth_pairs  # noqa: E402


def parse_grid(value: str) -> list[float]:
    values = sorted({float(item) for item in value.split(",") if item})
    if not values or values[0] < 0.0 or values[-1] > 1.0:
        raise ValueError("alpha grid must contain values in [0, 1]")
    return values


def matrix(table: object, names: list[str]) -> np.ndarray:
    return np.column_stack(
        [table[name].combine_chunks().to_numpy(zero_copy_only=False) for name in names]
    ).astype(np.float32, copy=False)


def normalized(values: np.ndarray) -> np.ndarray:
    low = float(values.min())
    span = float(values.max()) - low
    if span <= 0.0:
        return np.zeros_like(values, dtype=np.float64)
    return (values.astype(np.float64, copy=False) - low) / span


def rank_positions(scores: np.ndarray, pre_rank: np.ndarray, banners: np.ndarray) -> np.ndarray:
    order = np.lexsort((banners, pre_rank, -scores))
    positions = np.empty(len(order), dtype=np.int32)
    positions[order] = np.arange(1, len(order) + 1, dtype=np.int32)
    return positions


def main() -> int:
    started = time.monotonic()
    parser = argparse.ArgumentParser(
        description="Tune a cheap CatBoost/RRF blend on an existing temporal holdout"
    )
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--alphas", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    alphas = parse_grid(args.alphas)
    metadata = json.loads(
        (args.run / "models" / "catboost.json").read_text(encoding="utf-8")
    )
    feature_names = list(metadata["feature_names"])
    model = CatBoostRanker()
    model.load_model(str(args.run / "models" / "catboost.cbm"))
    requests = read_request_parquet(args.run / "data" / "holdout_requests.parquet")
    truth = truth_pairs(requests)
    truth_by_request: dict[str, set[int]] = defaultdict(set)
    for request_id, banner_id in truth:
        truth_by_request[request_id].add(banner_id)

    found: dict[tuple[str, float], dict[tuple[str, int], int]] = {
        (method, alpha): {}
        for method in ("rank_linear", "score_minmax")
        for alpha in alphas
    }
    columns = [
        "request_id",
        "banner_id",
        "pre_rank",
        "rrf_score",
        *feature_names,
    ]
    # rrf_score is itself a configured feature in current runs; de-duplicate it.
    columns = list(dict.fromkeys(columns))
    processed_requests = 0
    for path in sorted((args.run / "features" / "holdout").glob("part-*.parquet")):
        table = pq.read_table(path, columns=columns)
        if table.num_rows == 0:
            continue
        catboost = np.asarray(model.predict(matrix(table, feature_names)), dtype=np.float64)
        request_ids = np.asarray(table["request_id"].to_pylist(), dtype=object)
        banners = table["banner_id"].combine_chunks().to_numpy(zero_copy_only=False).astype(np.int64)
        pre_rank = table["pre_rank"].combine_chunks().to_numpy(zero_copy_only=False).astype(np.int32)
        rrf = table["rrf_score"].combine_chunks().to_numpy(zero_copy_only=False).astype(np.float64)
        starts = np.r_[0, np.flatnonzero(request_ids[1:] != request_ids[:-1]) + 1]
        ends = np.r_[starts[1:], len(request_ids)]
        for start, end in zip(starts, ends):
            request_id = str(request_ids[start])
            clicked = truth_by_request.get(request_id)
            if not clicked:
                continue
            group_banners = banners[start:end]
            group_pre_rank = pre_rank[start:end]
            group_catboost = catboost[start:end]
            group_rrf = rrf[start:end]
            cb_positions = rank_positions(group_catboost, group_pre_rank, group_banners)
            cb_norm = normalized(group_catboost)
            rrf_norm = normalized(group_rrf)
            for alpha in alphas:
                rank_score = -(
                    alpha * cb_positions.astype(np.float64)
                    + (1.0 - alpha) * group_pre_rank.astype(np.float64)
                )
                score = alpha * cb_norm + (1.0 - alpha) * rrf_norm
                for method, blend_score in (
                    ("rank_linear", rank_score),
                    ("score_minmax", score),
                ):
                    positions = rank_positions(blend_score, group_pre_rank, group_banners)
                    target = found[(method, alpha)]
                    for index, banner_id in enumerate(group_banners):
                        pair = (request_id, int(banner_id))
                        if int(banner_id) in clicked:
                            target[pair] = int(positions[index])
            processed_requests += 1

    results = []
    for (method, alpha), ranks in found.items():
        records = [
            {
                "rank": int(ranks.get(pair, MISS_RANK)),
                "source_cost": source_cost,
            }
            for pair, source_cost in truth.items()
        ]
        metrics = recall_metrics(records, [50, 100, 500])
        results.append({"method": method, "alpha": alpha, "metrics": metrics})
    results.sort(
        key=lambda item: (
            -item["metrics"]["50"]["sourcecost_recall"],
            -item["metrics"]["50"]["recall"],
        )
    )
    report = {
        "run": str(args.run),
        "requests": processed_requests,
        "clicks": len(truth),
        "wall_seconds": time.monotonic() - started,
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
        "best": results[0],
        "results": results,
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
