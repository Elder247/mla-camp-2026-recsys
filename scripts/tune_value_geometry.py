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
from mla_recsys.rank_blend import (  # noqa: E402
    rank_value_geometric_order,
    value_geometric_from_base_order,
)


def parse_float_grid(value: str) -> list[float]:
    return sorted({float(item) for item in value.split(",") if item})


def parse_int_grid(value: str) -> list[int]:
    values = sorted({int(item) for item in value.split(",") if item})
    if not values or values[0] <= 0:
        raise ValueError("top-n grid must contain positive integers")
    return values


def matrix(table: object, names: list[str]) -> np.ndarray:
    return np.column_stack(
        [table[name].combine_chunks().to_numpy(zero_copy_only=False) for name in names]
    ).astype(np.float32, copy=False)


def main() -> int:
    started = time.monotonic()
    parser = argparse.ArgumentParser(
        description="Tune a bounded geometric SourceCost prior on temporal features"
    )
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--catboost-weights", default="0.6")
    parser.add_argument("--exponents", default="0,0.02,0.05,0.08,0.1,0.15,0.2")
    parser.add_argument("--rerank-top-n", default="75,100,150,250,500,750")
    parser.add_argument("--source-cost-scale", type=float, default=1_000_000.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    weights = parse_float_grid(args.catboost_weights)
    exponents = parse_float_grid(args.exponents)
    top_ns = parse_int_grid(args.rerank_top_n)
    if not weights or weights[0] < 0.0 or weights[-1] > 1.0:
        raise ValueError("catboost weights must be in [0, 1]")
    if not exponents or exponents[0] < 0.0:
        raise ValueError("exponents must be non-negative")
    if args.source_cost_scale <= 0.0:
        raise ValueError("source-cost scale must be positive")

    metadata = json.loads(
        (args.run / "models" / "catboost.json").read_text(encoding="utf-8")
    )
    feature_names = list(metadata["feature_names"])
    model = CatBoostRanker()
    model.load_model(str(args.run / "models" / "catboost.cbm"))
    truth = truth_pairs(
        read_request_parquet(args.run / "data" / "holdout_requests.parquet")
    )
    clicked_by_request: dict[str, set[int]] = defaultdict(set)
    for request_id, banner_id in truth:
        clicked_by_request[request_id].add(banner_id)
    combinations = [
        (weight, exponent, top_n)
        for weight in weights
        for exponent in exponents
        for top_n in top_ns
        if exponent > 0.0 or top_n == max(top_ns)
    ]
    found = {combination: {} for combination in combinations}
    columns = list(
        dict.fromkeys(
            [
                "request_id",
                "hit_log_id",
                "banner_id",
                "pre_rank",
                "source_cost_raw",
                *feature_names,
            ]
        )
    )
    processed_requests = 0
    for path in sorted((args.run / "features" / "holdout").glob("part-*.parquet")):
        table = pq.read_table(path, columns=columns)
        if table.num_rows == 0:
            continue
        predictions = np.asarray(
            model.predict(matrix(table, feature_names)), dtype=np.float64
        )
        request_ids = np.asarray(table["request_id"].to_pylist(), dtype=object)
        hit_logs = table["hit_log_id"].combine_chunks().to_numpy(zero_copy_only=False)
        banners = table["banner_id"].combine_chunks().to_numpy(zero_copy_only=False)
        pre_ranks = table["pre_rank"].combine_chunks().to_numpy(zero_copy_only=False)
        costs = table["source_cost_raw"].combine_chunks().to_numpy(zero_copy_only=False)
        starts = np.r_[0, np.flatnonzero(request_ids[1:] != request_ids[:-1]) + 1]
        ends = np.r_[starts[1:], len(request_ids)]
        for start, end in zip(starts, ends):
            request_id = str(request_ids[start])
            clicked = clicked_by_request.get(request_id)
            if not clicked:
                continue
            values = [
                (
                    float(predictions[index]),
                    int(pre_ranks[index]),
                    int(banners[index]),
                    int(hit_logs[index]),
                    float(costs[index]),
                )
                for index in range(start, end)
            ]
            base_by_weight = {
                weight: rank_value_geometric_order(
                    values,
                    catboost_weight=weight,
                    source_cost_scale=args.source_cost_scale,
                    exponent=0.0,
                    rerank_top_n=max(top_ns),
                )
                for weight in weights
            }
            for weight in weights:
                base = base_by_weight[weight]
                for exponent in exponents:
                    effective_top_ns = max(top_ns) if exponent == 0.0 else top_ns
                    if isinstance(effective_top_ns, int):
                        effective_top_ns = [effective_top_ns]
                    for top_n in effective_top_ns:
                        combination = (weight, exponent, top_n)
                        ordered = value_geometric_from_base_order(
                            base,
                            source_cost_scale=args.source_cost_scale,
                            exponent=exponent,
                            rerank_top_n=top_n,
                        )
                        target = found[combination]
                        for rank, value in enumerate(ordered, start=1):
                            banner_id = value[2]
                            if banner_id in clicked:
                                target[(request_id, banner_id)] = rank
            processed_requests += 1

    results = []
    for (weight, exponent, top_n), ranks in found.items():
        records = [
            {"rank": int(ranks.get(pair, MISS_RANK)), "source_cost": source_cost}
            for pair, source_cost in truth.items()
        ]
        results.append(
            {
                "catboost_weight": weight,
                "source_cost_scale": args.source_cost_scale,
                "exponent": exponent,
                "rerank_top_n": top_n,
                "metrics": recall_metrics(records, [50, 100, 500]),
            }
        )
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
        "combinations": len(combinations),
        "wall_seconds": time.monotonic() - started,
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        * 1024,
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
