#!/usr/bin/env python3
"""Tune a bounded RRF over two already-ranked temporal candidate pools."""
from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from catboost import CatBoostRanker

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mla_recsys.artifacts import fingerprint_file  # noqa: E402
from mla_recsys.data import read_request_parquet  # noqa: E402
from mla_recsys.metrics import MISS_RANK, recall_metrics, truth_pairs  # noqa: E402
from mla_recsys.rank_blend import (  # noqa: E402
    two_model_rank_linear_order,
    value_geometric_from_base_order,
)


Ranked = tuple[int, int, float]


def float_grid(raw: str) -> list[float]:
    return sorted({float(value) for value in raw.split(",") if value})


def int_grid(raw: str) -> list[int]:
    return sorted({int(value) for value in raw.split(",") if value})


def matrix(table: Any, names: list[str]) -> np.ndarray:
    return np.column_stack(
        [table[name].combine_chunks().to_numpy(zero_copy_only=False) for name in names]
    ).astype(np.float32, copy=False)


def load_model(run: Path) -> tuple[CatBoostRanker, list[str]]:
    metadata = json.loads((run / "models" / "catboost.json").read_text(encoding="utf-8"))
    model = CatBoostRanker()
    model.load_model(str(run / "models" / "catboost.cbm"))
    return model, list(metadata["feature_names"])


def ranked_pool(
    *,
    run: Path,
    model_a_run: Path,
    model_b_run: Path,
    model_a_weight: float,
    catboost_weight: float,
    exponent: float,
    rerank_top_n: int,
) -> dict[str, list[Ranked]]:
    model_a, names_a = load_model(model_a_run)
    model_b, names_b = load_model(model_b_run)
    if names_a != names_b:
        raise ValueError("Two-model feature contracts differ")
    names = names_a
    output: dict[str, list[Ranked]] = {}
    columns = list(
        dict.fromkeys(
            [
                "request_id",
                "hit_log_id",
                "banner_id",
                "pre_rank",
                "source_cost_raw",
                *names,
            ]
        )
    )
    for path in sorted((run / "features" / "holdout").glob("part-*.parquet")):
        table = pq.read_table(path, columns=columns)
        values = matrix(table, names)
        scores_a = np.asarray(model_a.predict(values), dtype=np.float64)
        scores_b = np.asarray(model_b.predict(values), dtype=np.float64)
        rows = table.select(
            ["request_id", "hit_log_id", "banner_id", "pre_rank", "source_cost_raw"]
        ).to_pylist()
        grouped: dict[str, list[tuple[float, float, int, int, int, float]]] = defaultdict(list)
        for row, score_a, score_b in zip(rows, scores_a, scores_b):
            grouped[str(row["request_id"])].append(
                (
                    float(score_a),
                    float(score_b),
                    int(row["pre_rank"]),
                    int(row["banner_id"]),
                    int(row["hit_log_id"]),
                    float(row["source_cost_raw"]),
                )
            )
        for request_id, candidates in grouped.items():
            base = two_model_rank_linear_order(
                candidates,
                model_a_weight=model_a_weight,
                catboost_weight=catboost_weight,
            )
            geometry_base = [
                (0.0, rank, value[3], value[4], value[5])
                for rank, value in enumerate(base, start=1)
            ]
            ordered = value_geometric_from_base_order(
                geometry_base,
                source_cost_scale=1_000_000.0,
                exponent=exponent,
                rerank_top_n=rerank_top_n,
            )
            output[request_id] = [
                (int(value[2]), int(value[3]), float(value[4])) for value in ordered
            ]
    return output


def fuse_orders(
    old: list[Ranked], new: list[Ranked], *, new_weight: float, rrf_constant: float
) -> list[tuple[float, int, int, int, float]]:
    if not 0.0 <= new_weight <= 1.0:
        raise ValueError("new_weight must be in [0, 1]")
    if rrf_constant < 0.0:
        raise ValueError("rrf_constant must be non-negative")
    merged: dict[int, list[float | int]] = {}
    for source_weight, ranking in ((1.0 - new_weight, old), (new_weight, new)):
        for rank, (banner_id, hit_log_id, source_cost) in enumerate(ranking, start=1):
            state = merged.setdefault(
                banner_id,
                [0.0, 10**9, hit_log_id, max(0.0, source_cost)],
            )
            state[0] = float(state[0]) + source_weight / (rrf_constant + rank)
            state[1] = min(int(state[1]), rank)
            state[3] = max(float(state[3]), max(0.0, source_cost))
    ordered = sorted(
        (
            float(value[0]),
            int(value[1]),
            int(banner_id),
            int(value[2]),
            float(value[3]),
        )
        for banner_id, value in merged.items()
    )
    ordered.sort(key=lambda value: (-value[0], value[1], value[2]))
    return ordered


def metrics_for_orders(
    orders: dict[str, list[tuple]], truth: dict[tuple[str, int], float]
) -> dict:
    found: dict[tuple[str, int], int] = {}
    clicked: dict[str, set[int]] = defaultdict(set)
    for request_id, banner_id in truth:
        clicked[request_id].add(banner_id)
    for request_id, order in orders.items():
        targets = clicked.get(request_id)
        if not targets:
            continue
        for rank, value in enumerate(order, start=1):
            banner_id = int(value[2])
            if banner_id in targets:
                found[(request_id, banner_id)] = rank
    records = [
        {"rank": int(found.get(pair, MISS_RANK)), "source_cost": source_cost}
        for pair, source_cost in truth.items()
    ]
    return recall_metrics(records, [50, 100, 500])


def main() -> int:
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-run", type=Path, required=True)
    parser.add_argument("--old-model-a-run", type=Path, required=True)
    parser.add_argument("--old-model-b-run", type=Path, required=True)
    parser.add_argument("--new-run", type=Path, required=True)
    parser.add_argument("--new-model-a-run", type=Path, required=True)
    parser.add_argument("--new-model-b-run", type=Path, required=True)
    parser.add_argument("--old-model-a-weight", type=float, default=0.5)
    parser.add_argument("--old-catboost-weight", type=float, default=0.5)
    parser.add_argument("--old-exponent", type=float, default=0.2)
    parser.add_argument("--old-top-n", type=int, default=75)
    parser.add_argument("--new-model-a-weight", type=float, default=0.65)
    parser.add_argument("--new-catboost-weight", type=float, default=0.6)
    parser.add_argument("--new-exponent", type=float, default=0.2)
    parser.add_argument("--new-top-n", type=int, default=75)
    parser.add_argument("--new-weights", default="0.25,0.4,0.5,0.6,0.75")
    parser.add_argument("--rrf-constants", default="10,20,40,60")
    parser.add_argument("--geometry-exponents", default="0,0.05,0.1,0.15,0.2")
    parser.add_argument("--geometry-top-n", default="50,75,100,150")
    parser.add_argument("--refine-top", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    old_truth = truth_pairs(read_request_parquet(args.old_run / "data/holdout_requests.parquet"))
    new_truth = truth_pairs(read_request_parquet(args.new_run / "data/holdout_requests.parquet"))
    if old_truth != new_truth:
        raise ValueError("Cross-pool runs use different holdout truth")
    old = ranked_pool(
        run=args.old_run,
        model_a_run=args.old_model_a_run,
        model_b_run=args.old_model_b_run,
        model_a_weight=args.old_model_a_weight,
        catboost_weight=args.old_catboost_weight,
        exponent=args.old_exponent,
        rerank_top_n=args.old_top_n,
    )
    new = ranked_pool(
        run=args.new_run,
        model_a_run=args.new_model_a_run,
        model_b_run=args.new_model_b_run,
        model_a_weight=args.new_model_a_weight,
        catboost_weight=args.new_catboost_weight,
        exponent=args.new_exponent,
        rerank_top_n=args.new_top_n,
    )
    if set(old) != set(new):
        raise ValueError("Cross-pool runs cover different requests")

    base_results = []
    base_orders: dict[tuple[float, float], dict[str, list[tuple]]] = {}
    for new_weight in float_grid(args.new_weights):
        for constant in float_grid(args.rrf_constants):
            key = (new_weight, constant)
            orders = {
                request_id: fuse_orders(
                    old[request_id],
                    new[request_id],
                    new_weight=new_weight,
                    rrf_constant=constant,
                )
                for request_id in old
            }
            base_orders[key] = orders
            base_results.append(
                {
                    "new_weight": new_weight,
                    "rrf_constant": constant,
                    "metrics": metrics_for_orders(orders, old_truth),
                }
            )
    base_results.sort(
        key=lambda value: (
            -value["metrics"]["50"]["sourcecost_recall"],
            -value["metrics"]["50"]["recall"],
        )
    )

    geometry_results = []
    exponents = float_grid(args.geometry_exponents)
    top_ns = int_grid(args.geometry_top_n)
    for base_result in base_results[: args.refine_top]:
        key = (float(base_result["new_weight"]), float(base_result["rrf_constant"]))
        for exponent in exponents:
            for top_n in ([max(top_ns)] if exponent == 0.0 else top_ns):
                orders = {
                    request_id: value_geometric_from_base_order(
                        order,
                        source_cost_scale=1_000_000.0,
                        exponent=exponent,
                        rerank_top_n=top_n,
                    )
                    for request_id, order in base_orders[key].items()
                }
                geometry_results.append(
                    {
                        "new_weight": key[0],
                        "rrf_constant": key[1],
                        "exponent": exponent,
                        "rerank_top_n": top_n,
                        "metrics": metrics_for_orders(orders, old_truth),
                    }
                )
    geometry_results.sort(
        key=lambda value: (
            -value["metrics"]["50"]["sourcecost_recall"],
            -value["metrics"]["50"]["recall"],
        )
    )
    report = {
        "old_run": str(args.old_run),
        "new_run": str(args.new_run),
        "requests": len(old),
        "clicks": len(old_truth),
        "inputs": {
            "old_model_a": fingerprint_file(args.old_model_a_run / "models/catboost.cbm"),
            "old_model_b": fingerprint_file(args.old_model_b_run / "models/catboost.cbm"),
            "new_model_a": fingerprint_file(args.new_model_a_run / "models/catboost.cbm"),
            "new_model_b": fingerprint_file(args.new_model_b_run / "models/catboost.cbm"),
        },
        "base_combinations": len(base_results),
        "best_base": base_results[0],
        "geometry_combinations": len(geometry_results),
        "best": geometry_results[0],
        "base_results": base_results,
        "geometry_results": geometry_results,
        "wall_seconds": time.monotonic() - started,
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("best_base", "best", "wall_seconds")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
