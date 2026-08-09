#!/usr/bin/env python3
"""Tune a small two-model/RRF ensemble on an existing temporal holdout."""
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

from mla_recsys.artifacts import fingerprint_file  # noqa: E402
from mla_recsys.data import read_request_parquet  # noqa: E402
from mla_recsys.metrics import MISS_RANK, recall_metrics, truth_pairs  # noqa: E402
from mla_recsys.rank_blend import value_geometric_from_base_order  # noqa: E402


def parse_grid(value: str) -> list[float]:
    values = sorted({float(item) for item in value.split(",") if item})
    if not values or values[0] < 0.0 or values[-1] > 1.0:
        raise ValueError("weight grid must contain values in [0, 1]")
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


def rank_positions(
    scores: np.ndarray, pre_rank: np.ndarray, banners: np.ndarray
) -> np.ndarray:
    order = np.lexsort((banners, pre_rank, -scores))
    positions = np.empty(len(order), dtype=np.int32)
    positions[order] = np.arange(1, len(order) + 1, dtype=np.int32)
    return positions


def load_model(run: Path) -> tuple[CatBoostRanker, list[str], dict]:
    metadata = json.loads((run / "models" / "catboost.json").read_text(encoding="utf-8"))
    model = CatBoostRanker()
    model.load_model(str(run / "models" / "catboost.cbm"))
    return model, list(metadata["feature_names"]), metadata


def main() -> int:
    started = time.monotonic()
    parser = argparse.ArgumentParser(
        description="Tune a bounded two-CatBoost/RRF ensemble on temporal features"
    )
    parser.add_argument("--run", type=Path, required=True, help="Run providing features/truth")
    parser.add_argument("--model-a-run", type=Path, required=True)
    parser.add_argument("--model-b-run", type=Path, required=True)
    parser.add_argument("--model-a-weights", default="0.5,0.65,0.75,0.85")
    parser.add_argument("--catboost-weights", default="0.45,0.5,0.55,0.6,0.65")
    parser.add_argument("--geometry-exponents", default="")
    parser.add_argument("--geometry-top-n", default="75,100,150")
    parser.add_argument("--source-cost-scale", type=float, default=1_000_000.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    model_a_weights = parse_grid(args.model_a_weights)
    catboost_weights = parse_grid(args.catboost_weights)
    model_a, feature_names_a, metadata_a = load_model(args.model_a_run)
    model_b, feature_names_b, metadata_b = load_model(args.model_b_run)
    if feature_names_a != feature_names_b:
        raise ValueError("Ensemble models use different ordered feature contracts")
    feature_names = feature_names_a

    truth = truth_pairs(read_request_parquet(args.run / "data" / "holdout_requests.parquet"))
    clicked_by_request: dict[str, set[int]] = defaultdict(set)
    for request_id, banner_id in truth:
        clicked_by_request[request_id].add(banner_id)
    combinations = [
        (method, model_a_weight, catboost_weight)
        for method in ("rank_linear", "score_minmax")
        for model_a_weight in model_a_weights
        for catboost_weight in catboost_weights
    ]
    found = {combination: {} for combination in combinations}
    columns = list(
        dict.fromkeys(
            ["request_id", "banner_id", "pre_rank", "rrf_score", *feature_names]
        )
    )
    processed_requests = 0
    for path in sorted((args.run / "features" / "holdout").glob("part-*.parquet")):
        table = pq.read_table(path, columns=columns)
        if table.num_rows == 0:
            continue
        values = matrix(table, feature_names)
        prediction_a = np.asarray(model_a.predict(values), dtype=np.float64)
        prediction_b = np.asarray(model_b.predict(values), dtype=np.float64)
        request_ids = np.asarray(table["request_id"].to_pylist(), dtype=object)
        banners = table["banner_id"].combine_chunks().to_numpy(zero_copy_only=False)
        pre_rank = table["pre_rank"].combine_chunks().to_numpy(zero_copy_only=False)
        rrf = table["rrf_score"].combine_chunks().to_numpy(zero_copy_only=False)
        starts = np.r_[0, np.flatnonzero(request_ids[1:] != request_ids[:-1]) + 1]
        ends = np.r_[starts[1:], len(request_ids)]
        for start, end in zip(starts, ends):
            request_id = str(request_ids[start])
            clicked = clicked_by_request.get(request_id)
            if not clicked:
                continue
            group_banners = banners[start:end].astype(np.int64, copy=False)
            group_pre_rank = pre_rank[start:end].astype(np.int32, copy=False)
            group_rrf = rrf[start:end].astype(np.float64, copy=False)
            group_a = prediction_a[start:end]
            group_b = prediction_b[start:end]
            positions_a = rank_positions(group_a, group_pre_rank, group_banners)
            positions_b = rank_positions(group_b, group_pre_rank, group_banners)
            norm_a = normalized(group_a)
            norm_b = normalized(group_b)
            norm_rrf = normalized(group_rrf)
            for model_a_weight in model_a_weights:
                ensemble_rank = (
                    model_a_weight * positions_a.astype(np.float64)
                    + (1.0 - model_a_weight) * positions_b.astype(np.float64)
                )
                ensemble_score = model_a_weight * norm_a + (1.0 - model_a_weight) * norm_b
                for catboost_weight in catboost_weights:
                    rank_score = -(
                        catboost_weight * ensemble_rank
                        + (1.0 - catboost_weight) * group_pre_rank.astype(np.float64)
                    )
                    score = (
                        catboost_weight * ensemble_score
                        + (1.0 - catboost_weight) * norm_rrf
                    )
                    for method, blend_score in (
                        ("rank_linear", rank_score),
                        ("score_minmax", score),
                    ):
                        positions = rank_positions(blend_score, group_pre_rank, group_banners)
                        target = found[(method, model_a_weight, catboost_weight)]
                        for index, banner_id in enumerate(group_banners):
                            if int(banner_id) in clicked:
                                target[(request_id, int(banner_id))] = int(positions[index])
            processed_requests += 1

    results = []
    for (method, model_a_weight, catboost_weight), ranks in found.items():
        records = [
            {"rank": int(ranks.get(pair, MISS_RANK)), "source_cost": source_cost}
            for pair, source_cost in truth.items()
        ]
        results.append(
            {
                "method": method,
                "model_a_weight": model_a_weight,
                "catboost_weight": catboost_weight,
                "metrics": recall_metrics(records, [50, 100, 500]),
            }
        )
    results.sort(
        key=lambda item: (
            -item["metrics"]["50"]["sourcecost_recall"],
            -item["metrics"]["50"]["recall"],
        )
    )
    geometry = None
    if args.geometry_exponents:
        exponents = parse_grid(args.geometry_exponents)
        top_ns = sorted({int(item) for item in args.geometry_top_n.split(",") if item})
        if not top_ns or top_ns[0] <= 0:
            raise ValueError("geometry top-n grid must contain positive integers")
        if args.source_cost_scale <= 0.0:
            raise ValueError("source-cost scale must be positive")
        geometry_combinations = [
            (exponent, top_n)
            for exponent in exponents
            for top_n in top_ns
            if exponent > 0.0 or top_n == max(top_ns)
        ]
        geometry_found = {combination: {} for combination in geometry_combinations}
        best_base = results[0]
        model_a_weight = float(best_base["model_a_weight"])
        catboost_weight = float(best_base["catboost_weight"])
        method = str(best_base["method"])
        geometry_columns = list(
            dict.fromkeys(
                [
                    "request_id",
                    "hit_log_id",
                    "banner_id",
                    "pre_rank",
                    "rrf_score",
                    "source_cost_raw",
                    *feature_names,
                ]
            )
        )
        for path in sorted((args.run / "features" / "holdout").glob("part-*.parquet")):
            table = pq.read_table(path, columns=geometry_columns)
            if table.num_rows == 0:
                continue
            values = matrix(table, feature_names)
            prediction_a = np.asarray(model_a.predict(values), dtype=np.float64)
            prediction_b = np.asarray(model_b.predict(values), dtype=np.float64)
            request_ids = np.asarray(table["request_id"].to_pylist(), dtype=object)
            hit_logs = table["hit_log_id"].combine_chunks().to_numpy(zero_copy_only=False)
            banners = table["banner_id"].combine_chunks().to_numpy(zero_copy_only=False)
            pre_rank = table["pre_rank"].combine_chunks().to_numpy(zero_copy_only=False)
            rrf = table["rrf_score"].combine_chunks().to_numpy(zero_copy_only=False)
            costs = table["source_cost_raw"].combine_chunks().to_numpy(zero_copy_only=False)
            starts = np.r_[0, np.flatnonzero(request_ids[1:] != request_ids[:-1]) + 1]
            ends = np.r_[starts[1:], len(request_ids)]
            for start, end in zip(starts, ends):
                request_id = str(request_ids[start])
                clicked = clicked_by_request.get(request_id)
                if not clicked:
                    continue
                group_banners = banners[start:end].astype(np.int64, copy=False)
                group_pre_rank = pre_rank[start:end].astype(np.int32, copy=False)
                group_a = prediction_a[start:end]
                group_b = prediction_b[start:end]
                if method == "rank_linear":
                    positions_a = rank_positions(group_a, group_pre_rank, group_banners)
                    positions_b = rank_positions(group_b, group_pre_rank, group_banners)
                    ensemble_rank = (
                        model_a_weight * positions_a.astype(np.float64)
                        + (1.0 - model_a_weight) * positions_b.astype(np.float64)
                    )
                    blend_score = -(
                        catboost_weight * ensemble_rank
                        + (1.0 - catboost_weight)
                        * group_pre_rank.astype(np.float64)
                    )
                else:
                    ensemble_score = (
                        model_a_weight * normalized(group_a)
                        + (1.0 - model_a_weight) * normalized(group_b)
                    )
                    blend_score = (
                        catboost_weight * ensemble_score
                        + (1.0 - catboost_weight) * normalized(rrf[start:end])
                    )
                order = np.lexsort((group_banners, group_pre_rank, -blend_score))
                base = [
                    (
                        float(blend_score[index]),
                        int(group_pre_rank[index]),
                        int(group_banners[index]),
                        int(hit_logs[start + index]),
                        float(costs[start + index]),
                    )
                    for index in order
                ]
                for combination in geometry_combinations:
                    exponent, top_n = combination
                    ordered = value_geometric_from_base_order(
                        base,
                        source_cost_scale=args.source_cost_scale,
                        exponent=exponent,
                        rerank_top_n=top_n,
                    )
                    target = geometry_found[combination]
                    for rank, value in enumerate(ordered, start=1):
                        if value[2] in clicked:
                            target[(request_id, value[2])] = rank
        geometry_results = []
        for (exponent, top_n), ranks in geometry_found.items():
            records = [
                {"rank": int(ranks.get(pair, MISS_RANK)), "source_cost": source_cost}
                for pair, source_cost in truth.items()
            ]
            geometry_results.append(
                {
                    "exponent": exponent,
                    "rerank_top_n": top_n,
                    "metrics": recall_metrics(records, [50, 100, 500]),
                }
            )
        geometry_results.sort(
            key=lambda item: (
                -item["metrics"]["50"]["sourcecost_recall"],
                -item["metrics"]["50"]["recall"],
            )
        )
        geometry = {
            "base": best_base,
            "source_cost_scale": args.source_cost_scale,
            "combinations": len(geometry_combinations),
            "best": geometry_results[0],
            "results": geometry_results,
        }
    report = {
        "run": str(args.run),
        "models": {
            "a": {
                "run": str(args.model_a_run),
                "kind": metadata_a.get("kind"),
                "model": fingerprint_file(args.model_a_run / "models" / "catboost.cbm"),
            },
            "b": {
                "run": str(args.model_b_run),
                "kind": metadata_b.get("kind"),
                "model": fingerprint_file(args.model_b_run / "models" / "catboost.cbm"),
            },
        },
        "requests": processed_requests,
        "clicks": len(truth),
        "combinations": len(combinations),
        "wall_seconds": time.monotonic() - started,
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
        "best": results[0],
        "results": results,
        "geometry": geometry,
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
