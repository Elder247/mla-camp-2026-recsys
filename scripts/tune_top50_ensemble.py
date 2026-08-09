#!/usr/bin/env python3
"""Tune a small RRF ensemble from cached top-50 or candidate rankings.

The parameter choice is made on the earlier half of the temporal holdout and
reported separately on its later half.  This keeps the cheap post-ranking
probe useful without treating the same clicks as both tuning and validation.
"""
from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mla_recsys.artifacts import fingerprint_file  # noqa: E402
from mla_recsys.data import read_request_parquet  # noqa: E402
from mla_recsys.metrics import MISS_RANK, recall_metrics  # noqa: E402
from mla_recsys.rank_blend import value_geometric_from_base_order  # noqa: E402
from mla_recsys.text import normalize  # noqa: E402


Rankings = dict[int, list[int]]
RankedValue = tuple[float, int, int, int, float]


def float_grid(raw: str) -> list[float]:
    return sorted({float(value) for value in raw.split(",") if value})


def int_grid(raw: str) -> list[int]:
    return sorted({int(value) for value in raw.split(",") if value})


def simplex_weights(
    count: int,
    step: float,
    *,
    minimum_first_weight: float = 0.0,
) -> list[tuple[float, ...]]:
    if count <= 0:
        raise ValueError("At least one ranking input is required")
    denominator = round(1.0 / step)
    if step <= 0.0 or abs(denominator * step - 1.0) > 1.0e-9:
        raise ValueError("weight-step must divide one exactly")

    def compositions(total: int, width: int) -> list[tuple[int, ...]]:
        if width == 1:
            return [(total,)]
        return [
            (head, *tail)
            for head in range(total + 1)
            for tail in compositions(total - head, width - 1)
        ]

    if not 0.0 <= minimum_first_weight <= 1.0:
        raise ValueError("minimum-first-weight must be in [0, 1]")
    return [
        tuple(value / denominator for value in row)
        for row in compositions(denominator, count)
        if value_at_least(row[0] / denominator, minimum_first_weight)
    ]


def value_at_least(value: float, threshold: float) -> bool:
    """Compare grid weights without dropping a boundary to float noise."""
    return value + 1.0e-12 >= threshold


def conditional_weights(
    weights: tuple[float, ...], *, use_secondary_sources: bool
) -> tuple[float, ...]:
    if use_secondary_sources:
        return weights
    return (1.0, *(0.0 for _ in weights[1:]))


def read_ranking(path: Path, *, candidate_top_k: int = 0) -> Rankings:
    if path.is_file():
        table = pq.read_table(path, columns=["HitLogID", "BannerID"])
        return {
            int(row["HitLogID"]): [int(value) for value in row["BannerID"]]
            for row in table.to_pylist()
        }
    parts = sorted(path.glob("part-*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No ranking parquet found at {path}")
    grouped: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for part in parts:
        filters = (
            [("source_rank", "<=", candidate_top_k)]
            if candidate_top_k > 0
            else None
        )
        table = pq.read_table(
            part,
            columns=["hit_log_id", "banner_id", "source_rank"],
            filters=filters,
        )
        for row in table.to_pylist():
            grouped[int(row["hit_log_id"])].append(
                (int(row["source_rank"]), int(row["banner_id"]))
            )
    return {
        hit_log_id: [banner_id for _, banner_id in sorted(values)]
        for hit_log_id, values in grouped.items()
    }


def fuse_rankings(
    rankings: list[list[int]],
    weights: tuple[float, ...],
    *,
    rrf_constant: float,
    hit_log_id: int,
    source_costs: dict[int, float],
) -> list[RankedValue]:
    if len(rankings) != len(weights):
        raise ValueError("Ranking and weight counts differ")
    merged: dict[int, list[float | int]] = {}
    for weight, ranking in zip(weights, rankings):
        if weight == 0.0:
            continue
        for rank, banner_id in enumerate(ranking, start=1):
            state = merged.setdefault(int(banner_id), [0.0, 10**9])
            state[0] = float(state[0]) + weight / (rrf_constant + rank)
            state[1] = min(int(state[1]), rank)
    ordered = [
        (
            float(value[0]),
            int(value[1]),
            int(banner_id),
            int(hit_log_id),
            max(0.0, float(source_costs.get(int(banner_id), 0.0))),
        )
        for banner_id, value in merged.items()
    ]
    ordered.sort(key=lambda value: (-value[0], value[1], value[2]))
    return ordered


def truth_records(
    requests: list[dict], allowed_hit_log_ids: set[int]
) -> dict[tuple[int, int], float]:
    truth = {}
    for request in requests:
        hit_log_id = int(request["hit_log_id"])
        if hit_log_id not in allowed_hit_log_ids:
            continue
        for banner_id, source_cost in zip(
            request.get("clicked_banner_ids") or (),
            request.get("clicked_source_costs") or (),
        ):
            truth[(hit_log_id, int(banner_id))] = float(source_cost)
    return truth


def metrics_for_orders(
    orders: dict[int, list[RankedValue]], truth: dict[tuple[int, int], float]
) -> dict:
    found = {}
    targets: dict[int, set[int]] = defaultdict(set)
    for hit_log_id, banner_id in truth:
        targets[hit_log_id].add(banner_id)
    for hit_log_id, order in orders.items():
        wanted = targets.get(hit_log_id)
        if not wanted:
            continue
        for rank, value in enumerate(order, start=1):
            if value[2] in wanted:
                found[(hit_log_id, value[2])] = rank
    records = [
        {"rank": found.get(pair, MISS_RANK), "source_cost": source_cost}
        for pair, source_cost in truth.items()
    ]
    return recall_metrics(records, [50, 100, 500])


def main() -> int:
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        required=True,
        help="Top-50 parquet or candidate partition directory; repeat per source",
    )
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--banner-index", type=Path, required=True)
    parser.add_argument("--weight-step", type=float, default=0.25)
    parser.add_argument(
        "--minimum-first-weight",
        type=float,
        default=0.0,
        help="Restrict the simplex to mixtures retaining this much of input 1",
    )
    parser.add_argument("--rrf-constants", default="0,10,40")
    parser.add_argument("--geometry-exponents", default="0,0.1,0.2")
    parser.add_argument("--geometry-top-n", default="50,75,100")
    parser.add_argument("--refine-top", type=int, default=5)
    parser.add_argument("--tune-fraction", type=float, default=0.5)
    parser.add_argument(
        "--tail-train-requests",
        type=Path,
        help=(
            "Past-only request parquet used to count normalized queries. When "
            "set, secondary sources are used only for sufficiently rare queries."
        ),
    )
    parser.add_argument(
        "--secondary-query-count-max",
        type=int,
        help="Use secondary sources only when past normalized-query count is <= this",
    )
    parser.add_argument(
        "--candidate-top-k",
        type=int,
        default=0,
        help=(
            "Read only this many rows per request from candidate directories; "
            "zero preserves the complete ranking. Use 100 for fast Recall@50 probes."
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0.0 < args.tune_fraction < 1.0:
        raise ValueError("tune-fraction must be in (0, 1)")

    if args.candidate_top_k < 0:
        raise ValueError("candidate-top-k must be non-negative")
    sources = [
        read_ranking(path, candidate_top_k=args.candidate_top_k)
        for path in args.input
    ]
    requests = sorted(
        read_request_parquet(args.requests),
        key=lambda row: (int(row.get("show_time") or 0), str(row["request_id"])),
    )
    if (args.tail_train_requests is None) != (
        args.secondary_query_count_max is None
    ):
        parser.error(
            "--tail-train-requests and --secondary-query-count-max must be set together"
        )
    if args.secondary_query_count_max is not None and args.secondary_query_count_max < 0:
        parser.error("--secondary-query-count-max must be non-negative")
    tail_hit_log_ids: set[int] | None = None
    if args.tail_train_requests is not None:
        past_counts = Counter(
            normalize(row["query"])
            for row in read_request_parquet(args.tail_train_requests)
        )
        tail_hit_log_ids = {
            int(row["hit_log_id"])
            for row in requests
            if past_counts[normalize(row["query"])]
            <= args.secondary_query_count_max
        }
    split = max(1, min(len(requests) - 1, int(len(requests) * args.tune_fraction)))
    tune_ids = {int(row["hit_log_id"]) for row in requests[:split]}
    validation_ids = {int(row["hit_log_id"]) for row in requests[split:]}
    all_ids = tune_ids | validation_ids
    tune_truth = truth_records(requests, tune_ids)
    validation_truth = truth_records(requests, validation_ids)
    all_truth = {**tune_truth, **validation_truth}
    index = pq.read_table(args.banner_index, columns=["BannerID", "SourceCost"])
    source_costs = {
        int(banner_id): float(source_cost or 0.0)
        for banner_id, source_cost in zip(index["BannerID"], index["SourceCost"])
    }

    base_results = []
    for weights in simplex_weights(
        len(sources),
        args.weight_step,
        minimum_first_weight=args.minimum_first_weight,
    ):
        for constant in float_grid(args.rrf_constants):
            orders = {
                hit_log_id: fuse_rankings(
                    [source.get(hit_log_id, []) for source in sources],
                    conditional_weights(
                        weights,
                        use_secondary_sources=(
                            tail_hit_log_ids is None or hit_log_id in tail_hit_log_ids
                        ),
                    ),
                    rrf_constant=constant,
                    hit_log_id=hit_log_id,
                    source_costs=source_costs,
                )
                for hit_log_id in all_ids
            }
            base_results.append(
                {
                    "weights": weights,
                    "rrf_constant": constant,
                    "tune_metrics": metrics_for_orders(orders, tune_truth),
                }
            )
    base_results.sort(
        key=lambda row: (
            -row["tune_metrics"]["50"]["sourcecost_recall"],
            -row["tune_metrics"]["50"]["recall"],
        )
    )

    refined = []
    exponents = float_grid(args.geometry_exponents)
    top_ns = int_grid(args.geometry_top_n)
    for base in base_results[: args.refine_top]:
        key = (tuple(base["weights"]), float(base["rrf_constant"]))
        base_order = {
            hit_log_id: fuse_rankings(
                [source.get(hit_log_id, []) for source in sources],
                conditional_weights(
                    key[0],
                    use_secondary_sources=(
                        tail_hit_log_ids is None or hit_log_id in tail_hit_log_ids
                    ),
                ),
                rrf_constant=key[1],
                hit_log_id=hit_log_id,
                source_costs=source_costs,
            )
            for hit_log_id in all_ids
        }
        for exponent in exponents:
            for top_n in ([max(top_ns)] if exponent == 0.0 else top_ns):
                orders = {
                    hit_log_id: value_geometric_from_base_order(
                        order,
                        source_cost_scale=1_000_000.0,
                        exponent=exponent,
                        rerank_top_n=top_n,
                    )
                    for hit_log_id, order in base_order.items()
                }
                refined.append(
                    {
                        "weights": key[0],
                        "rrf_constant": key[1],
                        "exponent": exponent,
                        "rerank_top_n": top_n,
                        "tune_metrics": metrics_for_orders(orders, tune_truth),
                        "validation_metrics": metrics_for_orders(
                            orders, validation_truth
                        ),
                        "full_metrics": metrics_for_orders(orders, all_truth),
                    }
                )
    refined.sort(
        key=lambda row: (
            -row["tune_metrics"]["50"]["sourcecost_recall"],
            -row["tune_metrics"]["50"]["recall"],
        )
    )
    report = {
        "inputs": [fingerprint_file(path) for path in args.input if path.is_file()],
        "input_paths": [str(path) for path in args.input],
        "candidate_top_k": args.candidate_top_k,
        "minimum_first_weight": args.minimum_first_weight,
        "tail_train_requests": (
            str(args.tail_train_requests) if args.tail_train_requests else None
        ),
        "secondary_query_count_max": args.secondary_query_count_max,
        "secondary_request_count": (
            len(tail_hit_log_ids) if tail_hit_log_ids is not None else len(requests)
        ),
        "requests": len(requests),
        "tune_requests": len(tune_ids),
        "validation_requests": len(validation_ids),
        "tune_clicks": len(tune_truth),
        "validation_clicks": len(validation_truth),
        "base_combinations": len(base_results),
        "refined_combinations": len(refined),
        "best": refined[0],
        "base_results": base_results,
        "refined_results": refined,
        "wall_seconds": time.monotonic() - started,
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        * 1024,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"best": report["best"], "wall_seconds": report["wall_seconds"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
