#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mla_recsys.artifacts import atomic_write_json  # noqa: E402
from mla_recsys.data import read_request_parquet  # noqa: E402
from mla_recsys.metrics import MISS_RANK, recall_metrics, truth_pairs  # noqa: E402
from mla_recsys.rank_blend import value_geometric_from_base_order  # noqa: E402


def parse_floats(value: str) -> list[float]:
    values = sorted({float(item) for item in value.split(",") if item})
    if not values or values[0] < 0:
        raise ValueError("exponents must be non-negative")
    return values


def parse_ints(value: str) -> list[int]:
    values = sorted({int(item) for item in value.split(",") if item})
    if not values or values[0] <= 0:
        raise ValueError("top-n values must be positive")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Tune bounded SourceCost geometry on direct TwoTower ranks"
    )
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--source", default="two_tower_v2")
    parser.add_argument("--split", default="holdout")
    parser.add_argument("--exponents", default="0,0.05,0.1,0.15,0.2,0.3,0.4")
    parser.add_argument("--rerank-top-n", default="50,75,100,150,250")
    parser.add_argument("--source-cost-scale", type=float, default=1_000_000.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    exponents = parse_floats(args.exponents)
    top_ns = parse_ints(args.rerank_top_n)
    metadata = pq.read_table(
        args.artifact / "candidate_metadata.parquet",
        columns=["banner_id", "source_cost"],
    ).to_pydict()
    costs = {
        int(banner): float(cost or 0.0)
        for banner, cost in zip(metadata["banner_id"], metadata["source_cost"])
    }
    requests = read_request_parquet(
        args.run / "data" / f"{args.split}_requests.parquet"
    )
    truth = truth_pairs(requests)
    clicked: dict[str, set[int]] = defaultdict(set)
    for request_id, banner_id in truth:
        clicked[request_id].add(banner_id)
    combinations = [
        (exponent, top_n)
        for exponent in exponents
        for top_n in top_ns
        if exponent > 0 or top_n == max(top_ns)
    ]
    found = {combination: {} for combination in combinations}
    for path in sorted(
        (args.run / "candidates" / args.split / args.source).glob("part-*.parquet")
    ):
        rows = pq.read_table(
            path,
            columns=[
                "request_id",
                "hit_log_id",
                "banner_id",
                "source_rank",
                "source_score",
            ],
        ).to_pylist()
        grouped: dict[str, list[tuple[float, int, int, int, float]]] = defaultdict(list)
        for row in rows:
            banner = int(row["banner_id"])
            grouped[str(row["request_id"])].append(
                (
                    float(row["source_score"]),
                    int(row["source_rank"]),
                    banner,
                    int(row["hit_log_id"]),
                    costs.get(banner, 0.0),
                )
            )
        for request_id, values in grouped.items():
            relevant = clicked.get(request_id)
            if not relevant:
                continue
            base = sorted(values, key=lambda value: (value[1], value[2]))
            for combination in combinations:
                exponent, top_n = combination
                ordered = value_geometric_from_base_order(
                    base,
                    source_cost_scale=args.source_cost_scale,
                    exponent=exponent,
                    rerank_top_n=top_n,
                )
                for rank, value in enumerate(ordered[:500], start=1):
                    if value[2] in relevant:
                        found[combination][(request_id, value[2])] = rank
    results = []
    for (exponent, top_n), ranks in found.items():
        records = [
            {
                "rank": int(ranks.get(pair, MISS_RANK)),
                "source_cost": source_cost,
            }
            for pair, source_cost in truth.items()
        ]
        results.append(
            {
                "exponent": exponent,
                "rerank_top_n": top_n,
                "source_cost_scale": args.source_cost_scale,
                "metrics": recall_metrics(records, [10, 50, 100, 500]),
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
        "artifact": str(args.artifact),
        "source": args.source,
        "split": args.split,
        "combinations": len(combinations),
        "best": results[0],
        "results": results,
        "wall_seconds": time.monotonic() - started,
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        * 1024,
    }
    output = args.output or args.run / "metrics" / "two_tower_geometry.json"
    atomic_write_json(output, report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
