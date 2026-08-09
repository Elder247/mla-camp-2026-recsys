#!/usr/bin/env python3
"""Tune a bounded train-prior correction on cached TwoTower scores."""
from __future__ import annotations

import argparse
import json
import math
import resource
import sys
import time
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from mla_recsys.artifacts import atomic_write_json  # noqa: E402
from mla_recsys.data import read_request_parquet  # noqa: E402
from scripts.tune_top50_ensemble import (  # noqa: E402
    RankedValue,
    metrics_for_orders,
    truth_records,
)


def float_grid(raw: str) -> list[float]:
    values = sorted({float(value) for value in raw.split(",") if value})
    if not values or 0.0 not in values:
        raise ValueError("alpha grid must include the zero control")
    return values


def int_grid(raw: str) -> list[int]:
    values = sorted({int(value) for value in raw.split(",") if value})
    if not values or values[0] <= 0:
        raise ValueError("top-n grid must contain positive values")
    return values


def logq_rerank(
    base: list[RankedValue],
    *,
    counts: dict[int, int],
    alpha: float,
    top_n: int,
    unseen_count: float = 1.0,
) -> list[RankedValue]:
    """Rerank only a bounded head by dot score + alpha * log(train count)."""

    if top_n <= 0 or unseen_count <= 0.0:
        raise ValueError("top_n and unseen_count must be positive")
    if alpha == 0.0:
        return list(base)
    head = list(base[:top_n])
    head.sort(
        key=lambda value: (
            -(
                float(value[0])
                + alpha
                * math.log(float(counts.get(int(value[2]), unseen_count)))
            ),
            int(value[1]),
            int(value[2]),
        )
    )
    return head + list(base[top_n:])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--prior-dir", type=Path, required=True)
    parser.add_argument(
        "--alphas", default="-0.05,-0.02,-0.01,0,0.01,0.02,0.05"
    )
    parser.add_argument("--rerank-top-n", default="50,75,100")
    parser.add_argument("--tune-fraction", type=float, default=0.5)
    parser.add_argument("--unseen-count", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 0.0 < args.tune_fraction < 1.0:
        raise ValueError("tune-fraction must be in (0, 1)")
    started = time.monotonic()
    manifest = json.loads(
        (args.prior_dir / "manifest.json").read_text(encoding="utf-8")
    )
    prior_file = args.prior_dir / str(manifest["file"]["name"])
    prior = pq.read_table(prior_file, columns=["banner_id", "count"]).to_pydict()
    counts = {
        int(banner_id): int(count)
        for banner_id, count in zip(prior["banner_id"], prior["count"])
    }
    requests = sorted(
        read_request_parquet(args.requests),
        key=lambda row: (int(row.get("show_time") or 0), str(row["request_id"])),
    )
    split = max(
        1, min(len(requests) - 1, int(len(requests) * args.tune_fraction))
    )
    early_ids = {int(row["hit_log_id"]) for row in requests[:split]}
    late_ids = {int(row["hit_log_id"]) for row in requests[split:]}
    early_truth = truth_records(requests, early_ids)
    late_truth = truth_records(requests, late_ids)
    full_truth = {**early_truth, **late_truth}
    grouped: dict[int, list[RankedValue]] = defaultdict(list)
    for path in sorted(args.candidates.glob("part-*.parquet")):
        rows = pq.read_table(
            path,
            columns=["hit_log_id", "banner_id", "source_rank", "source_score"],
        ).to_pylist()
        for row in rows:
            hit_log_id = int(row["hit_log_id"])
            grouped[hit_log_id].append(
                (
                    float(row["source_score"]),
                    int(row["source_rank"]),
                    int(row["banner_id"]),
                    hit_log_id,
                    0.0,
                )
            )
    base = {
        hit_log_id: sorted(values, key=lambda value: (value[1], value[2]))
        for hit_log_id, values in grouped.items()
    }
    results = []
    top_ns = int_grid(args.rerank_top_n)
    for alpha in float_grid(args.alphas):
        for top_n in ([max(top_ns)] if alpha == 0.0 else top_ns):
            orders = {
                hit_log_id: logq_rerank(
                    values,
                    counts=counts,
                    alpha=alpha,
                    top_n=top_n,
                    unseen_count=args.unseen_count,
                )
                for hit_log_id, values in base.items()
            }
            results.append(
                {
                    "alpha": alpha,
                    "rerank_top_n": top_n,
                    "early_metrics": metrics_for_orders(orders, early_truth),
                    "late_metrics": metrics_for_orders(orders, late_truth),
                    "full_metrics": metrics_for_orders(orders, full_truth),
                }
            )
    results.sort(
        key=lambda row: (
            -row["early_metrics"]["50"]["sourcecost_recall"],
            -row["early_metrics"]["50"]["recall"],
        )
    )
    control = next(row for row in results if row["alpha"] == 0.0)
    best = results[0]
    report = {
        "version": 1,
        "candidates": str(args.candidates),
        "requests": str(args.requests),
        "prior_dir": str(args.prior_dir),
        "prior_source": manifest["source"],
        "combinations": len(results),
        "control": control,
        "best": best,
        "gains": {
            scope: best[f"{scope}_metrics"]["50"]["sourcecost_recall"]
            - control[f"{scope}_metrics"]["50"]["sourcecost_recall"]
            for scope in ("early", "late", "full")
        },
        "accepted": all(
            best[f"{scope}_metrics"]["50"]["sourcecost_recall"]
            > control[f"{scope}_metrics"]["50"]["sourcecost_recall"]
            for scope in ("early", "late")
        ),
        "results": results,
        "wall_seconds": time.monotonic() - started,
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        * 1024,
    }
    atomic_write_json(args.output, report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
