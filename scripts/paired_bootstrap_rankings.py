#!/usr/bin/env python3
"""Paired request and user-cluster bootstrap for two cached rankings."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from mla_recsys.artifacts import atomic_write_json, fingerprint_file  # noqa: E402
from mla_recsys.data import read_request_parquet  # noqa: E402
from scripts.tune_top50_ensemble import read_ranking  # noqa: E402


def bootstrap_ratio_delta(
    control: np.ndarray,
    candidate: np.ndarray,
    denominator: np.ndarray,
    *,
    samples: int,
    seed: int,
    batch_size: int = 256,
) -> dict[str, float | list[float]]:
    if not (len(control) == len(candidate) == len(denominator)):
        raise ValueError("bootstrap arrays have different lengths")
    if len(control) == 0 or float(denominator.sum()) <= 0.0:
        raise ValueError("bootstrap denominator is empty")
    rng = np.random.default_rng(seed)
    deltas = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, batch_size):
        end = min(start + batch_size, samples)
        indices = rng.integers(0, len(control), size=(end - start, len(control)))
        sampled_denominator = denominator[indices].sum(axis=1)
        deltas[start:end] = (
            candidate[indices].sum(axis=1) / sampled_denominator
            - control[indices].sum(axis=1) / sampled_denominator
        )
    return {
        "mean": float(deltas.mean()),
        "median": float(np.median(deltas)),
        "ci90": [float(value) for value in np.quantile(deltas, [0.05, 0.95])],
        "ci95": [float(value) for value in np.quantile(deltas, [0.025, 0.975])],
        "p_delta_gt_0": float(np.mean(deltas > 0.0)),
        "p_delta_ge_0": float(np.mean(deltas >= 0.0)),
    }


def aggregate_by_cluster(values: np.ndarray, clusters: list[object]) -> np.ndarray:
    grouped: dict[object, float] = defaultdict(float)
    for value, cluster in zip(values, clusters):
        grouped[cluster] += float(value)
    return np.asarray(list(grouped.values()), dtype=np.float64)


def request_contributions(
    requests: list[dict],
    control: dict[int, list[int]],
    candidate: dict[int, list[int]],
    cutoff: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    control_sc = []
    candidate_sc = []
    total_sc = []
    control_hits = []
    candidate_hits = []
    total_clicks = []
    for request in requests:
        hit_log_id = int(request["hit_log_id"])
        control_set = set(control[hit_log_id][:cutoff])
        candidate_set = set(candidate[hit_log_id][:cutoff])
        costs = {
            int(banner_id): float(source_cost)
            for banner_id, source_cost in zip(
                request.get("clicked_banner_ids") or (),
                request.get("clicked_source_costs") or (),
            )
        }
        control_sc.append(sum(cost for banner, cost in costs.items() if banner in control_set))
        candidate_sc.append(sum(cost for banner, cost in costs.items() if banner in candidate_set))
        total_sc.append(sum(costs.values()))
        control_hits.append(sum(banner in control_set for banner in costs))
        candidate_hits.append(sum(banner in candidate_set for banner in costs))
        total_clicks.append(len(costs))
    return tuple(
        np.asarray(values, dtype=np.float64)
        for values in (
            control_sc,
            candidate_sc,
            total_sc,
            control_hits,
            candidate_hits,
            total_clicks,
        )
    )


def point_metrics(control: np.ndarray, candidate: np.ndarray, denominator: np.ndarray) -> dict:
    total = float(denominator.sum())
    control_value = float(control.sum() / total)
    candidate_value = float(candidate.sum() / total)
    return {
        "control": control_value,
        "candidate": candidate_value,
        "delta": candidate_value - control_value,
    }


def segment_report(
    requests: list[dict],
    control: dict[int, list[int]],
    candidate: dict[int, list[int]],
    *,
    samples: int,
    seed: int,
) -> dict:
    report = {}
    user_clusters = [int(row.get("crypta_id_v2") or 0) for row in requests]
    for cutoff in (10, 50):
        sc_a, sc_b, sc_d, hit_a, hit_b, hit_d = request_contributions(
            requests, control, candidate, cutoff
        )
        metrics = {}
        for name, first, second, denominator in (
            ("sourcecost_recall", sc_a, sc_b, sc_d),
            ("recall", hit_a, hit_b, hit_d),
        ):
            point = point_metrics(first, second, denominator)
            request_bootstrap = bootstrap_ratio_delta(
                first,
                second,
                denominator,
                samples=samples,
                seed=seed + cutoff,
            )
            cluster_first = aggregate_by_cluster(first, user_clusters)
            cluster_second = aggregate_by_cluster(second, user_clusters)
            cluster_denominator = aggregate_by_cluster(denominator, user_clusters)
            user_bootstrap = bootstrap_ratio_delta(
                cluster_first,
                cluster_second,
                cluster_denominator,
                samples=samples,
                seed=seed + cutoff + 10_000,
            )
            metrics[name] = {
                **point,
                "request_bootstrap": request_bootstrap,
                "user_cluster_bootstrap": user_bootstrap,
            }
        report[str(cutoff)] = metrics
    return report


def cost_strata(
    requests: list[dict],
    control: dict[int, list[int]],
    candidate: dict[int, list[int]],
) -> list[dict]:
    all_costs = np.asarray(
        [
            float(cost)
            for request in requests
            for cost in (request.get("clicked_source_costs") or ())
        ],
        dtype=np.float64,
    )
    boundaries = np.quantile(all_costs, [0.0, 0.5, 0.9, 0.99, 1.0])
    result = []
    for index, (low, high) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        total = control_hit = candidate_hit = 0.0
        count = 0
        for request in requests:
            hit_log_id = int(request["hit_log_id"])
            control_set = set(control[hit_log_id][:50])
            candidate_set = set(candidate[hit_log_id][:50])
            for banner_id, source_cost in zip(
                request.get("clicked_banner_ids") or (),
                request.get("clicked_source_costs") or (),
            ):
                cost = float(source_cost)
                included = low <= cost <= high if index == len(boundaries) - 2 else low <= cost < high
                if not included:
                    continue
                banner_id = int(banner_id)
                total += cost
                count += 1
                control_hit += cost * (banner_id in control_set)
                candidate_hit += cost * (banner_id in candidate_set)
        result.append(
            {
                "stratum": f"p{[0, 50, 90, 99][index]}-{[50, 90, 99, 100][index]}",
                "minimum": float(low),
                "maximum": float(high),
                "clicks": count,
                "control": control_hit / total,
                "candidate": candidate_hit / total,
                "delta": (candidate_hit - control_hit) / total,
            }
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.samples <= 0:
        parser.error("samples must be positive")
    control = read_ranking(args.control)
    candidate = read_ranking(args.candidate)
    requests = sorted(
        read_request_parquet(args.requests),
        key=lambda row: (int(row.get("show_time") or 0), str(row["request_id"])),
    )
    split = len(requests) // 2
    report = {
        "control": fingerprint_file(args.control),
        "candidate": fingerprint_file(args.candidate),
        "requests": fingerprint_file(args.requests),
        "samples": args.samples,
        "seed": args.seed,
        "resampling_unit": "complete request/SearchReqId",
        "sourcecost_estimator": "ratio_of_sums",
        "segments": {
            "early": segment_report(
                requests[:split], control, candidate, samples=args.samples, seed=args.seed
            ),
            "late": segment_report(
                requests[split:], control, candidate, samples=args.samples, seed=args.seed + 1
            ),
            "full": segment_report(
                requests, control, candidate, samples=args.samples, seed=args.seed + 2
            ),
        },
        "cost_strata_at_50": cost_strata(requests, control, candidate),
    }
    atomic_write_json(args.output, report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
