from __future__ import annotations

import csv
import json
import math
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from catboost import CatBoostRanker
from omegaconf import DictConfig

from .candidate_cache import enabled_sources, source_part_path
from .data import read_request_parquet
from .metrics import MISS_RANK, recall_metrics, records_from_found, truth_pairs
from .rank_blend import rank_linear_order


def source_gate_rows(
    *,
    truth: dict[tuple[str, int], float],
    found: dict[str, dict[tuple[str, int], int]],
    sources: list[str],
    metrics: dict[str, dict[str, dict[str, Any]]],
    baseline_sources: list[str],
    high_value_quantile: float,
) -> list[dict[str, Any]]:
    total_cost = sum(truth.values())
    baseline_pairs = {
        pair
        for pair in truth
        if any(pair in found[source] for source in baseline_sources)
    }
    baseline_cost = sum(truth[pair] for pair in baseline_pairs)
    ordered_costs = sorted(truth.values())
    quantile_index = min(
        len(ordered_costs) - 1,
        max(
            0,
            int(math.ceil(high_value_quantile * max(0, len(ordered_costs) - 1))),
        ),
    )
    threshold = ordered_costs[quantile_index] if ordered_costs else float("inf")
    output = []
    for source in sources:
        only_pairs = [
            pair
            for pair in truth
            if pair in found[source]
            and not any(pair in found[other] for other in sources if other != source)
        ]
        unique_cost = sum(truth[pair] for pair in only_pairs)
        added_pairs = {pair for pair in truth if pair in found[source]} - baseline_pairs
        added_cost = sum(truth[pair] for pair in added_pairs)
        high_value_pairs = [pair for pair in only_pairs if truth[pair] >= threshold]
        high_value_truth = [pair for pair in truth if truth[pair] >= threshold]
        high_value_found = [pair for pair in high_value_truth if pair in found[source]]
        high_value_total_cost = sum(truth[pair] for pair in high_value_truth)
        output.append(
            {
                "source": source,
                "unique_clicked_banners": len(only_pairs),
                "unique_sourcecost": unique_cost,
                "unique_sourcecost_share": unique_cost / total_cost if total_cost else 0.0,
                "unique_high_value_hits": len(high_value_pairs),
                "high_value_quantile": high_value_quantile,
                "high_value_hits": len(high_value_found),
                "high_value_sourcecost_recall": (
                    sum(truth[pair] for pair in high_value_found) / high_value_total_cost
                    if high_value_total_cost
                    else 0.0
                ),
                "incremental_oracle_hits_vs_baseline": len(added_pairs),
                "incremental_oracle_sourcecost": added_cost,
                "incremental_oracle_sc_share_vs_baseline": (
                    added_cost / total_cost if total_cost else 0.0
                ),
                "baseline_oracle_sc_share": baseline_cost / total_cost if total_cost else 0.0,
                "recall_at_50": metrics[source]["50"]["recall"],
                "sourcecost_recall_at_50": metrics[source]["50"]["sourcecost_recall"],
                "sourcecost_recall_at_500": metrics[source]["500"]["sourcecost_recall"],
            }
        )
    return output


def candidate_report(
    *,
    cfg: DictConfig,
    run_path: Path,
    split: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    requests = read_request_parquet(run_path / "data" / f"{split}_requests.parquet")
    truth = truth_pairs(requests)
    sources = enabled_sources(cfg)
    found: dict[str, dict[tuple[str, int], int]] = {source: {} for source in sources}
    found["merged"] = {}
    jaccard_sum: Counter[tuple[str, str]] = Counter()
    jaccard_count: Counter[tuple[str, str]] = Counter()
    partitions = int(cfg.data.partition_count)
    for partition in range(partitions):
        per_source_sets: dict[str, dict[str, set[int]]] = {}
        for source in sources:
            table = pq.read_table(
                source_part_path(run_path, split, source, partition),
                columns=["request_id", "banner_id", "source_rank"],
            ).to_pylist()
            sets: dict[str, set[int]] = {}
            for row in table:
                request_id = str(row["request_id"])
                banner_id = int(row["banner_id"])
                if int(row["source_rank"]) <= 50:
                    sets.setdefault(request_id, set()).add(banner_id)
                pair = (request_id, banner_id)
                if pair in truth:
                    found[source][pair] = min(
                        found[source].get(pair, MISS_RANK), int(row["source_rank"])
                    )
            per_source_sets[source] = sets
        request_ids = set().union(*(sets.keys() for sets in per_source_sets.values()))
        for left, right in combinations(sources, 2):
            key = (left, right)
            for request_id in request_ids:
                a = per_source_sets[left].get(request_id, set())
                b = per_source_sets[right].get(request_id, set())
                union = a | b
                jaccard_sum[key] += len(a & b) / len(union) if union else 0.0
                jaccard_count[key] += 1
        merged_path = run_path / "candidates" / split / "merged" / f"part-{partition:05d}.parquet"
        for row in pq.read_table(
            merged_path, columns=["request_id", "banner_id", "pre_rank"]
        ).to_pylist():
            pair = (str(row["request_id"]), int(row["banner_id"]))
            if pair in truth:
                found["merged"][pair] = min(
                    found["merged"].get(pair, MISS_RANK), int(row["pre_rank"])
                )

    cutoffs = [int(value) for value in cfg.evaluation.cutoffs]
    metrics = {
        name: recall_metrics(records_from_found(truth, ranks), cutoffs)
        for name, ranks in found.items()
    }
    membership = Counter()
    for pair, source_cost in truth.items():
        present = tuple(sorted(source for source in sources if pair in found[source]))
        membership[present or ("miss",)] += 1
    configured_baseline = cfg.evaluation.get("complementarity_baseline_sources", [])
    baseline_sources = [
        str(value) for value in (configured_baseline or sources[:2])
    ]
    unknown_baseline = set(baseline_sources) - set(sources)
    if unknown_baseline:
        raise ValueError(f"Unknown complementarity baseline sources: {unknown_baseline}")
    unique_rows = source_gate_rows(
        truth=truth,
        found=found,
        sources=sources,
        metrics=metrics,
        baseline_sources=baseline_sources,
        high_value_quantile=float(cfg.evaluation.get("high_value_quantile", 0.9)),
    )
    for row in unique_rows:
        source = str(row["source"])
        generation_path = run_path / "metrics" / f"generate_{split}_{source}.json"
        generation = json.loads(generation_path.read_text(encoding="utf-8"))
        source_size = sum(
            path.stat().st_size
            for path in (run_path / "candidates" / split / source).glob("part-*.parquet")
        )
        row.update(
            {
                "request_coverage": float(generation["coverage"]),
                "generated_rows": int(generation["rows"]),
                "rows_per_request": (
                    float(generation["rows"]) / float(generation["requests"])
                    if generation["requests"]
                    else 0.0
                ),
                "generation_wall_seconds": float(generation["wall_seconds"]),
                "parquet_bytes": source_size,
            }
        )
    overlap = {
        f"{left}|{right}": jaccard_sum[(left, right)] / jaccard_count[(left, right)]
        if jaccard_count[(left, right)]
        else 0.0
        for left, right in combinations(sources, 2)
    }
    report = {
        "split": split,
        "requests": len(requests),
        "clicks": len(truth),
        "metrics": metrics,
        "source_membership": {"+".join(key): value for key, value in sorted(membership.items())},
        "mean_jaccard_top50": overlap,
        "complementarity_baseline_sources": baseline_sources,
    }
    return report, unique_rows


def _matrix(table: pa.Table, feature_names: list[str]) -> np.ndarray:
    return np.column_stack(
        [table[name].combine_chunks().to_numpy(zero_copy_only=False) for name in feature_names]
    ).astype(np.float32, copy=False)


def ranker_report(
    *,
    cfg: DictConfig,
    run_path: Path,
    split: str,
) -> tuple[dict[str, Any], pa.Table]:
    metadata = __import__("json").loads(
        (run_path / "models" / "catboost.json").read_text(encoding="utf-8")
    )
    feature_names = list(metadata["feature_names"])
    model = CatBoostRanker()
    model.load_model(str(run_path / "models" / "catboost.cbm"))
    requests = read_request_parquet(run_path / "data" / f"{split}_requests.parquet")
    truth = truth_pairs(requests)
    rrf_found: dict[tuple[str, int], int] = {}
    ranker_found: dict[tuple[str, int], int] = {}
    blend_found: dict[tuple[str, int], int] = {}
    predictions: dict[str, tuple[int, list[int]]] = {}
    for path in sorted((run_path / "features" / split).glob("part-*.parquet")):
        columns = ["request_id", "hit_log_id", "banner_id", "pre_rank", *feature_names]
        table = pq.read_table(path, columns=columns)
        if table.num_rows == 0:
            continue
        scores = model.predict(_matrix(table, feature_names))
        rows = table.select(["request_id", "hit_log_id", "banner_id", "pre_rank"]).to_pylist()
        grouped: dict[str, list[tuple[float, int, int, int]]] = {}
        for row, score in zip(rows, scores):
            request_id = str(row["request_id"])
            banner_id = int(row["banner_id"])
            pre_rank = int(row["pre_rank"])
            grouped.setdefault(request_id, []).append(
                (float(score), pre_rank, banner_id, int(row["hit_log_id"]))
            )
            pair = (request_id, banner_id)
            if pair in truth:
                rrf_found[pair] = pre_rank
        for request_id, values in grouped.items():
            values.sort(key=lambda value: (-value[0], value[1], value[2]))
            predictions[request_id] = (values[0][3], [value[2] for value in values[:50]])
            for rank, value in enumerate(values, start=1):
                pair = (request_id, value[2])
                if pair in truth:
                    ranker_found[pair] = rank
            blended = rank_linear_order(
                values,
                catboost_weight=float(cfg.submission.blend.catboost_weight),
            )
            for rank, value in enumerate(blended, start=1):
                pair = (request_id, value[2])
                if pair in truth:
                    blend_found[pair] = rank
    prediction_rows = [
        {"HitLogID": hit_log_id, "BannerID": banner_ids}
        for _, (hit_log_id, banner_ids) in sorted(predictions.items())
    ]
    prediction_schema = pa.schema(
        [
            pa.field("HitLogID", pa.uint64(), nullable=False),
            pa.field("BannerID", pa.list_(pa.uint64()), nullable=False),
        ]
    )
    prediction_table = pa.Table.from_pylist(prediction_rows, schema=prediction_schema)
    cutoffs = [int(value) for value in cfg.evaluation.cutoffs]
    report = {
        "split": split,
        "requests": len(requests),
        "clicks": len(truth),
        "metrics": {
            "rrf": recall_metrics(records_from_found(truth, rrf_found), cutoffs),
            "catboost": recall_metrics(records_from_found(truth, ranker_found), cutoffs),
            "blend": recall_metrics(records_from_found(truth, blend_found), cutoffs),
        },
    }
    return report, prediction_table


def write_complementarity_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("source\n", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
