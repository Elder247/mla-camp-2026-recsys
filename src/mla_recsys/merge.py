from __future__ import annotations

import heapq
from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from omegaconf import DictConfig

from .candidate_cache import feature_name, source_part_path


def merged_schema(cfg: DictConfig) -> pa.Schema:
    fields = [
        pa.field("request_id", pa.string(), nullable=False),
        pa.field("hit_log_id", pa.uint64(), nullable=False),
        pa.field("banner_id", pa.uint64(), nullable=False),
        pa.field("pre_rank", pa.int32(), nullable=False),
        pa.field("rrf_score", pa.float64(), nullable=False),
        pa.field("source_count", pa.int16(), nullable=False),
        pa.field("history_click_count", pa.int64(), nullable=False),
        pa.field("history_source_cost_sum", pa.float64(), nullable=False),
        pa.field("history_query_present", pa.bool_(), nullable=False),
        pa.field("history_region_present", pa.bool_(), nullable=False),
    ]
    for source, item in cfg.candidates.generators.items():
        if not bool(item.get("enabled", False)):
            continue
        alias = feature_name(cfg, str(source))
        fields.extend(
            [
                pa.field(f"{alias}__present", pa.bool_(), nullable=False),
                pa.field(f"{alias}__rank", pa.int32(), nullable=False),
                pa.field(f"{alias}__score", pa.float64(), nullable=False),
            ]
        )
    return pa.schema(fields)


SourceRow = tuple[int, int, float, int, float, bool, bool]


def _source_groups(path: Path) -> dict[str, list[SourceRow]]:
    """Read one source without materialising a dict for every candidate row."""

    columns = pq.read_table(path).to_pydict()
    result: dict[str, list[SourceRow]] = defaultdict(list)
    for values in zip(
        columns["request_id"],
        columns["banner_id"],
        columns["source_rank"],
        columns["source_score"],
        columns["history_click_count"],
        columns["history_source_cost_sum"],
        columns["history_query_present"],
        columns["history_region_present"],
    ):
        request_id, banner_id, rank, score, clicks, source_cost, query, region = values
        result[str(request_id)].append(
            (
                int(banner_id),
                int(rank),
                float(score),
                int(clicks),
                float(source_cost),
                bool(query),
                bool(region),
            )
        )
    for rows in result.values():
        rows.sort(key=lambda row: (row[1], row[0]))
    return dict(result)


def merge_partition(
    *,
    cfg: DictConfig,
    run_path: Path,
    split: str,
    partition: int,
    requests: list[dict[str, Any]],
) -> pa.Table:
    sources = [
        str(name)
        for name, item in cfg.candidates.generators.items()
        if bool(item.get("enabled", False))
    ]
    grouped: dict[str, dict[str, list[SourceRow]]] = {
        source: _source_groups(source_part_path(run_path, split, source, partition))
        for source in sources
    }
    weights = {source: float(cfg.candidates.generators[source].weight) for source in sources}
    quotas = {source: int(cfg.candidates.generators[source].quota) for source in sources}
    aliases = {source: feature_name(cfg, source) for source in sources}
    rrf_constant = float(cfg.candidates.rrf_constant)
    max_candidates = int(cfg.candidates.union_max_candidates)
    output: list[dict[str, Any]] = []
    for request in requests:
        request_id = str(request["request_id"])
        # banner_id -> compact mutable state. The former implementation made a
        # deepcopy-heavy candidate object for every source row; only the fixed
        # retrieval fields below are needed by the parquet contract.
        merged: dict[int, dict[str, Any]] = {}
        for source in sources:
            accepted = 0
            seen: set[int] = set()
            for banner_id, rank, score, clicks, source_cost, query, region in grouped[
                source
            ].get(request_id, []):
                if banner_id in seen:
                    continue
                seen.add(banner_id)
                if accepted >= quotas[source]:
                    break
                accepted += 1
                state = merged.setdefault(
                    banner_id,
                    {
                        "rrf_score": 0.0,
                        "retrieval": {},
                        "history_click_count": 0,
                        "history_source_cost_sum": 0.0,
                        "history_query_present": False,
                        "history_region_present": False,
                    },
                )
                state["rrf_score"] += weights[source] / (rrf_constant + rank)
                state["retrieval"][source] = (rank, score)
                if aliases[source] == "history":
                    state["history_click_count"] = clicks
                    state["history_source_cost_sum"] = source_cost
                    state["history_query_present"] = query
                    state["history_region_present"] = region

        ranked = heapq.nsmallest(
            min(max_candidates, len(merged)),
            merged.items(),
            key=lambda item: (
                -float(item[1]["rrf_score"]),
                -len(item[1]["retrieval"]),
                int(item[0]),
            ),
        )
        for pre_rank, (banner_id, state) in enumerate(ranked, start=1):
            row: dict[str, Any] = {
                "request_id": request_id,
                "hit_log_id": int(request["hit_log_id"]),
                "banner_id": int(banner_id),
                "pre_rank": pre_rank,
                "rrf_score": float(state["rrf_score"]),
                "source_count": len(state["retrieval"]),
                "history_click_count": int(state["history_click_count"]),
                "history_source_cost_sum": float(state["history_source_cost_sum"]),
                "history_query_present": bool(state["history_query_present"]),
                "history_region_present": bool(state["history_region_present"]),
            }
            retrieval = state["retrieval"]
            for source in sources:
                alias = aliases[source]
                value = retrieval.get(source)
                row[f"{alias}__present"] = value is not None
                row[f"{alias}__rank"] = int(value[0]) if value else 0
                row[f"{alias}__score"] = float(value[1]) if value else 0.0
            output.append(row)
    return pa.Table.from_pylist(output, schema=merged_schema(cfg))
