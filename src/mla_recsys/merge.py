from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from omegaconf import DictConfig

from .candidate_cache import feature_name, source_part_path
from .fusion import fuse_rankings


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


def _source_groups(path: Path) -> dict[str, list[dict[str, Any]]]:
    rows = pq.read_table(path).to_pylist()
    result: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        contributions = {
            "history": {
                name: {}
                for name, present in (
                    ("query", row["history_query_present"]),
                    ("query_region", row["history_region_present"]),
                )
                if present
            },
            "click_count": row["history_click_count"],
            "source_cost_sum": row["history_source_cost_sum"],
        }
        result.setdefault(str(row["request_id"]), []).append(
            {
                "banner_id": int(row["banner_id"]),
                "score": float(row["source_score"]),
                "source_rank": int(row["source_rank"]),
                "_source_rank": int(row["source_rank"]),
                "contributions": contributions,
            }
        )
    for values in result.values():
        values.sort(key=lambda row: (row["source_rank"], row["banner_id"]))
    return result


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
    grouped = {
        source: _source_groups(source_part_path(run_path, split, source, partition))
        for source in sources
    }
    weights = {source: float(cfg.candidates.generators[source].weight) for source in sources}
    quotas = {source: int(cfg.candidates.generators[source].quota) for source in sources}
    output: list[dict[str, Any]] = []
    for request in requests:
        request_id = str(request["request_id"])
        rankings = {source: grouped[source].get(request_id, []) for source in sources}
        fused = fuse_rankings(
            rankings,
            weights=weights,
            quotas=quotas,
            rrf_constant=float(cfg.candidates.rrf_constant),
            max_candidates=int(cfg.candidates.union_max_candidates),
        )
        for pre_rank, candidate in enumerate(fused, start=1):
            row: dict[str, Any] = {
                "request_id": request_id,
                "hit_log_id": int(request["hit_log_id"]),
                "banner_id": int(candidate["banner_id"]),
                "pre_rank": pre_rank,
                "rrf_score": float(candidate["rrf_score"]),
                "source_count": int(candidate["source_count"]),
                "history_click_count": 0,
                "history_source_cost_sum": 0.0,
                "history_query_present": False,
                "history_region_present": False,
            }
            retrieval = candidate["retrieval"]
            for source in sources:
                alias = feature_name(cfg, source)
                value = retrieval.get(source)
                row[f"{alias}__present"] = value is not None
                row[f"{alias}__rank"] = int(value["rank"]) if value else 0
                row[f"{alias}__score"] = float(value.get("score") or 0.0) if value else 0.0
                if alias == "history" and value:
                    contributions = value.get("contributions") or {}
                    history_sources = contributions.get("history") or {}
                    row["history_click_count"] = int(contributions.get("click_count") or 0)
                    row["history_source_cost_sum"] = float(
                        contributions.get("source_cost_sum") or 0.0
                    )
                    row["history_query_present"] = "query" in history_sources
                    row["history_region_present"] = "query_region" in history_sources
            output.append(row)
    return pa.Table.from_pylist(output, schema=merged_schema(cfg))
