from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any


def _candidate_id(candidate: Mapping[str, Any]) -> int:
    if "banner_id" not in candidate:
        raise KeyError("Every candidate must contain banner_id")
    return int(candidate["banner_id"])


def fuse_rankings(
    rankings: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    weights: Mapping[str, float] | None = None,
    quotas: Mapping[str, int] | None = None,
    rrf_constant: float = 60.0,
    max_candidates: int = 2000,
) -> list[dict[str, Any]]:
    """Deduplicate source rankings and order the union with weighted RRF.

    Raw source ranks and scores are preserved under ``retrieval``.  The
    metadata is intentionally shaped as future CatBoost features; RRF itself
    is only the deterministic first-stage baseline.
    """

    if rrf_constant <= 0:
        raise ValueError("rrf_constant must be positive")
    if max_candidates <= 0:
        raise ValueError("max_candidates must be positive")

    source_weights = {name: float((weights or {}).get(name, 1.0)) for name in rankings}
    source_quotas = {
        name: max(0, int((quotas or {}).get(name, len(items))))
        for name, items in rankings.items()
    }
    merged: dict[int, dict[str, Any]] = {}

    for source, candidates in rankings.items():
        weight = source_weights[source]
        if weight < 0:
            raise ValueError(f"Negative fusion weight for {source}: {weight}")
        seen_in_source: set[int] = set()
        accepted = 0
        for source_rank, raw_candidate in enumerate(candidates, start=1):
            banner_id = _candidate_id(raw_candidate)
            if banner_id in seen_in_source:
                continue
            seen_in_source.add(banner_id)
            if accepted >= source_quotas[source]:
                break
            accepted += 1

            item = merged.get(banner_id)
            if item is None:
                item = deepcopy(dict(raw_candidate))
                item["banner_id"] = banner_id
                item["retrieval"] = {}
                item["rrf_score"] = 0.0
                merged[banner_id] = item

            raw_score = raw_candidate.get("score")
            item["retrieval"][source] = {
                "rank": source_rank,
                "reciprocal_rank": 1.0 / source_rank,
                "score": float(raw_score) if raw_score is not None else None,
                "contributions": deepcopy(raw_candidate.get("contributions")),
            }
            item["rrf_score"] += weight / (rrf_constant + source_rank)

    fused = list(merged.values())
    for item in fused:
        retrieval = item["retrieval"]
        item["source_count"] = len(retrieval)
        item["sources"] = sorted(retrieval)
        item["score"] = float(item["rrf_score"])
        previous = item.get("contributions")
        item["contributions"] = {
            "rrf": float(item["rrf_score"]),
            "source_count": float(item["source_count"]),
            "source_scores": retrieval,
            "original": previous,
        }

    fused.sort(
        key=lambda item: (
            -float(item["rrf_score"]),
            -int(item["source_count"]),
            int(item["banner_id"]),
        )
    )
    return fused[:max_candidates]
