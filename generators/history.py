from __future__ import annotations

import gzip
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any

from common.text import normalize


MODEL_FILENAME = "model.pkl.gz"
SOLUTION_NAME = "history_candidates"


def input_schema() -> list[dict[str, Any]]:
    return [
        {"name": "query", "path": "query", "type": "textarea", "primary": True},
        {
            "name": "region_id",
            "path": "context.region_id",
            "type": "integer",
            "nullable": True,
        },
    ]


def feature_schema() -> list[dict[str, Any]]:
    return []


def load_model(artifact_dir: Path) -> dict[str, Any]:
    model_path = artifact_dir / MODEL_FILENAME
    if not model_path.is_file():
        raise FileNotFoundError(f"History model does not exist: {model_path}")
    with gzip.open(model_path, "rb") as source:
        model = pickle.load(source)
    if model.get("version") != 1:
        raise ValueError(f"Unsupported history model version: {model.get('version')}")
    return model


def rank(
    *,
    model: dict[str, Any],
    example: dict[str, Any],
    features: dict[str, Any],
    top_k: int,
) -> list[dict[str, Any]]:
    del features
    query = normalize(example.get("query"))
    context = example.get("context") or {}
    region_id = int(context.get("region_id") or 0)
    rankings = model["rankings"]

    # Exact-region history is the precise source; query-only history is its
    # robust backoff. Their reciprocal ranks are merged before global fusion.
    sources = (
        ("query_region", rankings.get(("query_region", query, region_id), ()), 1.0),
        ("query", rankings.get(("query", query, 0), ()), 0.7),
    )
    merged: dict[int, dict[str, Any]] = {}
    for source_name, rows, weight in sources:
        for source_rank, row in enumerate(rows, start=1):
            banner_id, click_count, source_cost_sum, last_show_time = row
            candidate = merged.setdefault(
                int(banner_id),
                {
                    "score": 0.0,
                    "history": {},
                    "click_count": 0,
                    "source_cost_sum": 0.0,
                    "last_show_time": 0,
                },
            )
            contribution = weight / (20.0 + source_rank)
            candidate["score"] += contribution
            candidate["history"][source_name] = {
                "rank": source_rank,
                "score": contribution,
            }
            candidate["click_count"] = max(candidate["click_count"], int(click_count))
            candidate["source_cost_sum"] = max(
                candidate["source_cost_sum"], float(source_cost_sum)
            )
            candidate["last_show_time"] = max(
                candidate["last_show_time"], int(last_show_time)
            )

    ordered = sorted(
        merged.items(),
        key=lambda item: (-float(item[1]["score"]), int(item[0])),
    )[:top_k]
    metadata = model["candidates"]
    result = []
    for banner_id, history in ordered:
        candidate = metadata.get(banner_id)
        # Rankings can outlive a rebuilt one-million-banner index.  Candidate
        # membership must use the same frozen index contract as submission;
        # missing metadata is therefore an invalid candidate, not an empty one.
        if candidate is None:
            continue
        result.append(
            {
                "banner_id": banner_id,
                "title": candidate.get("title", ""),
                "text": candidate.get("text", ""),
                "url": candidate.get("url", ""),
                "source_cost": float(candidate.get("source_cost", 0.0)),
                "score": float(history["score"]),
                "contributions": {
                    "history": history["history"],
                    "click_count": history["click_count"],
                    "source_cost_sum": history["source_cost_sum"],
                    "last_show_time": history["last_show_time"],
                },
                "matched_tokens": [],
            }
        )
    return result
