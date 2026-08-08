from __future__ import annotations

import gzip
import pickle
from pathlib import Path
from typing import Any

from common.text import normalize


MODEL_FILENAME = "model.pkl.gz"
SOLUTION_NAME = "history_variants"


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
    return [{"name": "variant", "type": "string", "default": "query_sc"}]


def load_model(artifact_dir: Path) -> dict[str, Any]:
    model_path = artifact_dir / MODEL_FILENAME
    if not model_path.is_file():
        raise FileNotFoundError(f"History model does not exist: {model_path}")
    with gzip.open(model_path, "rb") as source:
        model = pickle.load(source)
    if model.get("version") != 1:
        raise ValueError(f"Unsupported history model version: {model.get('version')}")
    return model


def _ordered_rows(
    rankings: dict[tuple[str, str, int], list[tuple[int, int, float, int]]],
    *,
    query: str,
    region_id: int,
    variant: str,
) -> tuple[str, list[tuple[int, int, float, int]]]:
    if variant == "query_click":
        source_name = "query"
        rows = list(rankings.get(("query", query, 0), ()))
        rows.sort(key=lambda row: (-int(row[1]), -float(row[2]), -int(row[3]), int(row[0])))
        return source_name, rows
    if variant == "query_sc":
        source_name = "query"
        rows = list(rankings.get(("query", query, 0), ()))
        rows.sort(key=lambda row: (-float(row[2]), -int(row[1]), -int(row[3]), int(row[0])))
        return source_name, rows
    if variant == "query_region_sc":
        source_name = "query_region"
        rows = list(rankings.get(("query_region", query, region_id), ()))
        rows.sort(key=lambda row: (-float(row[2]), -int(row[1]), -int(row[3]), int(row[0])))
        return source_name, rows
    raise ValueError(f"Unknown history variant: {variant}")


def rank(
    *,
    model: dict[str, Any],
    example: dict[str, Any],
    features: dict[str, Any],
    top_k: int,
) -> list[dict[str, Any]]:
    query = normalize(example.get("query"))
    context = example.get("context") or {}
    region_id = int(context.get("region_id") or 0)
    variant = str(features.get("variant") or "query_sc")
    source_name, rows = _ordered_rows(
        model["rankings"],
        query=query,
        region_id=region_id,
        variant=variant,
    )
    metadata = model["candidates"]
    result = []
    for banner_id, click_count, source_cost_sum, last_show_time in rows[:top_k]:
        candidate = metadata.get(int(banner_id))
        if candidate is None:
            continue
        score = float(click_count if variant == "query_click" else source_cost_sum)
        result.append(
            {
                "banner_id": int(banner_id),
                "title": candidate.get("title", ""),
                "text": candidate.get("text", ""),
                "url": candidate.get("url", ""),
                "source_cost": float(candidate.get("source_cost", 0.0)),
                "score": score,
                "contributions": {
                    "history": {source_name: {}},
                    "click_count": int(click_count),
                    "source_cost_sum": float(source_cost_sum),
                    "last_show_time": int(last_show_time),
                },
                "matched_tokens": [],
            }
        )
    return result
