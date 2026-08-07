from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .features import extract_feature_rows


MODEL_FILENAME = "ranker.cbm"
METADATA_FILENAME = "ranker.json"


def load_ranker(artifact_dir: Path) -> dict[str, Any] | None:
    model_path = artifact_dir / MODEL_FILENAME
    metadata_path = artifact_dir / METADATA_FILENAME
    if not model_path.is_file():
        return None
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Ranker metadata does not exist: {metadata_path}")
    from catboost import CatBoostRanker

    model = CatBoostRanker()
    model.load_model(str(model_path))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return {"model": model, "metadata": metadata}


def rerank(
    ranker: dict[str, Any],
    example: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    generator_names = ranker["metadata"]["generator_names"]
    feature_rows = extract_feature_rows(example, candidates, generator_names)
    predictions = ranker["model"].predict(feature_rows)
    for candidate, prediction in zip(candidates, predictions):
        candidate["ranker_score"] = float(prediction)
        candidate["score"] = float(prediction)
        candidate["contributions"]["catboost"] = float(prediction)
    candidates.sort(
        key=lambda candidate: (
            -float(candidate["ranker_score"]),
            -float(candidate["rrf_score"]),
            int(candidate["banner_id"]),
        )
    )
    return candidates
