from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from mla_recsys.pipeline import MultiGeneratorPipeline  # noqa: E402

SOLUTION_NAME = "mla_two_stage_rrf"


def input_schema() -> list[dict[str, Any]]:
    return [
        {"name": "query", "path": "query", "label": "Запрос", "type": "textarea", "primary": True},
        {"name": "region_id", "path": "context.region_id", "label": "Регион", "type": "integer", "nullable": True},
        {"name": "device", "path": "context.device", "label": "Device", "type": "text", "nullable": True},
    ]


def feature_schema() -> list[dict[str, Any]]:
    return []


def load_model(artifact_dir: Path) -> dict[str, Any]:
    config_path = artifact_dir / "config.json"
    if not config_path.is_file():
        config_path = REPOSITORY_ROOT / "configs" / "baselines.json"
    pipeline = MultiGeneratorPipeline.from_config(config_path)
    return {"pipeline": pipeline, "metadata": {"solution": SOLUTION_NAME, "config": str(config_path)}}


def rank(
    *,
    model: dict[str, Any],
    example: dict[str, Any],
    features: dict[str, Any],
    top_k: int,
) -> list[dict[str, Any]]:
    del features
    return model["pipeline"].rank(example, top_k)

