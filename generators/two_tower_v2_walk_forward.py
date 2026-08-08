from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import torch

from mla_recsys.counters import week_start

from generators import two_tower_v2_batch as base


SOLUTION_NAME = "two_tower_v2_walk_forward"
input_schema = base.input_schema
feature_schema = base.feature_schema


def load_model(artifact_dir: Path) -> dict[str, Any]:
    manifest_path = artifact_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Walk-forward manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise RuntimeError(
            f"Walk-forward artifact is not completed: {manifest.get('status')}"
        )
    snapshots = {
        int(start): Path(str(value["path"]))
        for start, value in dict(manifest.get("snapshots") or {}).items()
    }
    if not snapshots:
        raise RuntimeError("Walk-forward artifact has no weekly snapshots")
    final = Path(str(manifest.get("final_artifact") or ""))
    if not (final / "manifest.json").is_file():
        raise FileNotFoundError(f"Final walk-forward snapshot is missing: {final}")
    return {
        "artifact_dir": artifact_dir,
        "snapshots": snapshots,
        "final": final,
        "active_path": None,
        "active_model": None,
        "metadata": {
            "solution": SOLUTION_NAME,
            "weeks": sorted(snapshots),
            "final": str(final),
        },
    }


def _artifact_for(model: dict[str, Any], example: dict[str, Any]) -> Path:
    timestamp = example.get("show_time")
    if timestamp is not None:
        snapshot = model["snapshots"].get(week_start(int(timestamp)))
        if snapshot is not None:
            return snapshot
    return model["final"]


def _activate(model: dict[str, Any], path: Path) -> dict[str, Any]:
    if model["active_path"] == path and model["active_model"] is not None:
        return model["active_model"]
    model["active_model"] = None
    model["active_path"] = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    loaded = base.load_model(path)
    model["active_model"] = loaded
    model["active_path"] = path
    return loaded


def rank_batch(
    *,
    model: dict[str, Any],
    examples: list[dict[str, Any]],
    features: dict[str, Any],
    top_k: int,
) -> list[list[dict[str, Any]]]:
    if not examples:
        return []
    output: list[list[dict[str, Any]] | None] = [None] * len(examples)
    position = 0
    while position < len(examples):
        artifact = _artifact_for(model, examples[position])
        end = position + 1
        while end < len(examples) and _artifact_for(model, examples[end]) == artifact:
            end += 1
        active = _activate(model, artifact)
        ranked = base.rank_batch(
            model=active,
            examples=examples[position:end],
            features=features,
            top_k=top_k,
        )
        output[position:end] = ranked
        position = end
    return [row for row in output if row is not None]


def rank(
    *,
    model: dict[str, Any],
    example: dict[str, Any],
    features: dict[str, Any],
    top_k: int,
) -> list[dict[str, Any]]:
    return rank_batch(
        model=model,
        examples=[example],
        features=features,
        top_k=top_k,
    )[0]
