from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .fusion import fuse_rankings
from .loading import load_module


@dataclass
class Generator:
    name: str
    module: Any
    model: Any
    top_k: int
    quota: int
    weight: float
    features: dict[str, Any]
    batch_size: int = 1

    def rank(self, example: dict[str, Any]) -> list[dict[str, Any]]:
        result = self.module.rank(
            model=self.model,
            example=example,
            features=self.features,
            top_k=self.top_k,
        )
        return [dict(item) for item in result]

    def rank_batch(self, examples: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        if hasattr(self.module, "rank_batch"):
            result = self.module.rank_batch(
                model=self.model,
                examples=examples,
                features=self.features,
                top_k=self.top_k,
            )
            if len(result) != len(examples):
                raise ValueError(
                    f"{self.name}.rank_batch returned {len(result)} rows for "
                    f"{len(examples)} examples"
                )
            return [[dict(item) for item in ranking] for ranking in result]
        return [self.rank(example) for example in examples]


class MultiGeneratorPipeline:
    def __init__(self, generators: list[Generator], fusion: dict[str, Any]) -> None:
        if not generators:
            raise ValueError("At least one generator is required")
        names = [generator.name for generator in generators]
        if len(set(names)) != len(names):
            raise ValueError(f"Generator names must be unique: {names}")
        self.generators = generators
        self.fusion = fusion

    @classmethod
    def from_config(cls, path: str | Path) -> "MultiGeneratorPipeline":
        config_path = Path(path).expanduser().resolve()
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("version") != 1:
            raise ValueError(f"Unsupported config version: {config.get('version')}")
        generators = []
        for item in config["generators"]:
            module = load_module(item["code"], item.get("python_paths"))
            artifact_dir = Path(item["artifact_dir"]).expanduser().resolve()
            model = module.load_model(artifact_dir)
            default_features = {}
            if hasattr(module, "feature_schema"):
                default_features = {
                    str(feature["name"]): feature.get("default")
                    for feature in module.feature_schema()
                    if "name" in feature and "default" in feature
                }
            configured_features = {
                **default_features,
                **dict(item.get("features") or {}),
            }
            generators.append(
                Generator(
                    name=str(item["name"]),
                    module=module,
                    model=model,
                    top_k=int(item["top_k"]),
                    quota=int(item.get("quota", item["top_k"])),
                    weight=float(item.get("weight", 1.0)),
                    features=configured_features,
                    batch_size=int(item.get("batch_size", 1)),
                )
            )
        return cls(generators, dict(config.get("fusion") or {}))

    def source_rankings(self, example: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
        return {generator.name: generator.rank(example) for generator in self.generators}

    def fuse(
        self,
        rankings: dict[str, list[dict[str, Any]]],
        *,
        max_candidates: int | None = None,
    ) -> list[dict[str, Any]]:
        configured_max = int(self.fusion.get("max_candidates", 2000))
        return fuse_rankings(
            rankings,
            weights={generator.name: generator.weight for generator in self.generators},
            quotas={generator.name: generator.quota for generator in self.generators},
            rrf_constant=float(self.fusion.get("rrf_constant", 60.0)),
            max_candidates=max_candidates or configured_max,
        )

    def rank(self, example: dict[str, Any], top_k: int) -> list[dict[str, Any]]:
        rankings = self.source_rankings(example)
        return self.fuse(rankings, max_candidates=max(top_k, int(self.fusion.get("max_candidates", 2000))))[:top_k]
