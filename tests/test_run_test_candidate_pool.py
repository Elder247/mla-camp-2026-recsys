from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from scripts.run_test_candidate_pool import configured_overrides


def values(**updates: object) -> argparse.Namespace:
    raw = {
        "source": "two_tower_v2",
        "artifact_dir": Path("/artifacts/model"),
        "runs": Path("/runs"),
        "cache": Path("/cache"),
        "immutable_artifacts": Path("/artifacts"),
        "quota": 100,
        "batch_size": 1024,
    }
    raw.update(updates)
    return argparse.Namespace(**raw)


def test_candidate_only_overrides_keep_quality_and_speed_controls_explicit() -> None:
    overrides = configured_overrides(values())

    assert "paths.two_tower_v2_artifact=/artifacts/model" in overrides
    assert "candidates.generators.two_tower_v2.top_k=100" in overrides
    assert "candidates.generators.two_tower_v2.quota=100" in overrides
    assert "candidates.generators.two_tower_v2.batch_size=1024" in overrides


@pytest.mark.parametrize("name", ["quota", "batch_size"])
def test_candidate_only_rejects_nonpositive_controls(name: str) -> None:
    with pytest.raises(ValueError, match="positive"):
        configured_overrides(values(**{name: 0}))
