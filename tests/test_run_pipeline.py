from __future__ import annotations

from mla_recsys.config import compose_config
from scripts.run_pipeline import stage_commands


def test_semantic_overrides_are_forwarded_to_every_stage() -> None:
    overrides = ["ranker.depth=7", "features.version=feature_test_v1"]
    cfg = compose_config(
        "i0_reproduce",
        run_id="20260808_0816_override",
        mode="smoke",
        overrides=overrides,
    )
    commands = stage_commands(cfg, overrides)
    assert commands
    for _, command in commands:
        assert "ranker.depth=7" in command
        assert "features.version=feature_test_v1" in command
