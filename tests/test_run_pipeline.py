from __future__ import annotations

import time

import pytest

from mla_recsys.config import compose_config
from scripts.run_pipeline import (
    _candidate_source,
    enforce_run_budget,
    execution_groups,
    stage_commands,
)


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


def test_i1_parallel_groups_never_overlap_gpu_and_pair_split_work() -> None:
    cfg = compose_config(
        "i1_more_cg_features_sc",
        run_id="20260808_0817_parallel",
        mode="smoke",
    )
    groups = execution_groups(cfg, stage_commands(cfg))
    assert max(map(len, groups)) == 3
    generate_groups = [
        group for group in groups if group[0][0].startswith("generate_")
    ]
    for group in generate_groups:
        gpu_sources = [
            _candidate_source(command)
            for _, command in group
            if str(
                cfg.candidates.generators[_candidate_source(command)].get(
                    "resource", "cpu"
                )
            )
            == "gpu"
        ]
        assert len(gpu_sources) <= 1
    first_sources = {_candidate_source(command) for _, command in generate_groups[0]}
    assert "tfidf_v1" in first_sources
    assert "two_tower_fps_v1" in first_sources
    assert any(
        {stage for stage, _ in group}
        == {"merge_candidates_train", "merge_candidates_holdout"}
        for group in groups
    )
    assert any(
        {stage for stage, _ in group}
        == {"build_features_train", "build_features_holdout"}
        for group in groups
    )


def test_rrf_full_skips_feature_and_ranker_stages() -> None:
    cfg = compose_config(
        "i1_fast_value",
        run_id="20260808_1700_rrf_full",
        mode="full",
        scope="full",
        overrides=["submission.ranking=rrf"],
    )
    names = [name for name, _ in stage_commands(cfg)]
    assert "prepare_counters" not in names
    assert not any(name.startswith("build_features") for name in names)
    assert "train_ranker" not in names
    assert "make_submission" in names


def test_pipeline_wall_budget_can_be_disabled_or_exhausted() -> None:
    enforce_run_budget(started=time.monotonic() - 10.0, max_wall_seconds=0)
    with pytest.raises(TimeoutError):
        enforce_run_budget(started=time.monotonic() - 10.0, max_wall_seconds=1)
