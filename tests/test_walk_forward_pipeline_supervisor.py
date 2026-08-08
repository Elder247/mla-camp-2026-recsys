from __future__ import annotations

from pathlib import Path

from scripts.continue_walk_forward_pipeline import pipeline_command


def test_pipeline_command_points_code_to_worktree_and_artifacts_to_shared_root() -> None:
    command = pipeline_command(
        python=Path("/venv/python"),
        experiment="i2_walk_forward_10m_fast_quality",
        run_id="20260808_1815_i2_wf10m_smoke",
        mode="smoke",
        scope="offline",
        output_runs=Path("/shared/runs"),
        cache=Path("/shared/cache"),
        immutable_artifacts=Path("/shared/artifacts"),
    )
    assert "paths.root=" + str(Path(__file__).resolve().parents[1]) in command
    assert "paths.runs=/shared/runs" in command
    assert "paths.immutable_artifacts=/shared/artifacts" in command
    assert "mode=smoke" in command
