from __future__ import annotations

from pathlib import Path

import json

from scripts.continue_walk_forward_pipeline import (
    pipeline_command,
    promote_final_artifact,
)


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


def test_final_artifact_promotion_is_atomic_and_fingerprinted(tmp_path: Path) -> None:
    artifact = tmp_path / "walk_forward"
    original = artifact / "final"
    selected = tmp_path / "full_quality"
    artifact.mkdir()
    for directory in (original, selected):
        directory.mkdir(parents=True)
        for name in (
            "model.pt",
            "candidate_embeddings.npy",
            "candidate_metadata.parquet",
            "manifest.json",
        ):
            (directory / name).write_bytes(name.encode())
    manifest = {"status": "completed", "final_artifact": str(original)}
    (artifact / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    result = promote_final_artifact(artifact, selected)
    promoted = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    assert result["previous"] == str(original)
    assert promoted["final_artifact"] == str(selected)
    assert promoted["final_artifact_source"] == "configured_full_quality_override"
    assert len(promoted["final_artifact_inputs"]) == 4
