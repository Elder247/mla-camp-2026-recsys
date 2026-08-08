from __future__ import annotations

import json
from pathlib import Path

from scripts.continue_walk_forward_training import validate_walk_forward_artifact


def test_walk_forward_artifact_contract(tmp_path: Path) -> None:
    snapshot = tmp_path / "snapshots" / "100"
    final = tmp_path / "final"
    for path in (snapshot, final):
        path.mkdir(parents=True)
        for name in (
            "model.pt",
            "candidate_embeddings.npy",
            "candidate_metadata.parquet",
            "manifest.json",
        ):
            (path / name).write_bytes(b"x")
    manifest = {
        "status": "completed",
        "weeks": [100],
        "snapshots": {"100": {"path": str(snapshot)}},
        "final_artifact": str(final),
        "oof_requests": str(tmp_path / "oof.parquet"),
        "validation_health": {"in_batch_accuracy": 0.8},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    result = validate_walk_forward_artifact(tmp_path)
    assert result["weeks"] == 1
    assert result["snapshots"] == 1
