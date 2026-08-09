from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.materialize_walk_forward_variant import (
    FINAL_FILES,
    REQUIRED_SHARED_FILES,
    materialize_variant,
)


def complete_final(path: Path) -> None:
    path.mkdir()
    for name in FINAL_FILES:
        (path / name).write_bytes(name.encode())


def test_variant_preserves_source_and_is_idempotent(tmp_path: Path) -> None:
    source = tmp_path / "walk_forward"
    source.mkdir()
    original = tmp_path / "old_final"
    selected = tmp_path / "chrono_final"
    complete_final(original)
    complete_final(selected)
    for name in REQUIRED_SHARED_FILES:
        (source / name).write_bytes(name.encode())
    source_manifest = {
        "status": "completed",
        "snapshots": {"123": {"path": "/snapshot/123"}},
        "final_artifact": str(original),
    }
    (source / "manifest.json").write_text(json.dumps(source_manifest), encoding="utf-8")
    target = tmp_path / "variant"

    first = materialize_variant(
        source_artifact=source,
        target_artifact=target,
        final_artifact=selected,
    )
    source_after = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    variant = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
    second = materialize_variant(
        source_artifact=source,
        target_artifact=target,
        final_artifact=selected,
    )

    assert first["status"] == "completed"
    assert second["status"] == "reused"
    assert source_after == source_manifest
    assert variant["final_artifact"] == str(selected.resolve())
    assert variant["variant"]["source_artifact"] == str(source.resolve())
    assert variant["variant"]["original_final_artifact"] == str(original)
    assert (target / "metrics.json").is_file()
    assert (target / "variant.json").is_file()
    for name in REQUIRED_SHARED_FILES:
        assert (target / name).is_file()
        assert (target / name).samefile(source / name)


def test_variant_rejects_overwrite_and_incomplete_final(tmp_path: Path) -> None:
    source = tmp_path / "walk_forward"
    source.mkdir()
    (source / "manifest.json").write_text(
        json.dumps({"status": "completed", "snapshots": {"1": {}}}),
        encoding="utf-8",
    )
    for name in REQUIRED_SHARED_FILES:
        (source / name).write_bytes(name.encode())
    with pytest.raises(ValueError):
        materialize_variant(
            source_artifact=source,
            target_artifact=source,
            final_artifact=tmp_path / "missing",
        )
    with pytest.raises(FileNotFoundError):
        materialize_variant(
            source_artifact=source,
            target_artifact=tmp_path / "variant",
            final_artifact=tmp_path / "missing",
        )
