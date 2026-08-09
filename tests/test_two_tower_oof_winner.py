from __future__ import annotations

import json

import pytest

from scripts.continue_two_tower_oof_winner import selected_variant


def test_selected_variant_maps_completed_selection(tmp_path) -> None:
    path = tmp_path / "selection.json"
    path.write_text(
        json.dumps({"status": "completed", "selected": "v6_context_metadata"})
    )

    name, variant = selected_variant(path)

    assert name == "v6_context_metadata"
    assert str(variant["config"]).endswith(
        "v6_context_metadata_walk_forward_100m_s10.yaml"
    )


def test_selected_variant_rejects_incomplete_selection(tmp_path) -> None:
    path = tmp_path / "selection.json"
    path.write_text(json.dumps({"status": "training_100m"}))

    with pytest.raises(RuntimeError, match="not completed"):
        selected_variant(path)
