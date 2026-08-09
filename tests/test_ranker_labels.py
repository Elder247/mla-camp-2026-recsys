from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest
from omegaconf import OmegaConf

from scripts.train_ranker import filter_training_window, group_weight_array, label_spec


def test_raw_sourcecost_label_is_not_log_surrogate() -> None:
    raw = OmegaConf.create(
        {"ranker": {"kind": "ranker_raw_sc_label", "raw_sc_scale": 1_000_000.0}}
    )
    log = OmegaConf.create(
        {"ranker": {"kind": "ranker_logsc", "raw_sc_scale": 1_000_000.0}}
    )
    assert label_spec(raw) == ("label_raw_sc", 1_000_000.0)
    assert label_spec(log) == ("label_logsc", 1.0)


def test_binary_ranker_uses_natural_pool_click_label() -> None:
    binary = OmegaConf.create({"ranker": {"kind": "ranker_binary"}})

    assert label_spec(binary) == ("label_binary", 1.0)


def test_source_cost_group_weights_are_clipped_normalized_and_expanded() -> None:
    table = pa.table(
        {
            "group_id": [10, 10, 20, 20, 20, 30],
            "label_raw_sc": [4.0, 6.0, 0.0, 100.0, 0.0, 1000.0],
        }
    )
    cfg = OmegaConf.create(
        {
            "ranker": {
                "group_weight": {
                    "kind": "source_cost",
                    "cap_quantile": 0.5,
                    "power": 1.0,
                    "minimum": 1.0e-6,
                }
            }
        }
    )

    weights, stats = group_weight_array(table, cfg)

    expected_groups = np.array([10.0, 100.0, 100.0])
    expected_groups /= expected_groups.mean()
    np.testing.assert_allclose(
        weights,
        np.repeat(expected_groups, [2, 3, 1]),
        rtol=1.0e-6,
    )
    assert stats["groups"] == 3
    assert stats["cap_value"] == pytest.approx(100.0)
    assert stats["normalized_mean"] == pytest.approx(1.0)


def test_group_weight_requires_contiguous_groups() -> None:
    table = pa.table(
        {
            "group_id": [10, 20, 10],
            "label_raw_sc": [1.0, 2.0, 0.0],
        }
    )
    cfg = OmegaConf.create(
        {"ranker": {"group_weight": {"kind": "source_cost"}}}
    )

    with pytest.raises(ValueError, match="contiguous"):
        group_weight_array(table, cfg)


def test_training_window_keeps_complete_recent_request_groups() -> None:
    features = pa.table(
        {
            "request_id": ["old", "old", "recent", "recent", "latest"],
            "group_id": [1, 1, 2, 2, 3],
        }
    )
    requests = pa.table(
        {
            "request_id": ["old", "recent", "latest"],
            "show_time": [100, 200_000, 300_000],
        }
    )
    cfg = OmegaConf.create({"ranker": {"training_window_days": 2.0}})

    filtered, stats = filter_training_window(features, requests, cfg)

    assert filtered["request_id"].to_pylist() == ["recent", "recent", "latest"]
    assert stats["cutoff_show_time"] == 127_200
    assert stats["requests_after"] == 2
    assert stats["rows_before"] == 5
    assert stats["rows_after"] == 3


def test_zero_training_window_preserves_all_rows() -> None:
    features = pa.table({"group_id": [1, 1, 2]})
    requests = pa.table({"request_id": ["a"], "show_time": [100]})
    cfg = OmegaConf.create({"ranker": {"training_window_days": 0}})

    filtered, stats = filter_training_window(features, requests, cfg)

    assert filtered is features
    assert stats == {
        "enabled": False,
        "days": 0.0,
        "rows_before": 3,
        "rows_after": 3,
    }


def test_negative_training_window_is_rejected() -> None:
    cfg = OmegaConf.create({"ranker": {"training_window_days": -1}})

    with pytest.raises(ValueError, match="non-negative"):
        filter_training_window(pa.table({"group_id": [1]}), pa.table({}), cfg)
