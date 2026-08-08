from __future__ import annotations

import numpy as np
import pyarrow as pa
import pytest
from omegaconf import OmegaConf

from scripts.train_ranker import group_weight_array, label_spec


def test_raw_sourcecost_label_is_not_log_surrogate() -> None:
    raw = OmegaConf.create(
        {"ranker": {"kind": "ranker_raw_sc_label", "raw_sc_scale": 1_000_000.0}}
    )
    log = OmegaConf.create(
        {"ranker": {"kind": "ranker_logsc", "raw_sc_scale": 1_000_000.0}}
    )
    assert label_spec(raw) == ("label_raw_sc", 1_000_000.0)
    assert label_spec(log) == ("label_logsc", 1.0)


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
