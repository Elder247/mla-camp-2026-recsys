from __future__ import annotations

import numpy as np
import pytest

from scripts.tune_model_ensemble import normalized, parse_grid, rank_positions


def test_parse_grid_is_bounded_and_deduplicated() -> None:
    assert parse_grid("0.75,0.5,0.75") == [0.5, 0.75]
    with pytest.raises(ValueError, match="\[0, 1\]"):
        parse_grid("1.1")


def test_normalized_constant_values_are_zero() -> None:
    assert normalized(np.array([4.0, 4.0])).tolist() == [0.0, 0.0]


def test_rank_positions_use_pre_rank_then_banner_for_ties() -> None:
    scores = np.array([1.0, 1.0, 1.0])
    pre_rank = np.array([2, 1, 1])
    banners = np.array([1, 20, 10])

    assert rank_positions(scores, pre_rank, banners).tolist() == [3, 2, 1]
