from __future__ import annotations

import pytest

from mla_recsys.rank_blend import rank_linear_order


def test_rank_linear_order_combines_catboost_and_pre_rank() -> None:
    values = [
        (0.9, 3, 30, 300),
        (0.8, 1, 10, 100),
        (0.7, 2, 20, 200),
    ]

    assert [value[2] for value in rank_linear_order(values, catboost_weight=1.0)] == [
        30,
        10,
        20,
    ]
    assert [value[2] for value in rank_linear_order(values, catboost_weight=0.0)] == [
        10,
        20,
        30,
    ]
    assert [value[2] for value in rank_linear_order(values, catboost_weight=0.5)] == [
        10,
        30,
        20,
    ]


def test_rank_linear_order_rejects_invalid_weight() -> None:
    with pytest.raises(ValueError, match="catboost_weight"):
        rank_linear_order([], catboost_weight=1.1)
