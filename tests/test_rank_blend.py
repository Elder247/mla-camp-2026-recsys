from __future__ import annotations

import pytest

from mla_recsys.rank_blend import rank_linear_order, rank_value_geometric_order


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


def test_value_geometry_zero_exponent_matches_base_blend() -> None:
    values = [
        (0.9, 3, 30, 300, 1.0),
        (0.8, 1, 10, 100, 1_000_000_000.0),
        (0.7, 2, 20, 200, 10.0),
    ]

    ordered = rank_value_geometric_order(
        values,
        catboost_weight=0.5,
        source_cost_scale=1_000_000.0,
        exponent=0.0,
        rerank_top_n=3,
    )

    assert [value[2] for value in ordered] == [10, 30, 20]


def test_value_geometry_promotes_value_only_inside_prefix() -> None:
    values = [
        (0.9, 1, 10, 100, 1.0),
        (0.8, 2, 20, 200, 1.0),
        (0.7, 3, 30, 300, 1_000_000_000.0),
        (0.6, 4, 40, 400, 1_000_000_000.0),
    ]

    ordered = rank_value_geometric_order(
        values,
        catboost_weight=1.0,
        source_cost_scale=1_000_000.0,
        exponent=1.0,
        rerank_top_n=3,
    )

    assert [value[2] for value in ordered] == [30, 10, 20, 40]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("source_cost_scale", 0.0, "source_cost_scale"),
        ("exponent", -0.1, "exponent"),
        ("rerank_top_n", 0, "rerank_top_n"),
    ],
)
def test_value_geometry_rejects_invalid_parameters(
    field: str, value: float, message: str
) -> None:
    kwargs = {
        "catboost_weight": 0.6,
        "source_cost_scale": 1_000_000.0,
        "exponent": 0.1,
        "rerank_top_n": 100,
    }
    kwargs[field] = value
    with pytest.raises(ValueError, match=message):
        rank_value_geometric_order([], **kwargs)
