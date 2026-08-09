from __future__ import annotations

import math
from collections.abc import Iterable


RankedCandidate = tuple[float, int, int, int]
ValueRankedCandidate = tuple[float, int, int, int, float]
TwoModelValueCandidate = tuple[float, float, int, int, int, float]


def rank_linear_order(
    values: Iterable[RankedCandidate],
    *,
    catboost_weight: float,
) -> list[RankedCandidate]:
    """Blend CatBoost and RRF positions; lower blended position is better."""

    if not 0.0 <= catboost_weight <= 1.0:
        raise ValueError("catboost_weight must be in [0, 1]")
    catboost = sorted(values, key=lambda value: (-value[0], value[1], value[2]))
    weighted = [
        (
            catboost_weight * catboost_rank
            + (1.0 - catboost_weight) * value[1],
            value[1],
            value[2],
            value,
        )
        for catboost_rank, value in enumerate(catboost, start=1)
    ]
    weighted.sort(key=lambda item: (item[0], item[1], item[2]))
    return [item[3] for item in weighted]


def two_model_rank_linear_order(
    values: Iterable[TwoModelValueCandidate],
    *,
    model_a_weight: float,
    catboost_weight: float,
) -> list[TwoModelValueCandidate]:
    """Blend two CatBoost ranks, then blend the ensemble rank with RRF."""

    if not 0.0 <= model_a_weight <= 1.0:
        raise ValueError("model_a_weight must be in [0, 1]")
    if not 0.0 <= catboost_weight <= 1.0:
        raise ValueError("catboost_weight must be in [0, 1]")
    rows = list(values)
    model_a = sorted(rows, key=lambda value: (-value[0], value[2], value[3]))
    model_b = sorted(rows, key=lambda value: (-value[1], value[2], value[3]))
    rank_a = {(value[3], value[4]): rank for rank, value in enumerate(model_a, 1)}
    rank_b = {(value[3], value[4]): rank for rank, value in enumerate(model_b, 1)}
    return sorted(
        rows,
        key=lambda value: (
            catboost_weight
            * (
                model_a_weight * rank_a[(value[3], value[4])]
                + (1.0 - model_a_weight) * rank_b[(value[3], value[4])]
            )
            + (1.0 - catboost_weight) * value[2],
            value[2],
            value[3],
        ),
    )


def rank_value_geometric_order(
    values: Iterable[ValueRankedCandidate],
    *,
    catboost_weight: float,
    source_cost_scale: float,
    exponent: float,
    rerank_top_n: int,
) -> list[ValueRankedCandidate]:
    """Apply a bounded SourceCost prior to the best relevance candidates.

    The base order is the honest CatBoost/RRF rank blend. Within its first
    ``rerank_top_n`` rows the effective rank is
    ``rank / (1 + SourceCost / scale) ** exponent``. Restricting the prior to
    a relevance-qualified prefix prevents expensive but unrelated tail items
    from flooding top-50. ``exponent=0`` exactly reproduces the base blend.
    """

    if not 0.0 <= catboost_weight <= 1.0:
        raise ValueError("catboost_weight must be in [0, 1]")
    if source_cost_scale <= 0.0:
        raise ValueError("source_cost_scale must be positive")
    if exponent < 0.0:
        raise ValueError("exponent must be non-negative")
    if rerank_top_n <= 0:
        raise ValueError("rerank_top_n must be positive")
    rows = list(values)
    catboost = sorted(rows, key=lambda value: (-value[0], value[1], value[2]))
    catboost_rank = {
        (value[2], value[3]): rank
        for rank, value in enumerate(catboost, start=1)
    }
    base = sorted(
        rows,
        key=lambda value: (
            catboost_weight * catboost_rank[(value[2], value[3])]
            + (1.0 - catboost_weight) * value[1],
            value[1],
            value[2],
        ),
    )
    return value_geometric_from_base_order(
        base,
        source_cost_scale=source_cost_scale,
        exponent=exponent,
        rerank_top_n=rerank_top_n,
    )


def value_geometric_from_base_order(
    base: Iterable[ValueRankedCandidate],
    *,
    source_cost_scale: float,
    exponent: float,
    rerank_top_n: int,
) -> list[ValueRankedCandidate]:
    """Apply the value prior to an already blended relevance order."""

    if source_cost_scale <= 0.0:
        raise ValueError("source_cost_scale must be positive")
    if exponent < 0.0:
        raise ValueError("exponent must be non-negative")
    if rerank_top_n <= 0:
        raise ValueError("rerank_top_n must be positive")
    rows = list(base)
    limit = min(rerank_top_n, len(rows))
    head = list(enumerate(rows[:limit], start=1))
    head.sort(
        key=lambda item: (
            math.log(float(item[0]))
            - exponent
            * math.log1p(max(0.0, item[1][4]) / source_cost_scale),
            item[1][1],
            item[1][2],
        )
    )
    return [value for _, value in head] + rows[limit:]
