from __future__ import annotations

from collections.abc import Iterable


RankedCandidate = tuple[float, int, int, int]


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
