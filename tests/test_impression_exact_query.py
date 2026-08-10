from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_impression_exact_query import add_prior, blend_order, order  # noqa: E402


def values(*, shows: float, clicks: float, value: float, last: float) -> dict[str, float]:
    return {
        "shows": shows,
        "clicks": clicks,
        "value": value,
        "last": last,
        "shows7": shows,
        "clicks7": clicks,
        "shows42": shows,
        "clicks42": clicks,
    }


def test_blend_order_moves_a_model_supported_candidate() -> None:
    exact = [10, 20, 30]
    assert blend_order(exact, {30: 1, 20: 2, 10: 3}, 0.0) == exact
    assert blend_order(exact, {30: 1, 20: 2, 10: 3}, 1.0) == [30, 20, 10]


def test_impression_orders_are_deterministic() -> None:
    stats = {
        10: values(shows=100, clicks=1, value=1, last=5),
        20: values(shows=10, clicks=5, value=10, last=4),
        30: values(shows=10, clicks=5, value=20, last=6),
    }
    assert order(stats, "shows") == [10, 30, 20]
    assert order(stats, "clicks") == [30, 20, 10]
    assert order(stats, "value") == [30, 20, 10]
    assert order(stats, "recency") == [30, 10, 20]


def test_add_prior_adds_only_observed_clicks() -> None:
    stats: dict = {}
    add_prior(
        stats,
        [
            {
                "query": "literal query",
                "show_time": 123,
                "clicked_banner_ids": [7],
                "clicked_source_costs": [11.5],
            }
        ],
    )
    candidate = stats["literal query"][7]
    assert candidate["shows"] == 1.0
    assert candidate["clicks"] == 1.0
    assert candidate["value"] == 11.5
    assert candidate["last"] == 123.0
