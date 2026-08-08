from __future__ import annotations

from generators.history_variants import rank


def model() -> dict:
    return {
        "version": 1,
        "rankings": {
            ("query", "кофе", 0): [
                (10, 2, 900.0, 4),
                (20, 8, 200.0, 5),
                (30, 1, 1_500.0, 3),
            ],
            ("query_region", "кофе", 213): [(40, 3, 700.0, 6)],
        },
        "candidates": {banner_id: {} for banner_id in (10, 20, 30, 40)},
    }


def test_query_click_and_sourcecost_are_independent_rankings() -> None:
    example = {"query": "Кофе", "context": {"region_id": 213}}
    by_click = rank(
        model=model(), example=example, features={"variant": "query_click"}, top_k=3
    )
    by_value = rank(
        model=model(), example=example, features={"variant": "query_sc"}, top_k=3
    )
    assert [row["banner_id"] for row in by_click] == [20, 10, 30]
    assert [row["banner_id"] for row in by_value] == [30, 10, 20]
    assert by_click[0]["score"] == 8.0
    assert by_value[0]["score"] == 1_500.0


def test_query_region_has_no_query_only_backoff() -> None:
    result = rank(
        model=model(),
        example={"query": "Кофе", "context": {"region_id": 213}},
        features={"variant": "query_region_sc"},
        top_k=10,
    )
    assert [row["banner_id"] for row in result] == [40]
    missing = rank(
        model=model(),
        example={"query": "Кофе", "context": {"region_id": 2}},
        features={"variant": "query_region_sc"},
        top_k=10,
    )
    assert missing == []
