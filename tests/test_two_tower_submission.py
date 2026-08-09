from __future__ import annotations

from scripts.make_two_tower_submission import rerank_rows


def test_direct_two_tower_geometry_is_bounded_to_prefix() -> None:
    values = [
        (0.9, 1, 1, 10, 0.0),
        (0.8, 2, 2, 10, 100_000_000.0),
        (0.7, 3, 3, 10, 0.0),
    ]
    ordered = rerank_rows(
        values,
        source_cost_scale=1_000_000.0,
        exponent=0.5,
        rerank_top_n=2,
    )
    assert [value[2] for value in ordered] == [2, 1, 3]
