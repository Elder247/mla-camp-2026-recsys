from __future__ import annotations

from scripts.tune_top50_ensemble import fuse_rankings, simplex_weights


def test_simplex_weights_cover_the_unit_simplex() -> None:
    values = simplex_weights(3, 0.5)
    assert len(values) == 6
    assert all(abs(sum(row) - 1.0) < 1.0e-9 for row in values)
    assert (0.0, 0.5, 0.5) in values


def test_fuse_rankings_rewards_shared_candidates() -> None:
    fused = fuse_rankings(
        [[1, 2, 3], [3, 4, 5]],
        (0.5, 0.5),
        rrf_constant=10.0,
        hit_log_id=7,
        source_costs={1: 10.0, 3: 30.0},
    )
    assert [row[2] for row in fused[:4]] == [3, 1, 2, 4]
    assert fused[0][3:] == (7, 30.0)
