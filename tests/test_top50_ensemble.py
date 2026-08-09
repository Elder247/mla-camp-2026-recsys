from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.tune_top50_ensemble import (
    conditional_weights,
    fuse_rankings,
    read_ranking,
    simplex_weights,
)


def test_simplex_weights_cover_the_unit_simplex() -> None:
    values = simplex_weights(3, 0.5)
    assert len(values) == 6
    assert all(abs(sum(row) - 1.0) < 1.0e-9 for row in values)
    assert (0.0, 0.5, 0.5) in values


def test_simplex_weights_can_preserve_a_strong_control_source() -> None:
    values = simplex_weights(3, 0.05, minimum_first_weight=0.8)
    assert len(values) == 15
    assert all(row[0] >= 0.8 for row in values)
    assert (0.8, 0.1, 0.1) in values
    assert (1.0, 0.0, 0.0) in values


def test_conditional_weights_fall_back_to_first_input_only() -> None:
    weights = (0.8, 0.1, 0.1)
    assert conditional_weights(weights, use_secondary_sources=True) == weights
    assert conditional_weights(weights, use_secondary_sources=False) == (
        1.0,
        0.0,
        0.0,
    )


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


def test_read_ranking_can_prune_candidate_partitions(tmp_path) -> None:
    pq.write_table(
        pa.table(
            {
                "hit_log_id": [1, 1, 1, 2, 2],
                "banner_id": [11, 12, 13, 21, 22],
                "source_rank": [1, 2, 3, 1, 2],
            }
        ),
        tmp_path / "part-00000.parquet",
    )

    assert read_ranking(tmp_path, candidate_top_k=2) == {
        1: [11, 12],
        2: [21, 22],
    }
