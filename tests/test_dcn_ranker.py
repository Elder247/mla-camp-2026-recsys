from __future__ import annotations

import numpy as np
import torch

from mla_recsys.dcn_ranker import DCNv2Ranker, sample_listwise_groups, stable_hash


def test_stable_hash_is_deterministic_and_bounded() -> None:
    assert stable_hash("query", 1024) == stable_hash("query", 1024)
    assert 0 <= stable_hash("query", 1024) < 1024
    assert stable_hash("query", 1024) != stable_hash("other", 1024)


def test_sample_listwise_groups_keeps_positives_and_normalizes() -> None:
    groups = np.asarray([1, 1, 1, 1, 2, 2, 2, 2], dtype=np.uint64)
    ranks = np.asarray([1, 2, 3, 4, 1, 2, 3, 4], dtype=np.int32)
    costs = np.asarray([0, 9, 0, 0, 0, 0, 16, 0], dtype=np.float64)
    first = sample_listwise_groups(
        groups,
        ranks,
        costs,
        candidates_per_group=3,
        hard_fraction=0.5,
        seed=7,
    )
    second = sample_listwise_groups(
        groups,
        ranks,
        costs,
        candidates_per_group=3,
        hard_fraction=0.5,
        seed=7,
    )
    np.testing.assert_array_equal(first.indices, second.indices)
    np.testing.assert_allclose(first.targets.sum(axis=1), 1.0)
    assert costs[first.indices[0]].max() == 9
    assert costs[first.indices[1]].max() == 16
    assert np.isclose(first.weights.mean(), 1.0)


def test_dcn_ranker_starts_as_base_ranking() -> None:
    model = DCNv2Ranker(
        4,
        [32, 16],
        [3, 2],
        cross_layers=2,
        deep_dims=(8, 4),
        dropout=0.0,
    )
    continuous = torch.randn(5, 4)
    categorical = torch.zeros(5, 2, dtype=torch.long)
    base = torch.linspace(-1.0, 0.0, 5)
    torch.testing.assert_close(model(continuous, categorical, base), base)
