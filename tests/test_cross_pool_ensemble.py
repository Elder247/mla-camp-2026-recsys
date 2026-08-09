from __future__ import annotations

from scripts.tune_cross_pool_ensemble import fuse_orders, metrics_for_orders


def test_fuse_orders_rewards_shared_and_weighted_candidates() -> None:
    old = [(1, 7, 10.0), (2, 7, 20.0), (3, 7, 30.0)]
    new = [(3, 7, 30.0), (4, 7, 40.0), (5, 7, 50.0)]
    fused = fuse_orders(old, new, new_weight=0.5, rrf_constant=10.0)
    assert [row[2] for row in fused[:4]] == [3, 1, 2, 4]


def test_cross_pool_metrics_use_union_ranks() -> None:
    orders = {
        "r": [
            (1.0, 1, 10, 5, 100.0),
            (0.5, 2, 20, 5, 200.0),
        ]
    }
    truth = {("r", 20): 200.0, ("r", 30): 300.0}
    result = metrics_for_orders(orders, truth)
    assert result["50"]["hits"] == 1
    assert result["50"]["sourcecost_recall"] == 0.4
