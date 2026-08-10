from __future__ import annotations

from scripts.materialize_lexical_tail import lexical_z, routed_banners


def test_flat_lexical_tail_replaces_only_rank_50() -> None:
    control = list(range(1, 51))
    lexical = [(100, 1.0), (101, 1.0), (1, 0.9)]
    routed, used, z_value = routed_banners(
        control, lexical, z_threshold=1.0, z_depth=3
    )
    assert used
    assert z_value <= 1.0
    assert routed[:49] == control[:49]
    assert routed[49] == 100


def test_confident_lexical_ranking_preserves_control() -> None:
    control = list(range(1, 51))
    lexical = [(100, 100.0), (101, 1.0), (102, 0.5)]
    routed, used, _ = routed_banners(
        control, lexical, z_threshold=1.0, z_depth=3
    )
    assert not used
    assert routed == control


def test_lexical_z_requires_two_scores() -> None:
    assert lexical_z([(1, 1.0)], depth=10) == float("inf")
