from scripts.materialize_consensus_tail import consensus_tail


def test_consensus_tail_preserves_control_prefix_and_uniqueness():
    control = list(range(1, 51))
    alternate = list(range(1, 49)) + [100, 101]

    result = consensus_tail(
        control,
        alternate,
        preserve_top=10,
        alternate_weight=0.5,
        rrf_constant=30.0,
    )

    assert result[:10] == control[:10]
    assert len(result) == 50
    assert len(set(result)) == 50


def test_consensus_tail_is_identity_for_identical_rankings():
    control = list(range(1, 51))

    result = consensus_tail(
        control,
        control,
        preserve_top=10,
        alternate_weight=0.5,
        rrf_constant=30.0,
    )

    assert result == control
