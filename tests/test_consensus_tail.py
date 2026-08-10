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


def test_consensus_tail_can_promote_a_high_value_alternate_tail():
    control = list(range(1, 51))
    alternate = list(range(1, 50)) + [100]
    costs = {banner: 1.0 for banner in control}
    costs[100] = 100_000_000.0

    result = consensus_tail(
        control,
        alternate,
        costs,
        preserve_top=40,
        alternate_weight=0.5,
        rrf_constant=30.0,
        source_cost_exponent=0.25,
    )

    assert result[:40] == control[:40]
    assert 100 in result
    assert len(result) == len(set(result)) == 50
