from __future__ import annotations

from mla_recsys.counters import CounterLookup, counter_feature_values


def event(timestamp: int, banner_id: int, cost: float) -> dict:
    return {
        "show_time": timestamp,
        "banner_id": banner_id,
        "group_id": 7,
        "domain": "example.test",
        "query_key": "q",
        "region_key": "213",
        "user_key": "42",
        "source_cost": cost,
    }


def test_strict_asof_excludes_same_timestamp_targets() -> None:
    lookup = CounterLookup([event(100, 1, 10.0), event(100, 1, 20.0)])
    assert lookup.stats("banner", "1", row_timestamp=100, window_days=0).clicks == 0
    stats = lookup.stats("banner", "1", row_timestamp=101, window_days=0)
    assert stats.clicks == 2
    assert stats.source_cost_sum == 30.0


def test_frozen_cutoff_is_used_for_timestamp_less_inference() -> None:
    lookup = CounterLookup([event(100, 1, 10.0), event(200, 1, 20.0)])
    stats = lookup.stats(
        "banner", "1", row_timestamp=None, window_days=0, frozen_cutoff=150
    )
    assert stats.clicks == 1
    assert stats.source_cost_sum == 10.0


def test_counter_feature_contract_has_configured_windows() -> None:
    lookup = CounterLookup([event(100, 1, 10.0)])
    values = counter_feature_values(
        lookup,
        {
            "query": "q",
            "region_id": 213,
            "crypta_id_v2": 42,
            "show_time": 101,
        },
        {"banner_id": 1, "group_id": 7, "domain": "example.test"},
        families=["banner"],
        windows_days=[7, 0],
        frozen_cutoff=999,
    )
    assert values["counter__banner__7d__present"] == 1.0
    assert values["counter__banner__all__sc_avg"] == 10.0
