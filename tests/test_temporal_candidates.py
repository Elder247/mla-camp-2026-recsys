from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf
import pyarrow as pa
import pyarrow.parquet as pq

from mla_recsys.counters import COUNTER_EVENT_SCHEMA, stable_text_key
from mla_recsys.data import write_request_parquet
from mla_recsys.temporal_candidates import TemporalHistoryState, temporal_rankings


def cfg():
    return OmegaConf.create(
        {
            "candidates": {
                "generators": {
                    "history_user_v1": {
                        "kind": "temporal_history",
                        "top_k": 10,
                        "quota": 10,
                        "min_clicks": 1,
                        "bayes_prior": 0.0,
                    }
                }
            }
        }
    )


def request(request_id: str, timestamp: int, banner_id: int) -> dict:
    return {
        "request_id": request_id,
        "hit_log_id": int(request_id.rsplit(":", 1)[-1]),
        "show_time": timestamp,
        "query": "q",
        "region_id": 1,
        "crypta_id_v2": 42,
        "device": None,
        "age": None,
        "gender": None,
        "clicked_banner_ids": [banner_id],
        "clicked_source_costs": [float(banner_id)],
    }


def test_equal_timestamp_targets_do_not_see_each_other(tmp_path: Path) -> None:
    rows = [request("1", 100, 10), request("2", 100, 20), request("3", 101, 30)]
    result = temporal_rankings(
        cfg=cfg(),
        run_path=tmp_path,
        split="train",
        source="history_user_v1",
        requests=rows,
    )
    assert result["1"] == []
    assert result["2"] == []
    assert {row["banner_id"] for row in result["3"]} == {10, 20}


def test_holdout_uses_frozen_train_state(tmp_path: Path) -> None:
    train = [request("1", 100, 10)]
    holdout = [request("2", 200, 20), request("3", 201, 30)]
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    write_request_parquet(data_dir / "train_requests.parquet", train)
    result = temporal_rankings(
        cfg=cfg(),
        run_path=tmp_path,
        split="holdout",
        source="history_user_v1",
        requests=holdout,
    )
    assert [row["banner_id"] for row in result["2"]] == [10]
    assert [row["banner_id"] for row in result["3"]] == [10]


def test_external_history_is_strictly_prior_for_oof_rows(tmp_path: Path) -> None:
    events = tmp_path / "history.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "show_time": 100,
                    "banner_id": 10,
                    "group_id": None,
                    "domain": "",
                    "query_key": stable_text_key("q"),
                    "region_key": "1",
                    "user_key": "42",
                    "source_cost": 100.0,
                }
            ],
            schema=COUNTER_EVENT_SCHEMA,
        ),
        events,
    )
    config = OmegaConf.create(
        {
            "paths": {"history_events": str(events)},
            "candidates": {
                "generators": {
                    "history_query_sc_oof_v1": {
                        "kind": "temporal_history",
                        "top_k": 10,
                        "quota": 10,
                        "min_clicks": 1,
                        "bayes_prior": 0.0,
                        "external_events_path_key": "history_events",
                    }
                }
            },
        }
    )
    rows = [request("oof:1", 100, 10), request("oof:2", 101, 20)]
    result = temporal_rankings(
        cfg=config,
        run_path=tmp_path,
        split="train",
        source="history_query_sc_oof_v1",
        requests=rows,
    )
    assert result["oof:1"] == []
    assert [row["banner_id"] for row in result["oof:2"]] == [10]


def test_history_state_rejects_banners_outside_frozen_index() -> None:
    state = TemporalHistoryState(
        "global_pop_sc_v1",
        min_clicks=1,
        bayes_prior=0.0,
        valid_banner_ids={10},
    )
    state.observe(request("1", 100, 10))
    state.observe(request("2", 101, 20))
    ranked = state.rank(request("3", 102, 30), top_k=10)
    assert [row["banner_id"] for row in ranked] == [10]


def test_temporal_rankings_use_configured_frozen_index(tmp_path: Path) -> None:
    banner_index = tmp_path / "banner_index.parquet"
    pq.write_table(pa.table({"BannerID": pa.array([10], type=pa.int64())}), banner_index)
    config = cfg()
    config.paths = {"banner_index": str(banner_index)}
    rows = [request("1", 100, 10), request("2", 101, 20), request("3", 102, 30)]
    result = temporal_rankings(
        cfg=config,
        run_path=tmp_path,
        split="train",
        source="history_user_v1",
        requests=rows,
    )
    assert result["1"] == []
    assert [row["banner_id"] for row in result["2"]] == [10]
    assert [row["banner_id"] for row in result["3"]] == [10]


def brute_force_ids(state: TemporalHistoryState, row: dict, top_k: int) -> list[int]:
    key = state._key(row)
    values = []
    for banner_id, item in state.stats.get(str(key), {}).items():
        if item.clicks >= state.min_clicks:
            values.append(
                (state._score(item), item.clicks, item.last_show_time, banner_id)
            )
    values.sort(key=lambda value: (-value[0], -value[1], -value[2], value[3]))
    return [value[3] for value in values[:top_k]]


def test_incremental_temporal_topk_matches_full_sort() -> None:
    state = TemporalHistoryState(
        "global_pop_sc_v1",
        min_clicks=1,
        bayes_prior=20.0,
    )
    probe = request("999", 999, 999)
    observations = [
        request("1", 100, 10),
        request("2", 101, 20),
        request("3", 102, 30),
        request("4", 103, 20),
        request("5", 104, 40),
        request("6", 105, 40),
        request("7", 106, 40),
        request("8", 107, 10),
    ]
    for row in observations:
        state.observe(row)
        expected = brute_force_ids(state, probe, top_k=2)
        actual = [item["banner_id"] for item in state.rank(probe, top_k=2)]
        assert actual == expected

    # Expanding K must fall back to the complete state rather than only the
    # previously cached top two.
    assert [item["banner_id"] for item in state.rank(probe, top_k=4)] == (
        brute_force_ids(state, probe, top_k=4)
    )
