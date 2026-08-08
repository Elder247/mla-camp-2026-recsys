from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf
import pyarrow as pa
import pyarrow.parquet as pq

from mla_recsys.counters import COUNTER_EVENT_SCHEMA, stable_text_key
from mla_recsys.data import write_request_parquet
from mla_recsys.temporal_candidates import temporal_rankings


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
