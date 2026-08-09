from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

from scripts.validate_cache_parity import temporal_rankings_by_source


def request(request_id: str, timestamp: int, banner_id: int) -> dict:
    return {
        "request_id": request_id,
        "hit_log_id": int(request_id),
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


def test_parallel_temporal_parity_matches_serial(tmp_path: Path) -> None:
    cfg = OmegaConf.create(
        {
            "candidates": {
                "generators": {
                    "history_user_v1": {
                        "kind": "temporal_history",
                        "top_k": 10,
                        "min_clicks": 1,
                    },
                    "global_pop_sc_v1": {
                        "kind": "temporal_history",
                        "top_k": 10,
                        "min_clicks": 1,
                        "bayes_prior": 0.0,
                    },
                }
            }
        }
    )
    rows = [request("1", 100, 10), request("2", 101, 20)]
    sources = ["history_user_v1", "global_pop_sc_v1"]

    serial = temporal_rankings_by_source(
        cfg=cfg,
        run_path=tmp_path,
        split="train",
        sources=sources,
        requests=rows,
        workers=1,
    )
    parallel = temporal_rankings_by_source(
        cfg=cfg,
        run_path=tmp_path,
        split="train",
        sources=sources,
        requests=rows,
        workers=2,
    )

    assert parallel == serial
