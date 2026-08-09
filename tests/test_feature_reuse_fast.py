from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf

import scripts.build_features as build_features


def test_feature_reuse_runs_before_heavy_worker_initialization(
    tmp_path: Path,
    monkeypatch,
) -> None:
    cfg = OmegaConf.create(
        {
            "features": {"version": "feature_v1", "reuse_run": "/donor"},
            "runtime": {"scope": "offline"},
        }
    )
    calls = []

    def fake_reuse(**kwargs):
        calls.append(kwargs["partition"])
        assert "banner_index" not in build_features._FEATURE_STATE
        assert "counter_lookup" not in build_features._FEATURE_STATE
        return {
            "rows": 10,
            "groups": 1,
            "positive_groups": 1,
            "missed_positive_groups": 0,
        }

    monkeypatch.setattr(build_features, "_try_reuse_feature_partition", fake_reuse)
    results = build_features._preinitialize_feature_reuse(
        cfg=cfg,
        run_path=tmp_path,
        split="train",
        request_path=tmp_path / "requests.parquet",
        banner_index_path=tmp_path / "banners.parquet",
        partitions=3,
        config_sha="sha",
        force=False,
    )

    assert calls == [0, 1, 2]
    assert results is not None
    assert [stats["reused"] for _, stats in results] == [1, 1, 1]
