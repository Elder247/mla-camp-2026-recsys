from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
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


def test_history_patch_replaces_only_history_features(tmp_path: Path) -> None:
    donor = tmp_path / "donor.parquet"
    merged = tmp_path / "merged.parquet"
    output = tmp_path / "output.parquet"
    pq.write_table(
        pa.table(
            {
                "request_id": ["r1", "r1"],
                "banner_id": pa.array([1, 2], type=pa.uint64()),
                "untouched": pa.array([7.0, 8.0], type=pa.float32()),
                "history_click_count_log1p": pa.array([0.0, 0.0], type=pa.float32()),
                "history_source_cost_log1p": pa.array([0.0, 0.0], type=pa.float32()),
                "history_query_present": pa.array([0.0, 0.0], type=pa.float32()),
                "history_region_present": pa.array([0.0, 0.0], type=pa.float32()),
            }
        ),
        donor,
    )
    pq.write_table(
        pa.table(
            {
                "request_id": ["r1", "r1"],
                "banner_id": pa.array([1, 2], type=pa.uint64()),
                "history_click_count": pa.array([4, 0], type=pa.int64()),
                "history_source_cost_sum": [500.0, 0.0],
                "history_query_present": [True, False],
                "history_region_present": [True, False],
            }
        ),
        merged,
    )

    build_features._patch_history_feature_columns(
        donor_output=donor,
        merged_path=merged,
        output=output,
    )

    actual = pq.read_table(output).to_pydict()
    assert pq.read_table(output).schema == pq.read_table(donor).schema
    assert actual["untouched"] == [7.0, 8.0]
    assert np.isclose(actual["history_click_count_log1p"][0], np.log1p(4.0))
    assert np.isclose(actual["history_source_cost_log1p"][0], np.log1p(500.0))
    assert actual["history_query_present"] == [1.0, 0.0]
    assert actual["history_region_present"] == [1.0, 0.0]
