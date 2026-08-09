from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from generators.two_tower_v2_batch import load_logq_restore_bias  # noqa: E402
from scripts.tune_two_tower_logq_geometry import logq_rerank  # noqa: E402
from two_tower_v2.training import file_sha256  # noqa: E402


def test_logq_rerank_is_bounded_and_keeps_zero_control_exact() -> None:
    base = [
        (0.90, 1, 10, 1, 0.0),
        (0.89, 2, 20, 1, 0.0),
        (0.88, 3, 30, 1, 0.0),
    ]
    assert logq_rerank(base, counts={20: 100}, alpha=0.0, top_n=2) == base
    reranked = logq_rerank(base, counts={10: 1, 20: 100}, alpha=0.1, top_n=2)
    assert [value[2] for value in reranked] == [20, 10, 30]
    assert reranked[2] == base[2]


def test_inference_bias_validates_prior_and_aligns_candidate_ids(tmp_path) -> None:
    prior_dir = tmp_path / "prior"
    artifact_dir = tmp_path / "artifact"
    prior_dir.mkdir()
    artifact_dir.mkdir()
    prior_file = prior_dir / "banner_frequency.parquet"
    pq.write_table(
        pa.table({"banner_id": [10, 20], "count": [4, 100]}), prior_file
    )
    manifest = {
        "version": 1,
        "kind": "global_banner_frequency",
        "source": {"table": "//train", "row_count": 104},
        "file": {
            "name": prior_file.name,
            "sha256": file_sha256(prior_file),
        },
    }
    manifest_path = prior_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (artifact_dir / "inference_config.json").write_text(
        json.dumps(
            {
                "version": 1,
                "kind": "global_banner_logq_restore",
                "alpha": 0.05,
                "unseen_count": 1.0,
                "prior_dir": str(prior_dir),
                "prior_manifest_sha256": file_sha256(manifest_path),
            }
        ),
        encoding="utf-8",
    )
    bias, info = load_logq_restore_bias(
        artifact_dir,
        candidate_metadata={"banner_id": [20, 30]},
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert bias is not None
    assert bias.tolist() == pytest.approx([0.05 * math.log(100.0), 0.0])
    assert info["candidate_coverage"] == pytest.approx(0.5)
