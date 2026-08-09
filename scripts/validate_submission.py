#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mla_recsys.artifacts import atomic_write_json  # noqa: E402
from mla_recsys.command import load_stage_context  # noqa: E402
from mla_recsys.data import read_request_parquet  # noqa: E402
from mla_recsys.submission import validate_submission  # noqa: E402


def main() -> int:
    context = load_stage_context(
        "Strictly validate the run test prediction",
        extra_keys=("prediction", "valid_banner_index"),
    )
    prediction = Path(
        context.values.get(
            "prediction",
            str(context.store.path / "predictions" / "test_top50.parquet"),
        )
    )
    requests = read_request_parquet(context.store.path / "data" / "test_requests.parquet")
    expected = {int(row["hit_log_id"]) for row in requests}
    valid_banner_index = Path(
        context.values.get("valid_banner_index", str(context.cfg.paths.banner_index))
    )
    index = pq.read_table(valid_banner_index, columns=["BannerID"])
    valid = {int(value) for value in index["BannerID"].to_pylist()}
    report = validate_submission(
        prediction,
        expected_hitlog_ids=expected,
        valid_banner_ids=valid,
        top_k=int(context.cfg.evaluation.submission_top_k),
        allow_short=bool(context.cfg.submission.allow_fewer_than_top_k),
    )
    report["valid_banner_index"] = str(valid_banner_index)
    atomic_write_json(context.store.path / "metrics" / "submission_validation.json", report)
    context.store.update_result(submission_validation=report)
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
