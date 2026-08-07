#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mla_recsys.artifacts import atomic_output_path, atomic_write_json  # noqa: E402
from mla_recsys.command import load_stage_context  # noqa: E402


def main() -> int:
    context = load_stage_context("Evaluate candidates and CatBoost on temporal holdout")
    if str(context.cfg.runtime.scope) != "offline":
        raise ValueError("evaluate_run is only valid for offline scope")
    from mla_recsys.evaluation import (  # noqa: PLC0415
        candidate_report,
        ranker_report,
        write_complementarity_csv,
    )

    candidates, complementarity = candidate_report(
        cfg=context.cfg,
        run_path=context.store.path,
        split="holdout",
    )
    ranker, predictions = ranker_report(
        cfg=context.cfg,
        run_path=context.store.path,
        split="holdout",
    )
    atomic_write_json(context.store.path / "metrics" / "candidates.json", candidates)
    atomic_write_json(context.store.path / "metrics" / "ranker.json", ranker)
    atomic_write_json(context.store.path / "metrics" / "holdout.json", ranker)
    write_complementarity_csv(
        context.store.path / "reports" / "source_complementarity.csv",
        complementarity,
    )
    prediction_path = context.store.path / "predictions" / "holdout_top50.parquet"
    with atomic_output_path(prediction_path) as temporary:
        pq.write_table(predictions, temporary, compression="zstd")
    primary = ranker["metrics"]["catboost"]["50"]
    context.store.update_result(
        candidate_metrics=candidates["metrics"]["merged"],
        ranker_metrics=ranker["metrics"]["catboost"],
        primary_metric={"name": "sourcecost_recall@50", "value": primary["sourcecost_recall"]},
    )
    print(json.dumps({"candidates": candidates, "ranker": ranker}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

