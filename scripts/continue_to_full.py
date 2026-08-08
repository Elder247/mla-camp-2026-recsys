#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mla_recsys.artifacts import atomic_write_json, utc_now
from mla_recsys.config import compose_config


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def select_ranking(metrics: dict, candidates: list[str]) -> tuple[str, float]:
    values = [
        (name, float(metrics[name]["50"]["sourcecost_recall"]))
        for name in candidates
    ]
    # Prefer the cheaper RRF path on an exact metric tie.
    return max(values, key=lambda item: (item[1], item[0] == "rrf"))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Wait for an honest temporal gate and launch detached-safe full refit"
    )
    parser.add_argument("--temporal-run", required=True)
    parser.add_argument("--full-run", required=True)
    parser.add_argument("--experiment", default="i1_more_cg_features_sc")
    parser.add_argument("--source-runs", type=Path, required=True)
    parser.add_argument("--output-runs", type=Path, required=True)
    parser.add_argument("--immutable-artifacts", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")

    cfg = compose_config(
        args.experiment,
        run_id=args.full_run,
        mode="full",
        scope="full",
    )
    temporal_path = args.source_runs / args.temporal_run
    decision_path = Path(f"/tmp/{args.full_run}.promotion.json")
    decision = {
        "version": 1,
        "temporal_run": args.temporal_run,
        "full_run": args.full_run,
        "status": "waiting",
        "started_at": utc_now(),
    }
    atomic_write_json(decision_path, decision)

    while True:
        result_path = temporal_path / "result.json"
        if result_path.is_file():
            result = read_json(result_path)
            status = str(result.get("status"))
            if status == "completed":
                break
            if status == "failed":
                decision.update(
                    status="temporal_failed",
                    finished_at=utc_now(),
                    temporal_error=result.get("error"),
                )
                atomic_write_json(decision_path, decision)
                return 2
        time.sleep(args.poll_seconds)

    candidates = read_json(temporal_path / "metrics" / "candidates.json")
    ranker = read_json(temporal_path / "metrics" / "ranker.json")
    candidate_value = float(
        candidates["metrics"]["merged"]["500"]["sourcecost_recall"]
    )
    ranking_candidates = [
        str(value)
        for value in cfg.promotion_gate.get("ranking_candidates", ["catboost"])
    ]
    selected_ranking, ranker_value = select_ranking(
        ranker["metrics"], ranking_candidates
    )
    candidate_threshold = float(
        cfg.promotion_gate.candidate_sourcecost_recall_at_500
    )
    ranker_threshold = float(cfg.promotion_gate.ranker_sourcecost_recall_at_50)
    passed = candidate_value > candidate_threshold and ranker_value > ranker_threshold
    model_metadata = read_json(temporal_path / "models" / "catboost.json")
    best_iteration = int(model_metadata.get("best_iteration", -1))
    full_iterations = best_iteration + 1 if best_iteration >= 0 else int(cfg.ranker.iterations)
    decision.update(
        status="gate_passed" if passed else "gate_rejected",
        evaluated_at=utc_now(),
        candidate_sourcecost_recall_at_500=candidate_value,
        candidate_threshold=candidate_threshold,
        ranker_sourcecost_recall_at_50=ranker_value,
        ranker_threshold=ranker_threshold,
        selected_ranking=selected_ranking,
        ranking_candidates=ranking_candidates,
        temporal_best_iteration=best_iteration,
        full_iterations=full_iterations,
    )
    atomic_write_json(decision_path, decision)
    if not passed:
        return 3

    command = [
        str(cfg.paths.python),
        str(ROOT / "scripts" / "run_pipeline.py"),
        f"experiment={args.experiment}",
        f"run_id={args.full_run}",
        "mode=full",
        "scope=full",
        f"paths.root={ROOT}",
        f"paths.runs={args.output_runs}",
        f"paths.cache={args.output_runs.parent / 'cache'}",
        f"paths.immutable_artifacts={args.immutable_artifacts}",
        f"ranker.iterations={full_iterations}",
        f"submission.ranking={selected_ranking}",
    ]
    decision.update(status="full_running", full_started_at=utc_now(), command=command)
    atomic_write_json(decision_path, decision)
    return_code = subprocess.run(command, cwd=ROOT, check=False).returncode
    decision.update(
        status="completed" if return_code == 0 else "full_failed",
        full_return_code=return_code,
        finished_at=utc_now(),
    )
    atomic_write_json(decision_path, decision)
    full_metrics = args.output_runs / args.full_run / "metrics"
    if full_metrics.is_dir():
        atomic_write_json(full_metrics / "promotion.json", decision)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
