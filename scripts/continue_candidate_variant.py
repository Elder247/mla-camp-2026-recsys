#!/usr/bin/env python3
"""Tune a completed candidate variant and conditionally run a detached-safe full."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mla_recsys.artifacts import atomic_write_json, utc_now  # noqa: E402


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sc50(result: dict) -> float:
    return float(result["metrics"]["50"]["sourcecost_recall"])


def select_variant(single: dict, ensemble: dict) -> dict:
    single_best = dict(single["best"])
    single_choice = {
        "ranking": "value_geometry",
        "score": sc50(single_best),
        "catboost_weight": float(single_best["catboost_weight"]),
        "exponent": float(single_best["exponent"]),
        "rerank_top_n": int(single_best["rerank_top_n"]),
    }
    geometry = ensemble.get("geometry")
    if not geometry:
        return single_choice
    base = dict(geometry["base"])
    best = dict(geometry["best"])
    ensemble_choice = {
        "ranking": "model_ensemble",
        "score": sc50(best),
        "qrmse_weight": float(base["model_a_weight"]),
        "external_yeti_weight": 1.0 - float(base["model_a_weight"]),
        "catboost_weight": float(base["catboost_weight"]),
        "exponent": float(best["exponent"]),
        "rerank_top_n": int(best["rerank_top_n"]),
    }
    return max(
        (single_choice, ensemble_choice),
        key=lambda item: (float(item["score"]), item["ranking"] == "model_ensemble"),
    )


def run_logged(command: list[str], log: Path) -> dict:
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with log.open("w", encoding="utf-8") as stream:
        return_code = subprocess.run(
            command,
            cwd=ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        ).returncode
    return {
        "command": command,
        "return_code": return_code,
        "status": "completed" if return_code == 0 else "failed",
        "wall_seconds": time.monotonic() - started,
        "log": str(log),
    }


def full_command(args: argparse.Namespace, *, iterations: int, choice: dict) -> list[str]:
    command = [
        str(args.python),
        str(ROOT / "scripts" / "run_pipeline.py"),
        f"experiment={args.experiment}",
        f"run_id={args.full_run}",
        "mode=full",
        "scope=full",
        f"paths.root={ROOT}",
        f"paths.runs={args.runs}",
        f"paths.cache={args.cache}",
        f"paths.immutable_artifacts={args.immutable_artifacts}",
        f"paths.two_tower_v2_walk_forward_artifact={args.artifact_override}",
        f"candidates.reuse_run={args.full_reuse_run}",
        "ranker.version=catboost_queryrmse_chrono_final_full_v1",
        "ranker.loss_function=QueryRMSE",
        f"ranker.iterations={iterations}",
        "ranker.early_stopping_rounds=75",
        f"submission.ranking={choice['ranking']}",
        f"submission.blend.catboost_weight={choice['catboost_weight']}",
        f"submission.value_geometry.exponent={choice['exponent']}",
        f"submission.value_geometry.rerank_top_n={choice['rerank_top_n']}",
    ]
    if choice["ranking"] == "model_ensemble":
        command.extend(
            [
                "submission.model_ensemble.method=rank_linear",
                f"submission.model_ensemble.model_a_path={args.full_external_yeti_model}",
                f"submission.model_ensemble.model_a_weight={choice['external_yeti_weight']}",
                f"submission.model_ensemble.catboost_weight={choice['catboost_weight']}",
            ]
        )
    return command


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Wait for temporal metrics, tune bounded ranks, gate and run full"
    )
    parser.add_argument("--temporal-run", required=True)
    parser.add_argument("--full-run", required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--immutable-artifacts", type=Path, required=True)
    parser.add_argument("--artifact-override", type=Path, required=True)
    parser.add_argument("--full-reuse-run", type=Path, required=True)
    parser.add_argument("--temporal-yeti-run", type=Path, required=True)
    parser.add_argument("--full-external-yeti-model", type=Path, required=True)
    parser.add_argument("--accepted-sc50", type=float, required=True)
    parser.add_argument("--candidate-sc500-floor", type=float, default=0.70)
    parser.add_argument("--poll-seconds", type=int, default=20)
    args = parser.parse_args()

    temporal = args.runs / args.temporal_run
    full = args.runs / args.full_run
    decision_path = Path(f"/tmp/{args.full_run}.variant.json")
    decision = {
        "version": 1,
        "status": "waiting_for_temporal",
        "started_at": utc_now(),
        "temporal_run": args.temporal_run,
        "full_run": args.full_run,
        "commands": [],
    }
    atomic_write_json(decision_path, decision)
    while True:
        result_path = temporal / "result.json"
        if result_path.is_file():
            result = read_json(result_path)
            if result.get("status") == "completed":
                break
            if result.get("status") == "failed":
                decision.update(status="temporal_failed", finished_at=utc_now())
                atomic_write_json(decision_path, decision)
                return 2
        time.sleep(max(1, args.poll_seconds))

    temporary_terminal = Path(f"/tmp/{args.temporal_run}.terminal.log")
    if temporary_terminal.is_file():
        target = temporal / "logs" / "terminal.log"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(temporary_terminal, target)

    single_output = temporal / "metrics" / "chrono_value_geometry.json"
    single_command = [
        str(args.python),
        str(ROOT / "scripts" / "tune_value_geometry.py"),
        "--run",
        str(temporal),
        "--catboost-weights",
        "0.45,0.5,0.55,0.6,0.65,0.7",
        "--exponents",
        "0,0.05,0.1,0.15,0.2",
        "--rerank-top-n",
        "50,75,100,150",
        "--output",
        str(single_output),
    ]
    ensemble_output = temporal / "metrics" / "chrono_model_ensemble.json"
    ensemble_command = [
        str(args.python),
        str(ROOT / "scripts" / "tune_model_ensemble.py"),
        "--run",
        str(temporal),
        "--model-a-run",
        str(temporal),
        "--model-b-run",
        str(args.temporal_yeti_run),
        "--model-a-weights",
        "0.35,0.5,0.65",
        "--catboost-weights",
        "0.45,0.5,0.55,0.6,0.65",
        "--geometry-exponents",
        "0,0.05,0.1,0.15,0.2",
        "--geometry-top-n",
        "50,75,100,150",
        "--output",
        str(ensemble_output),
    ]
    for name, command, output in (
        ("single_geometry", single_command, single_output),
        ("model_ensemble", ensemble_command, ensemble_output),
    ):
        decision.update(status=f"{name}_running")
        atomic_write_json(decision_path, decision)
        record = run_logged(command, temporal / "logs" / f"{name}.log")
        decision["commands"].append(record)
        atomic_write_json(decision_path, decision)
        if record["return_code"] != 0 or not output.is_file():
            decision.update(status=f"{name}_failed", finished_at=utc_now())
            atomic_write_json(decision_path, decision)
            return int(record["return_code"] or 4)

    choice = select_variant(read_json(single_output), read_json(ensemble_output))
    candidates = read_json(temporal / "metrics" / "candidates.json")
    candidate_sc500 = float(
        candidates["metrics"]["merged"]["500"]["sourcecost_recall"]
    )
    passed = (
        float(choice["score"]) > args.accepted_sc50
        and candidate_sc500 >= args.candidate_sc500_floor
    )
    metadata = read_json(temporal / "models" / "catboost.json")
    best_iteration = int(metadata.get("best_iteration", -1))
    iterations = best_iteration + 1 if best_iteration >= 0 else 185
    decision.update(
        status="gate_passed" if passed else "gate_rejected",
        evaluated_at=utc_now(),
        selected=choice,
        candidate_sc500=candidate_sc500,
        candidate_sc500_floor=args.candidate_sc500_floor,
        accepted_sc50=args.accepted_sc50,
        temporal_best_iteration=best_iteration,
        full_iterations=iterations,
    )
    atomic_write_json(decision_path, decision)
    if not passed:
        decision["finished_at"] = utc_now()
        atomic_write_json(decision_path, decision)
        return 3

    command = full_command(args, iterations=iterations, choice=choice)
    decision.update(status="full_running", full_started_at=utc_now(), full_command=command)
    atomic_write_json(decision_path, decision)
    record = run_logged(command, Path(f"/tmp/{args.full_run}.terminal.log"))
    decision["commands"].append(record)
    full_result = read_json(full / "result.json") if (full / "result.json").is_file() else {}
    decision.update(
        status="completed" if record["return_code"] == 0 else "full_failed",
        full_result_status=full_result.get("status"),
        finished_at=utc_now(),
    )
    atomic_write_json(decision_path, decision)
    if full.is_dir():
        logs = full / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        shutil.copy2(Path(record["log"]), logs / "terminal.log")
        metrics = full / "metrics"
        metrics.mkdir(parents=True, exist_ok=True)
        atomic_write_json(metrics / "variant_promotion.json", decision)
    return int(record["return_code"])


if __name__ == "__main__":
    raise SystemExit(main())
