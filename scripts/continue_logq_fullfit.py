#!/usr/bin/env python3
"""Promote an honest logQ ensemble to a minimal fullfit/test candidate pool."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mla_recsys.artifacts import atomic_write_json, utc_now  # noqa: E402


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ensemble-report", type=Path, required=True)
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--finetune-config", type=Path, required=True)
    parser.add_argument("--finetune-artifact", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--immutable-artifacts", type=Path, required=True)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=10)
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    parser.add_argument("--minimum-gain", type=float, default=0.0001)
    parser.add_argument("--quota", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1024)
    return parser.parse_args()


def sc50(row: dict, split: str) -> float:
    return float(row[split]["50"]["sourcecost_recall"])


def robust_best(report: dict) -> dict:
    rows = list(report.get("refined_results") or ())
    if not rows:
        raise ValueError("Ensemble report has no refined_results")
    return max(
        rows,
        key=lambda row: (
            min(sc50(row, "tune_metrics"), sc50(row, "validation_metrics")),
            sc50(row, "full_metrics"),
        ),
    )


def gate(ensemble: dict, baseline: dict, minimum_gain: float) -> dict:
    candidate = robust_best(ensemble)
    control = robust_best(baseline)
    gains = {
        "early": sc50(candidate, "tune_metrics")
        - sc50(control, "tune_metrics"),
        "late": sc50(candidate, "validation_metrics")
        - sc50(control, "validation_metrics"),
        "full": sc50(candidate, "full_metrics")
        - sc50(control, "full_metrics"),
    }
    return {
        "accepted": all(value >= minimum_gain for value in gains.values()),
        "minimum_gain": float(minimum_gain),
        "gains": gains,
        "candidate": candidate,
        "control": control,
    }


def run_logged(command: list[str], path: Path) -> float:
    path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with path.open("a", encoding="utf-8") as target:
        result = subprocess.run(
            command,
            cwd=ROOT,
            stdout=target,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with {result.returncode}: {command}")
    return time.monotonic() - started


def completed_artifact(path: Path) -> bool:
    return (path / "manifest.json").is_file()


def completed_run(path: Path) -> bool:
    result = path / "result.json"
    return result.is_file() and json.loads(result.read_text(encoding="utf-8")).get(
        "status"
    ) == "completed"


def main() -> int:
    args = arguments()
    if args.poll_seconds <= 0 or args.timeout_seconds <= 0:
        raise ValueError("poll and timeout must be positive")
    state = {
        "version": 1,
        "status": "waiting_for_gate",
        "started_at": utc_now(),
        "ensemble_report": str(args.ensemble_report),
        "baseline_report": str(args.baseline_report),
    }
    atomic_write_json(args.decision, state)
    started = time.monotonic()
    try:
        while not (args.ensemble_report.is_file() and args.baseline_report.is_file()):
            if time.monotonic() - started >= args.timeout_seconds:
                raise TimeoutError("Timed out waiting for logQ ensemble reports")
            time.sleep(args.poll_seconds)
        result = gate(
            json.loads(args.ensemble_report.read_text(encoding="utf-8")),
            json.loads(args.baseline_report.read_text(encoding="utf-8")),
            args.minimum_gain,
        )
        state.update(gate=result)
        if not result["accepted"]:
            state.update(status="rejected", finished_at=utc_now())
            atomic_write_json(args.decision, state)
            return 0

        logs = args.runs / args.run_id / "logs"
        state["status"] = "finetuning"
        atomic_write_json(args.decision, state)
        if not completed_artifact(args.finetune_artifact):
            state["finetune_wall_seconds"] = run_logged(
                [
                    sys.executable,
                    str(ROOT / "scripts/finetune_two_tower_validation.py"),
                    "--config",
                    str(args.finetune_config),
                ],
                logs / "finetune.log",
            )

        state["status"] = "generating_test_candidates"
        atomic_write_json(args.decision, state)
        run_path = args.runs / args.run_id
        if not completed_run(run_path):
            state["candidate_wall_seconds"] = run_logged(
                [
                    sys.executable,
                    str(ROOT / "scripts/run_test_candidate_pool.py"),
                    "--run-id",
                    args.run_id,
                    "--artifact-dir",
                    str(args.finetune_artifact),
                    "--runs",
                    str(args.runs),
                    "--cache",
                    str(args.cache),
                    "--immutable-artifacts",
                    str(args.immutable_artifacts),
                    "--quota",
                    str(args.quota),
                    "--batch-size",
                    str(args.batch_size),
                ],
                logs / "candidate_only_supervisor.log",
            )
        state.update(
            status="completed",
            finished_at=utc_now(),
            artifact=str(args.finetune_artifact),
            run=str(run_path),
            candidates=str(run_path / "candidates/test/two_tower_v2"),
        )
        atomic_write_json(args.decision, state)
        return 0
    except BaseException as error:
        state.update(status="failed", finished_at=utc_now(), error=str(error))
        atomic_write_json(args.decision, state)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
