#!/usr/bin/env python3
"""Promote one completed TwoTower probe through a bounded direct ensemble."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from mla_recsys.artifacts import atomic_write_json, utc_now  # noqa: E402
from scripts.continue_logq_fullfit import gate  # noqa: E402


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-decision", type=Path, required=True)
    parser.add_argument("--primary-temporal", type=Path, required=True)
    parser.add_argument("--primary-test", type=Path, required=True)
    parser.add_argument("--temporal-requests", type=Path, required=True)
    parser.add_argument("--banner-index", type=Path, required=True)
    parser.add_argument("--finetune-config", type=Path, required=True)
    parser.add_argument("--finetune-artifact", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--immutable-artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--minimum-gain", type=float, default=0.0001)
    parser.add_argument("--minimum-first-weight", type=float, default=0.6)
    parser.add_argument("--weight-step", type=float, default=0.05)
    parser.add_argument("--poll-seconds", type=int, default=20)
    parser.add_argument("--timeout-seconds", type=int, default=10800)
    parser.add_argument("--quota", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1024)
    return parser.parse_args()


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


def read_probe(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def completed_artifact(path: Path) -> bool:
    return (path / "manifest.json").is_file()


def completed_run(path: Path) -> bool:
    result = path / "result.json"
    return result.is_file() and json.loads(result.read_text(encoding="utf-8")).get(
        "status"
    ) == "completed"


def main() -> int:
    args = arguments()
    state = {
        "version": 1,
        "status": "waiting_for_probe",
        "started_at": utc_now(),
        "probe_decision": str(args.probe_decision),
    }
    atomic_write_json(args.decision, state)
    started = time.monotonic()
    try:
        while True:
            probe = read_probe(args.probe_decision)
            if probe and probe.get("status") == "completed":
                break
            if probe and probe.get("status") in {
                "failed",
                "artifact_failed",
                "artifact_invalid",
                "artifact_timeout",
            }:
                raise RuntimeError(f"Probe stopped with status={probe.get('status')}")
            if time.monotonic() - started >= args.timeout_seconds:
                raise TimeoutError("Timed out waiting for completed TwoTower probe")
            time.sleep(max(1, args.poll_seconds))

        probe_run = args.runs / str(probe["run_id"])
        new_temporal = probe_run / "candidates/holdout/two_tower_v2"
        run_path = args.runs / args.run_id
        logs = run_path / "logs"
        metrics = run_path / "metrics"
        ensemble_report = metrics / "bounded_direct_ensemble.json"
        baseline_report = metrics / "primary_control.json"
        state.update(
            status="tuning",
            probe_run=str(probe_run),
            new_temporal=str(new_temporal),
            ensemble_report=str(ensemble_report),
            baseline_report=str(baseline_report),
        )
        atomic_write_json(args.decision, state)
        common = [
            "--requests",
            str(args.temporal_requests),
            "--banner-index",
            str(args.banner_index),
            "--candidate-top-k",
            "100",
        ]
        if not ensemble_report.is_file():
            state["tune_wall_seconds"] = run_logged(
                [
                    sys.executable,
                    str(ROOT / "scripts/tune_top50_ensemble.py"),
                    "--input",
                    str(args.primary_temporal),
                    "--input",
                    str(new_temporal),
                    *common,
                    "--weight-step",
                    str(args.weight_step),
                    "--minimum-first-weight",
                    str(args.minimum_first_weight),
                    "--rrf-constants",
                    "0,5,10,20",
                    "--geometry-exponents",
                    "0,0.05,0.1,0.15,0.2",
                    "--geometry-top-n",
                    "75,100",
                    "--refine-top",
                    "10",
                    "--output",
                    str(ensemble_report),
                ],
                logs / "tune.log",
            )
        if not baseline_report.is_file():
            state["control_wall_seconds"] = run_logged(
                [
                    sys.executable,
                    str(ROOT / "scripts/tune_top50_ensemble.py"),
                    "--input",
                    str(args.primary_temporal),
                    *common,
                    "--weight-step",
                    "1",
                    "--minimum-first-weight",
                    "1",
                    "--rrf-constants",
                    "0",
                    "--geometry-exponents",
                    "0",
                    "--geometry-top-n",
                    "75",
                    "--refine-top",
                    "1",
                    "--output",
                    str(baseline_report),
                ],
                logs / "control.log",
            )
        result = gate(
            json.loads(ensemble_report.read_text(encoding="utf-8")),
            json.loads(baseline_report.read_text(encoding="utf-8")),
            args.minimum_gain,
        )
        state["gate"] = result
        if not result["accepted"]:
            state.update(status="rejected", finished_at=utc_now())
            atomic_write_json(args.decision, state)
            return 0

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

        selected = result["candidate"]
        weights = [float(value) for value in selected["weights"]]
        if len(weights) != 2:
            raise RuntimeError(f"Expected two selected weights, got {weights}")
        state.update(status="materializing", selected=selected)
        atomic_write_json(args.decision, state)
        report_path = args.output.with_name(args.output.name + ".report.json")
        if not (args.output.is_file() and report_path.is_file()):
            state["materialize_wall_seconds"] = run_logged(
                [
                    sys.executable,
                    str(ROOT / "scripts/make_top50_ensemble_submission.py"),
                    "--input",
                    str(args.primary_test),
                    "--weight",
                    str(weights[0]),
                    "--input",
                    str(run_path / "candidates/test/two_tower_v2"),
                    "--weight",
                    str(weights[1]),
                    "--requests",
                    str(run_path / "data/test_requests.parquet"),
                    "--banner-index",
                    str(args.banner_index),
                    "--candidate-top-k",
                    "100",
                    "--rrf-constant",
                    str(selected["rrf_constant"]),
                    "--exponent",
                    str(selected["exponent"]),
                    "--rerank-top-n",
                    str(selected["rerank_top_n"]),
                    "--output",
                    str(args.output),
                ],
                logs / "materialize.log",
            )
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("status") != "completed":
            raise RuntimeError("Submission contract did not complete")
        state.update(
            status="completed",
            finished_at=utc_now(),
            artifact=str(args.finetune_artifact),
            run=str(run_path),
            output=str(args.output),
            submission=report,
        )
        atomic_write_json(args.decision, state)
        return 0
    except BaseException as error:
        state.update(status="failed", finished_at=utc_now(), error=str(error))
        atomic_write_json(args.decision, state)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
