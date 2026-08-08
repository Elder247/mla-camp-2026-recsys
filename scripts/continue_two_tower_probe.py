#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mla_recsys.artifacts import atomic_write_json, utc_now  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Wait for TwoTower artifact and run a detached candidate-only gate"
    )
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--experiment", default="i2_two_tower_v2_probe")
    parser.add_argument("--mode", choices=("smoke", "offline"), default="offline")
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--immutable-artifacts", type=Path, required=True)
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--baseline-source", default="two_tower_fps_v1")
    parser.add_argument("--training-pid", type=int)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--max-wait-seconds", type=int, default=3600)
    args = parser.parse_args()
    decision_path = Path(f"/tmp/{args.run_id}.probe.json")
    decision = {
        "version": 1,
        "run_id": args.run_id,
        "status": "waiting_for_artifact",
        "started_at": utc_now(),
        "artifact_dir": str(args.artifact_dir),
    }
    atomic_write_json(decision_path, decision)
    started = time.monotonic()
    manifest = args.artifact_dir / "manifest.json"
    while not manifest.is_file():
        if time.monotonic() - started >= args.max_wait_seconds:
            decision.update(status="artifact_timeout", finished_at=utc_now())
            atomic_write_json(decision_path, decision)
            return 2
        if args.training_pid and not Path(f"/proc/{args.training_pid}").exists():
            decision.update(status="artifact_failed", finished_at=utc_now())
            atomic_write_json(decision_path, decision)
            return 3
        time.sleep(args.poll_seconds)

    resolved = json.loads(manifest.read_text(encoding="utf-8"))
    if int(resolved.get("candidates", {}).get("candidates", 0)) <= 0:
        decision.update(status="artifact_invalid", finished_at=utc_now())
        atomic_write_json(decision_path, decision)
        return 4
    common = [
        f"experiment={args.experiment}",
        f"run_id={args.run_id}",
        f"mode={args.mode}",
        "scope=offline",
        f"paths.root={ROOT}",
        f"paths.runs={args.runs}",
        f"paths.cache={args.cache}",
        f"paths.immutable_artifacts={args.immutable_artifacts}",
        f"paths.two_tower_v2_artifact={args.artifact_dir}",
    ]
    commands = [
        (
            "prepare_data",
            [sys.executable, str(ROOT / "scripts" / "prepare_data.py"), *common],
        ),
        (
            "generate_holdout_two_tower_v2",
            [
                sys.executable,
                str(ROOT / "scripts" / "generate_candidates.py"),
                *common,
                "split=holdout",
                "cg=two_tower_v2",
            ],
        ),
        (
            "evaluate_holdout_two_tower_v2",
            [
                sys.executable,
                str(ROOT / "scripts" / "evaluate_source_retrieval.py"),
                *common,
                "split=holdout",
                "cg=two_tower_v2",
                f"baseline_run={args.baseline_run}",
                f"baseline_source={args.baseline_source}",
            ],
        ),
    ]
    decision.update(status="running", artifact_manifest=resolved, commands=[])
    atomic_write_json(decision_path, decision)
    run_path = args.runs / args.run_id
    for stage, command in commands:
        stage_started = time.monotonic()
        temporary_log = Path(f"/tmp/{args.run_id}.{stage}.log")
        with temporary_log.open("w", encoding="utf-8") as log:
            return_code = subprocess.run(
                command,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            ).returncode
        log_dir = run_path / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        final_log = log_dir / f"{stage}.log"
        shutil.move(str(temporary_log), final_log)
        record = {
            "stage": stage,
            "return_code": return_code,
            "wall_seconds": time.monotonic() - stage_started,
            "log": str(final_log),
            "command": command,
        }
        decision["commands"].append(record)
        atomic_write_json(decision_path, decision)
        if return_code != 0:
            decision.update(status="failed", failed_stage=stage, finished_at=utc_now())
            atomic_write_json(decision_path, decision)
            return return_code
    report_path = run_path / "metrics" / "retrieval_holdout_two_tower_v2.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    decision.update(
        status="completed",
        finished_at=utc_now(),
        metrics=report["metrics"],
        complementarity=report["complementarity"],
    )
    atomic_write_json(decision_path, decision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
