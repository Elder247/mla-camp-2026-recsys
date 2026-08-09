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

from mla_recsys.artifacts import atomic_write_json, utc_now  # noqa: E402


def retrieval_gate(
    metrics: dict,
    *,
    min_sc50_gain: float,
    max_sc500_loss: float,
    min_union_sc500_gain: float,
) -> dict:
    current50 = float(metrics["current"]["50"]["sourcecost_recall"])
    baseline50 = float(metrics["baseline"]["50"]["sourcecost_recall"])
    current500 = float(metrics["current"]["500"]["sourcecost_recall"])
    baseline500 = float(metrics["baseline"]["500"]["sourcecost_recall"])
    union500 = float(metrics["oracle_union"]["500"]["sourcecost_recall"])
    checks = {
        "sc50_gain": current50 - baseline50 >= min_sc50_gain,
        "sc500_floor": current500 >= baseline500 - max_sc500_loss,
        "union_sc500_gain": union500 - max(current500, baseline500)
        >= min_union_sc500_gain,
    }
    return {
        "accepted": all(checks.values()),
        "checks": checks,
        "current_sc50": current50,
        "baseline_sc50": baseline50,
        "current_sc500": current500,
        "baseline_sc500": baseline500,
        "union_sc500": union500,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Gate v3 retrieval, then start predict-before-update OOF training"
    )
    parser.add_argument("--probe-decision", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=10)
    parser.add_argument("--max-wait-seconds", type=int, default=3600)
    parser.add_argument("--min-sc50-gain", type=float, default=0.002)
    parser.add_argument("--max-sc500-loss", type=float, default=0.01)
    parser.add_argument("--min-union-sc500-gain", type=float, default=0.002)
    args = parser.parse_args()
    started = time.monotonic()
    state = {
        "version": 1,
        "status": "waiting_for_probe",
        "started_at": utc_now(),
        "probe_decision": str(args.probe_decision),
        "config": str(args.config.resolve()),
    }
    atomic_write_json(args.decision, state)
    while True:
        probe = (
            json.loads(args.probe_decision.read_text(encoding="utf-8"))
            if args.probe_decision.is_file()
            else {}
        )
        status = str(probe.get("status") or "missing")
        if status == "completed":
            break
        if status in {
            "failed",
            "artifact_failed",
            "artifact_invalid",
            "artifact_timeout",
        }:
            state.update(status="probe_failed", probe_status=status, finished_at=utc_now())
            atomic_write_json(args.decision, state)
            return 2
        if time.monotonic() - started >= args.max_wait_seconds:
            state.update(status="probe_timeout", probe_status=status, finished_at=utc_now())
            atomic_write_json(args.decision, state)
            return 3
        time.sleep(max(1, args.poll_seconds))

    gate = retrieval_gate(
        probe["metrics"],
        min_sc50_gain=args.min_sc50_gain,
        max_sc500_loss=args.max_sc500_loss,
        min_union_sc500_gain=args.min_union_sc500_gain,
    )
    state.update(gate=gate)
    if not gate["accepted"]:
        state.update(status="rejected", finished_at=utc_now())
        atomic_write_json(args.decision, state)
        return 0

    commands = [
        [
            sys.executable,
            str(ROOT / "scripts" / "prepare_two_tower_weekly_dataset.py"),
            "--config",
            str(args.config.resolve()),
            "--validate-only",
        ],
        [
            sys.executable,
            str(ROOT / "scripts" / "continue_walk_forward_training.py"),
            "--config",
            str(args.config.resolve()),
            "--poll-seconds",
            "5",
            "--wait-timeout-seconds",
            "3600",
        ],
    ]
    state.update(status="running", commands=[], gate=gate)
    atomic_write_json(args.decision, state)
    for command in commands:
        stage_started = time.monotonic()
        result = subprocess.run(command, cwd=ROOT, check=False)
        state["commands"].append(
            {
                "command": command,
                "return_code": result.returncode,
                "wall_seconds": time.monotonic() - stage_started,
            }
        )
        atomic_write_json(args.decision, state)
        if result.returncode != 0:
            state.update(status="failed", finished_at=utc_now())
            atomic_write_json(args.decision, state)
            return result.returncode
    state.update(
        status="completed",
        finished_at=utc_now(),
        wall_seconds=time.monotonic() - started,
    )
    atomic_write_json(args.decision, state)
    print(json.dumps(state, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
