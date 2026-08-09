#!/usr/bin/env python3
"""Tune and materialize old/chrono/v7 rankings after the direct v7 run."""
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


def completed(path: Path) -> bool:
    return path.is_file() and json.loads(path.read_text(encoding="utf-8")).get(
        "status"
    ) == "completed"


def selected_parameters(report: dict) -> dict:
    best = report["best"]
    weights = tuple(float(value) for value in best["weights"])
    if len(weights) != 3:
        raise ValueError("Three-pool ensemble requires exactly three weights")
    return {
        "weights": weights,
        "rrf_constant": float(best["rrf_constant"]),
        "exponent": float(best["exponent"]),
        "rerank_top_n": int(best["rerank_top_n"]),
        "tune_sc50": float(best["tune_metrics"]["50"]["sourcecost_recall"]),
        "validation_sc50": float(
            best["validation_metrics"]["50"]["sourcecost_recall"]
        ),
        "full_sc50": float(best["full_metrics"]["50"]["sourcecost_recall"]),
    }


def run_logged(command: list[str], log: Path) -> float:
    started = time.monotonic()
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as target:
        result = subprocess.run(
            command,
            cwd=ROOT,
            stdout=target,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with {result.returncode}: {' '.join(command)}")
    return time.monotonic() - started


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-decision", type=Path, required=True)
    parser.add_argument("--direct-report", type=Path, required=True)
    parser.add_argument("--old-temporal-ranking", type=Path, required=True)
    parser.add_argument("--chrono-temporal-ranking", type=Path, required=True)
    parser.add_argument("--old-full-ranking", type=Path, required=True)
    parser.add_argument("--chrono-full-ranking", type=Path, required=True)
    parser.add_argument("--direct-run", type=Path, required=True)
    parser.add_argument("--banner-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=10)
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    args = parser.parse_args()
    state = {
        "version": 1,
        "status": "waiting",
        "started_at": utc_now(),
        "probe_decision": str(args.probe_decision),
        "direct_report": str(args.direct_report),
    }
    atomic_write_json(args.decision, state)
    started = time.monotonic()
    try:
        while not (
            completed(args.probe_decision) and completed(args.direct_report)
        ):
            if time.monotonic() - started >= args.timeout_seconds:
                raise TimeoutError("Timed out waiting for v7 probe/direct candidates")
            time.sleep(max(1, args.poll_seconds))
        probe = json.loads(args.probe_decision.read_text(encoding="utf-8"))
        probe_run = args.direct_run.parent / str(probe["run_id"])
        temporal_output = args.direct_run / "metrics/top50_three_pool_temporal.json"
        log = args.direct_run / "logs/top50_three_pool.log"
        state.update(status="tuning", probe_run=str(probe_run))
        atomic_write_json(args.decision, state)
        state["tune_wall_seconds"] = run_logged(
            [
                sys.executable,
                str(ROOT / "scripts/tune_top50_ensemble.py"),
                "--input",
                str(args.old_temporal_ranking),
                "--input",
                str(args.chrono_temporal_ranking),
                "--input",
                str(probe_run / "candidates/holdout/two_tower_v2"),
                "--requests",
                str(probe_run / "data/holdout_requests.parquet"),
                "--banner-index",
                str(args.banner_index),
                "--weight-step",
                "0.1",
                "--rrf-constants",
                "0,5,10,20,40",
                "--geometry-exponents",
                "0,0.1,0.2,0.3",
                "--geometry-top-n",
                "50,75,100,150",
                "--refine-top",
                "10",
                "--candidate-top-k",
                "100",
                "--output",
                str(temporal_output),
            ],
            log,
        )
        selected = selected_parameters(
            json.loads(temporal_output.read_text(encoding="utf-8"))
        )
        state.update(status="materializing", selected=selected)
        atomic_write_json(args.decision, state)
        command = [
            sys.executable,
            str(ROOT / "scripts/make_top50_ensemble_submission.py"),
        ]
        for ranking in (
            args.old_full_ranking,
            args.chrono_full_ranking,
            args.direct_run / "candidates/test/two_tower_v2",
        ):
            command.extend(["--input", str(ranking)])
        for weight in selected["weights"]:
            command.extend(["--weight", str(weight)])
        command.extend(
            [
                "--requests",
                str(args.direct_run / "data/test_requests.parquet"),
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
            ]
        )
        state["materialize_wall_seconds"] = run_logged(command, log)
        report = json.loads(
            args.output.with_name(args.output.name + ".report.json").read_text(
                encoding="utf-8"
            )
        )
        if report.get("status") != "completed":
            raise RuntimeError("Three-pool submission contract failed")
        state.update(
            status="completed",
            finished_at=utc_now(),
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
