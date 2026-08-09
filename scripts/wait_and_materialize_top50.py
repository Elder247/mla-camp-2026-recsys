#!/usr/bin/env python3
"""Wait for a full run and materialize one configured cached top-50 ensemble."""
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


def read_status(path: Path) -> str:
    if not path.is_file():
        return "missing"
    return str(json.loads(path.read_text(encoding="utf-8")).get("status"))


def materialize_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts" / "make_top50_ensemble_submission.py"),
    ]
    for path in args.input:
        command.extend(["--input", str(path)])
    for weight in args.weight:
        command.extend(["--weight", str(weight)])
    command.extend(
        [
            "--requests",
            str(args.requests),
            "--banner-index",
            str(args.banner_index),
            "--candidate-top-k",
            str(args.candidate_top_k),
            "--rrf-constant",
            str(args.rrf_constant),
            "--exponent",
            str(args.exponent),
            "--rerank-top-n",
            str(args.rerank_top_n),
            "--output",
            str(args.output),
            "--report",
            str(args.report),
        ]
    )
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-run", type=Path, required=True)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--weight", type=float, action="append", required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--banner-index", type=Path, required=True)
    parser.add_argument("--candidate-top-k", type=int, default=100)
    parser.add_argument("--rrf-constant", type=float, required=True)
    parser.add_argument("--exponent", type=float, required=True)
    parser.add_argument("--rerank-top-n", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=10)
    parser.add_argument("--timeout-seconds", type=int, default=14400)
    args = parser.parse_args()
    if len(args.input) != len(args.weight):
        parser.error("--input and --weight counts must match")
    if args.poll_seconds <= 0 or args.timeout_seconds <= 0:
        parser.error("poll and timeout values must be positive")

    state = {
        "version": 1,
        "status": "waiting",
        "started_at": utc_now(),
        "wait_run": str(args.wait_run),
        "inputs": [str(path) for path in args.input],
        "weights": [float(value) for value in args.weight],
        "rrf_constant": float(args.rrf_constant),
        "exponent": float(args.exponent),
        "rerank_top_n": int(args.rerank_top_n),
        "output": str(args.output),
    }
    atomic_write_json(args.decision, state)
    started = time.monotonic()
    try:
        result_path = args.wait_run / "result.json"
        while True:
            status = read_status(result_path)
            if status == "completed":
                break
            if status == "failed":
                raise RuntimeError(f"Upstream full run failed: {args.wait_run}")
            if time.monotonic() - started >= args.timeout_seconds:
                raise TimeoutError(f"Timed out waiting for {args.wait_run}")
            time.sleep(args.poll_seconds)

        command = materialize_command(args)
        state.update(
            status="materializing",
            materialize_started_at=utc_now(),
            command=command,
        )
        atomic_write_json(args.decision, state)
        args.log.parent.mkdir(parents=True, exist_ok=True)
        stage_started = time.monotonic()
        with args.log.open("a", encoding="utf-8") as target:
            result = subprocess.run(
                command,
                cwd=ROOT,
                stdout=target,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode != 0:
            raise RuntimeError(f"Materializer failed with {result.returncode}")
        report = json.loads(args.report.read_text(encoding="utf-8"))
        if report.get("status") != "completed" or not report.get(
            "validation", {}
        ).get("ok"):
            raise RuntimeError("Materialized submission contract failed")
        state.update(
            status="completed",
            finished_at=utc_now(),
            materialize_wall_seconds=time.monotonic() - stage_started,
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
