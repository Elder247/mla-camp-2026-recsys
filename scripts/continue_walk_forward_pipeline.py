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

from mla_recsys.artifacts import (  # noqa: E402
    atomic_write_json,
    fingerprint_file,
    utc_now,
)


def promote_final_artifact(artifact_dir: Path, final_artifact: Path) -> dict:
    manifest_path = artifact_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise RuntimeError("Cannot promote final artifact before weekly completion")
    required = [
        final_artifact / name
        for name in (
            "model.pt",
            "candidate_embeddings.npy",
            "candidate_metadata.parquet",
            "manifest.json",
        )
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Final override is incomplete: {missing}")
    previous = str(manifest.get("final_artifact") or "")
    manifest.update(
        original_walk_forward_final_artifact=previous,
        final_artifact=str(final_artifact),
        final_artifact_source="configured_full_quality_override",
        final_lifecycle={
            "predict_state": "configured_full_quality_override",
            "trained_on": "full_prevalidation_raw_train",
            "target_week_seen": False,
            "purpose": "validation_and_test",
        },
        final_artifact_inputs=[fingerprint_file(path) for path in required],
    )
    atomic_write_json(manifest_path, manifest)
    metrics_path = artifact_dir / "metrics.json"
    if metrics_path.is_file():
        atomic_write_json(metrics_path, manifest)
    return {
        "previous": previous,
        "selected": str(final_artifact),
        "inputs": manifest["final_artifact_inputs"],
    }


def pipeline_command(
    *,
    python: Path,
    experiment: str,
    run_id: str,
    mode: str,
    scope: str,
    output_runs: Path,
    cache: Path,
    immutable_artifacts: Path,
) -> list[str]:
    return [
        str(python),
        str(ROOT / "scripts" / "run_pipeline.py"),
        f"experiment={experiment}",
        f"run_id={run_id}",
        f"mode={mode}",
        f"scope={scope}",
        f"paths.root={ROOT}",
        f"paths.runs={output_runs}",
        f"paths.cache={cache}",
        f"paths.immutable_artifacts={immutable_artifacts}",
    ]


def completed_run(path: Path) -> bool:
    result = path / "result.json"
    if not result.is_file():
        return False
    return json.loads(result.read_text(encoding="utf-8")).get("status") == "completed"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run smoke, temporal gate and full after walk-forward training"
    )
    parser.add_argument("--training-state", type=Path, required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--smoke-run", required=True)
    parser.add_argument("--temporal-run", required=True)
    parser.add_argument("--full-run", required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--immutable-artifacts", type=Path, required=True)
    parser.add_argument("--walk-forward-artifact", type=Path)
    parser.add_argument("--final-artifact-override", type=Path)
    parser.add_argument("--poll-seconds", type=int, default=20)
    parser.add_argument("--max-wait-seconds", type=int, default=7200)
    args = parser.parse_args()
    decision_path = Path(f"/tmp/{args.temporal_run}.sequence.json")
    decision = {
        "version": 1,
        "status": "waiting_for_walk_forward",
        "started_at": utc_now(),
        "experiment": args.experiment,
        "smoke_run": args.smoke_run,
        "temporal_run": args.temporal_run,
        "full_run": args.full_run,
    }
    atomic_write_json(decision_path, decision)
    started = time.monotonic()
    while True:
        training = (
            json.loads(args.training_state.read_text(encoding="utf-8"))
            if args.training_state.is_file()
            else {}
        )
        if training.get("status") == "completed":
            break
        if training.get("status") == "failed":
            decision.update(
                status="walk_forward_failed",
                finished_at=utc_now(),
                training=training,
            )
            atomic_write_json(decision_path, decision)
            return 2
        if time.monotonic() - started >= args.max_wait_seconds:
            decision.update(status="walk_forward_timeout", finished_at=utc_now())
            atomic_write_json(decision_path, decision)
            return 3
        time.sleep(max(1, args.poll_seconds))

    if bool(args.walk_forward_artifact) != bool(args.final_artifact_override):
        parser.error(
            "--walk-forward-artifact and --final-artifact-override must be used together"
        )
    if args.walk_forward_artifact and args.final_artifact_override:
        final_promotion = promote_final_artifact(
            args.walk_forward_artifact, args.final_artifact_override
        )
        decision["final_artifact_promotion"] = final_promotion
        atomic_write_json(decision_path, decision)

    phases = [
        (args.smoke_run, "smoke", "offline"),
        (args.temporal_run, "offline", "offline"),
    ]
    decision["commands"] = []
    for run_id, mode, scope in phases:
        command = pipeline_command(
            python=args.python,
            experiment=args.experiment,
            run_id=run_id,
            mode=mode,
            scope=scope,
            output_runs=args.runs,
            cache=args.cache,
            immutable_artifacts=args.immutable_artifacts,
        )
        if completed_run(args.runs / run_id):
            decision["commands"].append(
                {"run_id": run_id, "status": "resume_completed", "command": command}
            )
            atomic_write_json(decision_path, decision)
            continue
        phase_started = time.monotonic()
        decision.update(status=f"{mode}_running", active_run=run_id)
        atomic_write_json(decision_path, decision)
        return_code = subprocess.run(command, cwd=ROOT, check=False).returncode
        record = {
            "run_id": run_id,
            "mode": mode,
            "status": "completed" if return_code == 0 else "failed",
            "return_code": return_code,
            "wall_seconds": time.monotonic() - phase_started,
            "command": command,
        }
        decision["commands"].append(record)
        atomic_write_json(decision_path, decision)
        if return_code != 0:
            decision.update(status=f"{mode}_failed", finished_at=utc_now())
            atomic_write_json(decision_path, decision)
            return return_code

    decision.update(status="promotion_running", active_run=args.full_run)
    atomic_write_json(decision_path, decision)
    promotion = [
        str(args.python),
        str(ROOT / "scripts" / "continue_to_full.py"),
        "--temporal-run",
        args.temporal_run,
        "--full-run",
        args.full_run,
        "--experiment",
        args.experiment,
        "--source-runs",
        str(args.runs),
        "--output-runs",
        str(args.runs),
        "--immutable-artifacts",
        str(args.immutable_artifacts),
        "--poll-seconds",
        str(args.poll_seconds),
    ]
    promotion_started = time.monotonic()
    return_code = subprocess.run(promotion, cwd=ROOT, check=False).returncode
    decision["commands"].append(
        {
            "run_id": args.full_run,
            "mode": "promotion_and_full",
            "return_code": return_code,
            "wall_seconds": time.monotonic() - promotion_started,
            "command": promotion,
        }
    )
    decision.update(
        status="completed" if return_code == 0 else "promotion_or_full_failed",
        finished_at=utc_now(),
        wall_seconds=time.monotonic() - started,
    )
    atomic_write_json(decision_path, decision)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
