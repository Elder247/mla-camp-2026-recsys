#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mla_recsys.artifacts import atomic_write_json, utc_now  # noqa: E402


def load_config(path: Path):
    cfg = OmegaConf.load(path)
    parent = cfg.get("extends")
    if parent:
        base = load_config((path.parent / str(parent)).resolve())
        cfg = OmegaConf.merge(base, cfg)
        del cfg["extends"]
    return cfg


def validate_walk_forward_artifact(artifact_dir: Path) -> dict:
    manifest_path = artifact_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise RuntimeError(f"Walk-forward status is {manifest.get('status')}")
    snapshots = dict(manifest.get("snapshots") or {})
    if len(snapshots) != len(manifest.get("weeks") or []):
        raise RuntimeError("Snapshot/week count mismatch")
    required = (
        "model.pt",
        "candidate_embeddings.npy",
        "candidate_metadata.parquet",
        "manifest.json",
    )
    paths = [Path(str(value["path"])) for value in snapshots.values()]
    paths.append(Path(str(manifest["final_artifact"])))
    missing = [
        str(path / name)
        for path in paths
        for name in required
        if not (path / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(f"Incomplete walk-forward snapshots: {missing[:5]}")
    return {
        "weeks": len(manifest["weeks"]),
        "snapshots": len(snapshots),
        "final_artifact": str(manifest["final_artifact"]),
        "oof_requests": str(manifest["oof_requests"]),
        "validation_health": manifest["validation_health"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Wait for weekly YT data, then train the walk-forward model"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=int, default=20)
    parser.add_argument("--wait-timeout-seconds", type=int, default=3600)
    args = parser.parse_args()
    cfg = load_config(args.config.resolve())
    artifact_dir = Path(str(cfg.paths.artifact_dir))
    state_path = artifact_dir / "training_supervisor.json"
    dataset_state_path = artifact_dir / "weekly_dataset.json"
    started = time.monotonic()
    atomic_write_json(
        state_path,
        {
            "version": 1,
            "status": "waiting_for_weekly_dataset",
            "started_at": utc_now(),
            "config": str(args.config.resolve()),
        },
    )
    while True:
        state = (
            json.loads(dataset_state_path.read_text(encoding="utf-8"))
            if dataset_state_path.is_file()
            else {}
        )
        if state.get("status") in {"completed", "cache_hit"}:
            break
        if time.monotonic() - started >= args.wait_timeout_seconds:
            atomic_write_json(
                state_path,
                {
                    "version": 1,
                    "status": "failed",
                    "error": "weekly_dataset_timeout",
                    "last_dataset_status": state.get("status"),
                    "finished_at": utc_now(),
                },
            )
            return 1
        time.sleep(max(1, args.poll_seconds))

    atomic_write_json(
        state_path,
        {
            "version": 1,
            "status": "training",
            "started_at": utc_now(),
            "dataset_operation_id": state.get("operation_id"),
        },
    )
    training_started = time.monotonic()
    command = [
        str(Path(sys.executable)),
        str(ROOT / "scripts" / "train_two_tower_walk_forward.py"),
        "--config",
        str(args.config.resolve()),
    ]
    result = subprocess.run(command, cwd=ROOT, check=False)
    if result.returncode != 0:
        atomic_write_json(
            state_path,
            {
                "version": 1,
                "status": "failed",
                "error": "training_process_failed",
                "return_code": result.returncode,
                "finished_at": utc_now(),
                "wall_seconds": time.monotonic() - started,
            },
        )
        return result.returncode
    contract = validate_walk_forward_artifact(artifact_dir)
    atomic_write_json(
        state_path,
        {
            "version": 1,
            "status": "completed",
            "finished_at": utc_now(),
            "wall_seconds": time.monotonic() - started,
            "training_seconds": time.monotonic() - training_started,
            "contract": contract,
        },
    )
    print(json.dumps(contract, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
