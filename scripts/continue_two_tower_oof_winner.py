#!/usr/bin/env python3
"""Run leakage-safe walk-forward OOF for the selected TwoTower addition."""
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

from mla_recsys.artifacts import (  # noqa: E402
    atomic_write_json,
    fingerprint_file,
    utc_now,
)


OOF_VARIANTS = {
    "v4_scweighted": {
        "config": ROOT / "configs/two_tower/v4_scweighted_walk_forward_100m_s10.yaml",
        "artifact": Path(
            "/home/astrofimuk/workspace/mla_two_stage/artifacts/"
            "two_tower_v4_scweighted_walk_forward_100m_s10"
        ),
    },
    "v5_ad_metadata": {
        "config": ROOT / "configs/two_tower/v5_ad_metadata_walk_forward_100m_s10.yaml",
        "artifact": Path(
            "/home/astrofimuk/workspace/mla_two_stage/artifacts/"
            "two_tower_v5_ad_metadata_walk_forward_100m_s10"
        ),
    },
    "v6_context_metadata": {
        "config": ROOT
        / "configs/two_tower/v6_context_metadata_walk_forward_100m_s10.yaml",
        "artifact": Path(
            "/home/astrofimuk/workspace/mla_two_stage/artifacts/"
            "two_tower_v6_context_metadata_walk_forward_100m_s10"
        ),
    },
    "v7_large_batch": {
        "config": ROOT / "configs/two_tower/v7_large_batch_walk_forward_100m_s10.yaml",
        "artifact": Path(
            "/home/astrofimuk/workspace/mla_two_stage/artifacts/"
            "two_tower_v7_large_batch_walk_forward_100m_s10"
        ),
    },
}

DONOR = Path(
    "/home/astrofimuk/workspace/mla_two_stage/artifacts/"
    "two_tower_v3_bpe_multipos_walk_forward_100m_s10_r1"
)


def wait_for_process(pid: int, timeout_seconds: int) -> None:
    started = time.monotonic()
    while pid > 0 and Path(f"/proc/{pid}").exists():
        if time.monotonic() - started >= timeout_seconds:
            raise TimeoutError(f"Timed out waiting for PID {pid}")
        time.sleep(5)


def materialize_immutable(source: Path, target: Path) -> None:
    if target.is_file():
        if fingerprint_file(source)["sha256"] != fingerprint_file(target)["sha256"]:
            raise RuntimeError(f"Existing reused artifact differs: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def selected_variant(selection_path: Path) -> tuple[str, dict]:
    payload = json.loads(selection_path.read_text(encoding="utf-8"))
    if payload.get("status") != "completed":
        raise RuntimeError(
            f"TwoTower selection is not completed: {payload.get('status')}"
        )
    name = str(payload["selected"])
    if name not in OOF_VARIANTS:
        raise ValueError(f"Unknown selected TwoTower variant: {name}")
    return name, OOF_VARIANTS[name]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-pid", type=int, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    args = parser.parse_args()
    decision = {
        "version": 1,
        "status": "waiting",
        "started_at": utc_now(),
        "wait_pid": args.wait_pid,
        "selection": str(args.selection),
    }
    atomic_write_json(args.decision, decision)
    try:
        wait_for_process(args.wait_pid, args.timeout_seconds)
        name, variant = selected_variant(args.selection)
        artifact = Path(variant["artifact"])
        manifest = artifact / "manifest.json"
        if not manifest.is_file():
            materialize_immutable(
                DONOR / "oof_requests.parquet", artifact / "oof_requests.parquet"
            )
            materialize_immutable(
                DONOR / "history_events.parquet", artifact / "history_events.parquet"
            )
            decision.update(
                status="training",
                selected=name,
                config=str(variant["config"]),
                artifact=str(artifact),
                reused={
                    "oof_requests": fingerprint_file(artifact / "oof_requests.parquet"),
                    "history_events": fingerprint_file(artifact / "history_events.parquet"),
                    "donor": str(DONOR),
                },
            )
            atomic_write_json(args.decision, decision)
            started = time.monotonic()
            with args.log.open("w", encoding="utf-8") as log:
                result = subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / "scripts/train_two_tower_walk_forward.py"),
                        "--config",
                        str(variant["config"]),
                    ],
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            decision["wall_seconds"] = time.monotonic() - started
            decision["return_code"] = int(result.returncode)
            if result.returncode != 0:
                raise RuntimeError(f"Walk-forward training failed: {result.returncode}")
        resolved = json.loads(manifest.read_text(encoding="utf-8"))
        if resolved.get("status") != "completed":
            raise RuntimeError(f"Walk-forward manifest is not completed: {manifest}")
        decision.update(
            status="completed",
            finished_at=utc_now(),
            selected=name,
            artifact=str(artifact),
            weeks=len(resolved.get("weeks") or ()),
            snapshots=len(resolved.get("snapshots") or {}),
        )
        atomic_write_json(args.decision, decision)
        return 0
    except BaseException as error:
        decision.update(status="failed", finished_at=utc_now(), error=str(error))
        atomic_write_json(args.decision, decision)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
