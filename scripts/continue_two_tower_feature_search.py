#!/usr/bin/env python3
"""Select the best completed 10M TwoTower probe and scale it to 100M."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mla_recsys.artifacts import atomic_write_json, utc_now  # noqa: E402


TRIALS = (
    {
        "name": "v4_scweighted",
        "probe": Path("/tmp/20260809_1315_i4_tt_v4_10m_probe_r1.probe.json"),
        "config": ROOT / "configs/two_tower/v4_scweighted_chrono_100m.yaml",
        "artifact": Path(
            "/home/astrofimuk/workspace/mla_two_stage/artifacts/"
            "two_tower_v4_scweighted_chrono_100m_model"
        ),
    },
    {
        "name": "v5_ad_metadata",
        "probe": Path("/tmp/20260809_1345_i5_tt_metadata_10m_probe.probe.json"),
        "config": ROOT / "configs/two_tower/v5_ad_metadata_chrono_100m.yaml",
        "artifact": Path(
            "/home/astrofimuk/workspace/mla_two_stage/artifacts/"
            "two_tower_v5_ad_metadata_chrono_100m_model"
        ),
    },
    {
        "name": "v6_context_metadata",
        "probe": Path("/tmp/20260809_1355_i6_tt_context_10m_probe_r1.probe.json"),
        "config": ROOT / "configs/two_tower/v6_context_metadata_chrono_100m.yaml",
        "artifact": Path(
            "/home/astrofimuk/workspace/mla_two_stage/artifacts/"
            "two_tower_v6_context_metadata_chrono_100m_model"
        ),
    },
    {
        "name": "v7_large_batch",
        "probe": Path("/tmp/20260809_1405_i7_tt_large_batch_10m_probe.probe.json"),
        "config": ROOT / "configs/two_tower/v7_large_batch_chrono_100m.yaml",
        "artifact": Path(
            "/home/astrofimuk/workspace/mla_two_stage/artifacts/"
            "two_tower_v7_large_batch_chrono_100m_model"
        ),
    },
)


def process_exists(pid: int) -> bool:
    return pid > 0 and Path(f"/proc/{pid}").exists()


def wait_for_processes(pids: list[int], timeout_seconds: int) -> None:
    started = time.monotonic()
    while any(process_exists(pid) for pid in pids):
        if time.monotonic() - started >= timeout_seconds:
            raise TimeoutError(f"Timed out waiting for processes: {pids}")
        time.sleep(5)


def trial_metrics(probe: Path) -> dict[str, float]:
    payload = json.loads(probe.read_text(encoding="utf-8"))
    if payload.get("status") != "completed":
        raise RuntimeError(f"Probe is not completed: {probe}: {payload.get('status')}")
    current = payload["metrics"]["current"]
    oracle = payload["metrics"]["oracle_union"]
    complementarity = payload.get("complementarity") or payload.get("metrics", {}).get(
        "complementarity", {}
    )
    return {
        "recall_at_50": float(current["50"]["recall"]),
        "sourcecost_recall_at_50": float(current["50"]["sourcecost_recall"]),
        "sourcecost_recall_at_500": float(current["500"]["sourcecost_recall"]),
        "oracle_sourcecost_recall_at_50": float(
            oracle["50"]["sourcecost_recall"]
        ),
        "new_only_sourcecost_share": float(
            complementarity.get("new_only_sourcecost_share", 0.0)
        ),
    }


def select_trial(
    trials: list[dict[str, Any]], *, minimum_sourcecost_recall_at_500: float
) -> dict[str, Any]:
    eligible = [
        trial
        for trial in trials
        if trial["metrics"]["sourcecost_recall_at_500"]
        >= minimum_sourcecost_recall_at_500
    ]
    if not eligible:
        raise RuntimeError("Every 10M trial failed the SC@500 safety floor")
    return max(
        eligible,
        key=lambda trial: (
            trial["metrics"]["oracle_sourcecost_recall_at_50"],
            trial["metrics"]["sourcecost_recall_at_50"],
            trial["metrics"]["new_only_sourcecost_share"],
            trial["metrics"]["recall_at_50"],
            trial["metrics"]["sourcecost_recall_at_500"],
        ),
    )


def run_logged(command: list[str], log_path: Path) -> int:
    started = time.monotonic()
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return_code = int(result.returncode)
    print(
        json.dumps(
            {
                "command": command,
                "return_code": return_code,
                "wall_seconds": time.monotonic() - started,
                "log": str(log_path),
            }
        ),
        flush=True,
    )
    return return_code


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-pid", type=int, action="append", default=[])
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    parser.add_argument("--minimum-sourcecost-recall-at-500", type=float, default=0.665)
    parser.add_argument(
        "--decision",
        type=Path,
        default=Path("/tmp/20260809_1425_tt_10m_selection.json"),
    )
    parser.add_argument("--probe-run-id", default="20260809_1430_i8_tt_winner_100m_probe")
    args = parser.parse_args()
    decision: dict[str, Any] = {
        "version": 1,
        "status": "waiting",
        "started_at": utc_now(),
        "wait_pids": args.wait_pid,
    }
    atomic_write_json(args.decision, decision)
    try:
        wait_for_processes(args.wait_pid, args.timeout_seconds)
        measured = []
        for raw in TRIALS:
            trial = dict(raw)
            trial["metrics"] = trial_metrics(Path(trial["probe"]))
            measured.append(trial)
        selected = select_trial(
            measured,
            minimum_sourcecost_recall_at_500=args.minimum_sourcecost_recall_at_500,
        )
        serializable = [
            {
                **trial,
                "probe": str(trial["probe"]),
                "config": str(trial["config"]),
                "artifact": str(trial["artifact"]),
            }
            for trial in measured
        ]
        decision.update(
            status="selected",
            trials=serializable,
            selected=selected["name"],
            minimum_sourcecost_recall_at_500=args.minimum_sourcecost_recall_at_500,
        )
        atomic_write_json(args.decision, decision)

        artifact = Path(selected["artifact"])
        manifest = artifact / "manifest.json"
        if not manifest.is_file():
            if artifact.exists() and any(artifact.iterdir()):
                raise RuntimeError(f"Refusing incomplete non-empty artifact: {artifact}")
            decision["status"] = "training_100m"
            atomic_write_json(args.decision, decision)
            code = run_logged(
                [
                    sys.executable,
                    str(ROOT / "scripts/train_two_tower_v2.py"),
                    "--config",
                    str(selected["config"]),
                ],
                Path("/tmp/20260809_1425_tt_winner_100m.terminal.log"),
            )
            if code != 0:
                raise RuntimeError(f"100M training failed with return code {code}")
        decision["status"] = "probing_100m"
        atomic_write_json(args.decision, decision)
        code = run_logged(
            [
                sys.executable,
                str(ROOT / "scripts/continue_two_tower_probe.py"),
                "--artifact-dir",
                str(artifact),
                "--run-id",
                args.probe_run_id,
                "--runs",
                "/home/astrofimuk/workspace/mla_two_stage/runs",
                "--cache",
                "/home/astrofimuk/workspace/mla_two_stage/cache",
                "--immutable-artifacts",
                "/home/astrofimuk/workspace/mla_two_stage/artifacts",
                "--baseline-run",
                "/home/astrofimuk/workspace/mla_two_stage/runs/20260809_1040_i3_tt_v3_100m_probe",
                "--baseline-source",
                "two_tower_v2",
                "--poll-seconds",
                "5",
                "--max-wait-seconds",
                "900",
            ],
            Path("/tmp/20260809_1430_i8_tt_winner_100m_probe.supervisor.log"),
        )
        if code != 0:
            raise RuntimeError(f"100M probe failed with return code {code}")
        decision.update(status="completed", finished_at=utc_now(), artifact=str(artifact))
        atomic_write_json(args.decision, decision)
        return 0
    except BaseException as error:
        decision.update(status="failed", finished_at=utc_now(), error=str(error))
        atomic_write_json(args.decision, decision)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
