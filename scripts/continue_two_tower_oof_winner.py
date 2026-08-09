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
sys.path.insert(0, str(ROOT))
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


def selected_trial(selection_path: Path, name: str) -> dict:
    payload = json.loads(selection_path.read_text(encoding="utf-8"))
    matches = [trial for trial in payload.get("trials", []) if trial.get("name") == name]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one selected trial for {name}, got {len(matches)}")
    return matches[0]


def run_logged(command: list[str], *, log_path: Path) -> float:
    started = time.monotonic()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed with {result.returncode}: {' '.join(command)}"
        )
    return time.monotonic() - started


def make_direct_submission(
    *,
    selection_path: Path,
    selected: str,
    run_id: str,
    runs: Path,
    cache: Path,
    immutable_artifacts: Path,
    direct_artifact: Path | None = None,
) -> dict:
    trial = selected_trial(selection_path, selected)
    base_full_artifact = Path(str(trial["artifact"]))
    full_artifact = direct_artifact or base_full_artifact
    probe = json.loads(Path(str(trial["probe"])).read_text(encoding="utf-8"))
    probe_run = runs / str(probe["run_id"])
    direct_run = runs / run_id
    output = direct_run / "predictions/test_top50_two_tower_geometry.parquet"
    geometry = direct_run / "metrics/two_tower_geometry.json"
    log = direct_run / "logs/direct_two_tower_submission.log"
    common = [
        "experiment=i2_two_tower_v2_probe",
        f"run_id={run_id}",
        "mode=full",
        "scope=full",
        f"paths.root={ROOT}",
        f"paths.runs={runs}",
        f"paths.cache={cache}",
        f"paths.immutable_artifacts={immutable_artifacts}",
        f"paths.two_tower_v2_artifact={full_artifact}",
    ]
    timings = {}
    timings["prepare_data"] = run_logged(
        [sys.executable, str(ROOT / "scripts/prepare_data.py"), *common],
        log_path=log,
    )
    timings["geometry"] = run_logged(
        [
            sys.executable,
            str(ROOT / "scripts/tune_two_tower_geometry.py"),
            "--run",
            str(probe_run),
            "--artifact",
            str(full_artifact),
            "--source",
            "two_tower_v2",
            "--output",
            str(geometry),
        ],
        log_path=log,
    )
    geometry_report = json.loads(geometry.read_text(encoding="utf-8"))
    best = geometry_report["best"]
    timings["generate_test"] = run_logged(
        [
            sys.executable,
            str(ROOT / "scripts/generate_candidates.py"),
            *common,
            "split=test",
            "cg=two_tower_v2",
        ],
        log_path=log,
    )
    timings["make_submission"] = run_logged(
        [
            sys.executable,
            str(ROOT / "scripts/make_two_tower_submission.py"),
            "--run",
            str(direct_run),
            "--artifact",
            str(full_artifact),
            "--source",
            "two_tower_v2",
            "--exponent",
            str(best["exponent"]),
            "--rerank-top-n",
            str(best["rerank_top_n"]),
            "--output",
            str(output),
        ],
        log_path=log,
    )
    report = json.loads(
        (direct_run / "metrics/two_tower_submission.json").read_text(encoding="utf-8")
    )
    if report.get("status") != "completed":
        raise RuntimeError("Direct TwoTower submission contract failed")
    return {
        "status": "completed",
        "run_id": run_id,
        "probe_run": str(probe_run),
        "base_full_artifact": str(base_full_artifact),
        "full_artifact": str(full_artifact),
        "geometry": best,
        "submission": report,
        "timings": timings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-pid", type=int, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--direct-run-id")
    parser.add_argument(
        "--direct-only",
        action="store_true",
        help="Materialize the optional direct submission and skip OOF training",
    )
    parser.add_argument("--runs", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--immutable-artifacts", type=Path)
    parser.add_argument(
        "--direct-artifact-config",
        type=Path,
        help="Optional test-only full-validation fine-tune config",
    )
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
        direct_args = (args.runs, args.cache, args.immutable_artifacts)
        if args.direct_run_id and not all(direct_args):
            parser.error(
                "--runs, --cache and --immutable-artifacts are required with "
                "--direct-run-id"
            )
        if args.direct_run_id:
            decision.update(status="direct_submission", selected=name)
            atomic_write_json(args.decision, decision)
            try:
                direct_artifact = None
                if args.direct_artifact_config:
                    from scripts.finetune_two_tower_validation import load_config

                    fit_cfg = load_config(args.direct_artifact_config.resolve())
                    direct_artifact = Path(str(fit_cfg.paths.artifact_dir))
                    if not (direct_artifact / "manifest.json").is_file():
                        decision.update(
                            status="direct_artifact_training",
                            direct_artifact=str(direct_artifact),
                        )
                        atomic_write_json(args.decision, decision)
                        run_logged(
                            [
                                sys.executable,
                                str(ROOT / "scripts/finetune_two_tower_validation.py"),
                                "--config",
                                str(args.direct_artifact_config.resolve()),
                            ],
                            log_path=args.log,
                        )
                decision["direct_submission"] = make_direct_submission(
                    selection_path=args.selection,
                    selected=name,
                    run_id=args.direct_run_id,
                    runs=args.runs,
                    cache=args.cache,
                    immutable_artifacts=args.immutable_artifacts,
                    direct_artifact=direct_artifact,
                )
            except BaseException as error:
                # A fast direct leaderboard candidate is valuable but must not
                # block the leakage-safe OOF/temporal path.
                decision["direct_submission"] = {
                    "status": "failed",
                    "error": str(error),
                }
            atomic_write_json(args.decision, decision)
            if args.direct_only:
                if decision["direct_submission"].get("status") != "completed":
                    raise RuntimeError(
                        "Direct TwoTower submission failed: "
                        f"{decision['direct_submission'].get('error')}"
                    )
                decision.update(status="completed", finished_at=utc_now())
                atomic_write_json(args.decision, decision)
                return 0
        elif args.direct_only:
            parser.error("--direct-run-id is required with --direct-only")
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
