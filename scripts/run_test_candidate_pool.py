#!/usr/bin/env python3
"""Build one full-scope test candidate pool with the standard run contract."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mla_recsys.artifacts import RunStore  # noqa: E402
from mla_recsys.candidate_cache import enabled_sources  # noqa: E402
from mla_recsys.config import compose_config  # noqa: E402
from mla_recsys.stage_runner import StageRunner  # noqa: E402


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--experiment", default="i2_two_tower_v2_probe")
    parser.add_argument("--source", default="two_tower_v2")
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--immutable-artifacts", type=Path, required=True)
    parser.add_argument("--quota", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1024)
    return parser.parse_args()


def configured_overrides(args: argparse.Namespace) -> list[str]:
    if args.quota <= 0 or args.batch_size <= 0:
        raise ValueError("quota and batch-size must be positive")
    return [
        f"paths.root={ROOT}",
        f"paths.runs={args.runs}",
        f"paths.cache={args.cache}",
        f"paths.immutable_artifacts={args.immutable_artifacts}",
        f"paths.{args.source}_artifact={args.artifact_dir}",
        f"candidates.generators.{args.source}.top_k={args.quota}",
        f"candidates.generators.{args.source}.quota={args.quota}",
        f"candidates.generators.{args.source}.batch_size={args.batch_size}",
    ]


def completed(store: RunStore, stage: str) -> bool:
    path = store.path / "stages" / f"{stage}.json"
    return path.is_file() and json.loads(path.read_text(encoding="utf-8")).get(
        "status"
    ) == "completed"


def main() -> int:
    args = arguments()
    overrides = configured_overrides(args)
    cfg = compose_config(
        args.experiment,
        run_id=args.run_id,
        mode="full",
        scope="full",
        overrides=overrides,
    )
    sources = enabled_sources(cfg)
    if sources != [args.source]:
        raise ValueError(
            "Candidate-only experiment must enable exactly the requested source: "
            f"requested={args.source}, enabled={sources}"
        )
    artifact_key = str(cfg.candidates.generators[args.source].artifact_path_key)
    if Path(str(cfg.paths[artifact_key])) != args.artifact_dir:
        raise ValueError(f"Artifact override did not resolve through {artifact_key}")
    store = RunStore.initialize(cfg, repo_root=ROOT, resume=True)
    runner = StageRunner(store)
    child_args = [
        f"experiment={args.experiment}",
        f"run_id={args.run_id}",
        "mode=full",
        "scope=full",
        *overrides,
    ]
    commands = [
        (
            "prepare_data",
            [
                str(cfg.paths.python),
                str(ROOT / "scripts/prepare_data.py"),
                *child_args,
            ],
        ),
        (
            f"generate_test_{args.source}",
            [
                str(cfg.paths.python),
                str(ROOT / "scripts/generate_candidates.py"),
                *child_args,
                "split=test",
                f"cg={args.source}",
            ],
        ),
    ]
    try:
        for stage, command in commands:
            if not completed(store, stage):
                runner.run(stage, command, cwd=ROOT)
        store.finalize("completed")
        return 0
    except (OSError, subprocess.CalledProcessError) as error:
        store.finalize("failed", error=type(error).__name__)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
