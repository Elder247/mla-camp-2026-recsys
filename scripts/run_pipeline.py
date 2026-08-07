#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mla_recsys.artifacts import RunStore  # noqa: E402
from mla_recsys.config import compose_config, parse_cli_dotlist  # noqa: E402
from mla_recsys.stage_runner import StageRunner  # noqa: E402


def stage_commands(cfg: object) -> list[tuple[str, list[str]]]:
    commands = []
    for stage in cfg.pipeline.stages:
        if str(cfg.runtime.mode) not in [str(value) for value in stage.modes]:
            continue
        script = ROOT / "scripts" / str(stage.script)
        command = [
            str(cfg.paths.python),
            str(script),
            f"experiment={cfg.experiment.name}",
            f"run_id={cfg.runtime.run_id}",
            f"mode={cfg.runtime.mode}",
            f"scope={cfg.runtime.scope}",
        ]
        commands.append((str(stage.name), command))
    return commands


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the configured ML Camp pipeline as isolated subprocess stages",
        epilog=(
            "Example: run_pipeline.py experiment=i0_reproduce "
            "run_id=20260807_2200_i0 mode=smoke"
        ),
    )
    parser.add_argument("overrides", nargs="*", help="Hydra-style key=value overrides")
    parser.add_argument("--dry-run", action="store_true", help="Resolve and print commands only")
    parser.add_argument("--no-resume", action="store_true", help="Fail if run_id already exists")
    args = parser.parse_args()

    runtime, overrides = parse_cli_dotlist(args.overrides)
    if "experiment" not in runtime:
        parser.error("experiment=<name> is required")
    if "run_id" not in runtime:
        parser.error("run_id=<YYYYMMDD_HHMM_name> is required")
    mode = runtime.get("mode", "offline")
    scope = runtime.get("scope", "full" if mode == "full" else "offline")
    cfg = compose_config(
        runtime["experiment"],
        run_id=runtime["run_id"],
        mode=mode,
        scope=scope,
        overrides=overrides,
    )
    commands = stage_commands(cfg)
    if args.dry_run:
        print(json.dumps({"run_id": cfg.runtime.run_id, "commands": commands}, indent=2))
        return 0

    store = RunStore.initialize(cfg, repo_root=ROOT, resume=not args.no_resume)
    runner = StageRunner(store)
    try:
        for stage, command in commands:
            previous = store.path / "stages" / f"{stage}.json"
            if previous.is_file() and bool(cfg.runtime.resume):
                value = json.loads(previous.read_text(encoding="utf-8"))
                if value.get("status") == "completed":
                    print(f"resume: skip completed stage {stage}")
                    continue
            runner.run(stage, command, cwd=ROOT)
    except (OSError, subprocess.CalledProcessError) as error:
        store.finalize("failed", error=type(error).__name__)
        return 1
    store.finalize("completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

