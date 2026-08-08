#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mla_recsys.artifacts import RunStore  # noqa: E402
from mla_recsys.config import compose_config, parse_cli_dotlist  # noqa: E402
from mla_recsys.stage_runner import StageRunner  # noqa: E402


def stage_commands(
    cfg: object, child_overrides: list[str] | None = None
) -> list[tuple[str, list[str]]]:
    commands = []
    for stage in cfg.pipeline.stages:
        if str(cfg.runtime.mode) not in [str(value) for value in stage.modes]:
            continue
        if (
            str(cfg.runtime.mode) == "full"
            and str(cfg.submission.ranking) == "rrf"
            and str(stage.name)
            in {"prepare_counters", "build_features", "train_ranker"}
        ):
            continue
        script = ROOT / "scripts" / str(stage.script)
        base_command = [
            str(cfg.paths.python),
            str(script),
            f"experiment={cfg.experiment.name}",
            f"run_id={cfg.runtime.run_id}",
            f"mode={cfg.runtime.mode}",
            f"scope={cfg.runtime.scope}",
            *(child_overrides or []),
        ]
        name = str(stage.name)
        if name == "generate_candidates":
            splits = ["full_train", "test"] if str(cfg.runtime.mode) == "full" else ["train", "holdout"]
            for split in splits:
                for cg, item in cfg.candidates.generators.items():
                    if bool(item.get("enabled", False)):
                        commands.append(
                            (f"generate_{split}_{cg}", [*base_command, f"split={split}", f"cg={cg}"])
                        )
        elif name in {"merge_candidates", "build_features"}:
            splits = ["full_train", "test"] if str(cfg.runtime.mode) == "full" else ["train", "holdout"]
            for split in splits:
                commands.append((f"{name}_{split}", [*base_command, f"split={split}"]))
        else:
            commands.append((name, base_command))
    return commands


def _candidate_source(command: list[str]) -> str:
    return next(value.split("=", 1)[1] for value in command if value.startswith("cg="))


def execution_groups(
    cfg: object,
    commands: list[tuple[str, list[str]]],
) -> list[list[tuple[str, list[str]]]]:
    """Group only independent stages; never overlap two GPU generators."""

    workers = max(1, int(cfg.pipeline.max_parallel_cg))
    if workers == 1:
        return [[command] for command in commands]
    groups: list[list[tuple[str, list[str]]]] = []
    index = 0
    while index < len(commands):
        stage = commands[index][0]
        if stage.startswith("generate_"):
            end = index
            while end < len(commands) and commands[end][0].startswith("generate_"):
                end += 1
            gpu: list[tuple[str, list[str]]] = []
            cpu: list[tuple[str, list[str]]] = []
            for command in commands[index:end]:
                source = _candidate_source(command[1])
                resource = str(cfg.candidates.generators[source].get("resource", "cpu"))
                (gpu if resource == "gpu" else cpu).append(command)
            cpu.sort(
                key=lambda command: -int(
                    cfg.candidates.generators[_candidate_source(command[1])].get(
                        "parallel_priority", 0
                    )
                )
            )
            while gpu or cpu:
                group: list[tuple[str, list[str]]] = []
                if gpu:
                    group.append(gpu.pop(0))
                count = min(workers - len(group), len(cpu))
                group.extend(cpu[:count])
                del cpu[:count]
                groups.append(group)
            index = end
            continue
        if stage.startswith("merge_candidates_"):
            group = []
            while index < len(commands) and commands[index][0].startswith(
                "merge_candidates_"
            ):
                group.append(commands[index])
                index += 1
            groups.extend(
                group[offset : offset + workers]
                for offset in range(0, len(group), workers)
            )
            continue
        if stage.startswith("build_features_"):
            group = []
            while index < len(commands) and commands[index][0].startswith(
                "build_features_"
            ):
                group.append(commands[index])
                index += 1
            groups.extend(
                group[offset : offset + workers]
                for offset in range(0, len(group), workers)
            )
            continue
        groups.append([commands[index]])
        index += 1
    return groups


def pending_commands(
    store: RunStore,
    group: list[tuple[str, list[str]]],
    *,
    resume: bool,
) -> list[tuple[str, list[str]]]:
    pending = []
    for stage, command in group:
        previous = store.path / "stages" / f"{stage}.json"
        if previous.is_file() and resume:
            value = json.loads(previous.read_text(encoding="utf-8"))
            if value.get("status") == "completed":
                print(f"resume: skip completed stage {stage}")
                continue
        pending.append((stage, command))
    return pending


def enforce_run_budget(*, started: float, max_wall_seconds: int) -> None:
    if max_wall_seconds <= 0:
        return
    elapsed = time.monotonic() - started
    if elapsed >= max_wall_seconds:
        raise TimeoutError(
            f"Pipeline wall-time budget exhausted: {elapsed:.1f}s >= "
            f"{max_wall_seconds}s"
        )


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
    commands = stage_commands(cfg, overrides)
    if args.dry_run:
        print(json.dumps({"run_id": cfg.runtime.run_id, "commands": commands}, indent=2))
        return 0

    store = RunStore.initialize(cfg, repo_root=ROOT, resume=not args.no_resume)
    runner = StageRunner(store)
    started = time.monotonic()
    max_wall_seconds = int(cfg.pipeline.get("max_wall_seconds", 0))
    try:
        for group in execution_groups(cfg, commands):
            enforce_run_budget(
                started=started,
                max_wall_seconds=max_wall_seconds,
            )
            pending = pending_commands(
                store, group, resume=bool(cfg.runtime.resume)
            )
            if len(pending) == 1:
                stage, command = pending[0]
                runner.run(stage, command, cwd=ROOT)
            elif pending:
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=len(pending)
                ) as executor:
                    futures = [
                        executor.submit(
                            StageRunner(store).run,
                            stage,
                            command,
                            cwd=ROOT,
                        )
                        for stage, command in pending
                    ]
                    for future in futures:
                        future.result()
    except (OSError, subprocess.CalledProcessError, TimeoutError) as error:
        store.finalize("failed", error=type(error).__name__)
        return 1
    store.finalize("completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
