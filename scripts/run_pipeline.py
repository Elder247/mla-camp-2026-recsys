#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import subprocess
import sys
import time
from collections import deque
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mla_recsys.artifacts import RunStore  # noqa: E402
from mla_recsys.config import compose_config, parse_cli_dotlist  # noqa: E402
from mla_recsys.stage_runner import StageRunner  # noqa: E402
from mla_recsys.tracking import UnderdeepTracker, numeric_metrics  # noqa: E402


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


def run_resource_aware_candidates(
    cfg: object,
    commands: list[tuple[str, list[str]]],
    run_one: Callable[[str, list[str]], dict[str, object]],
) -> list[dict[str, object]]:
    """Keep candidate slots busy while allowing at most one GPU generator."""

    workers = max(1, int(cfg.pipeline.max_parallel_cg))
    gpu = deque()
    cpu = deque()
    for command in commands:
        source = _candidate_source(command[1])
        resource = str(cfg.candidates.generators[source].get("resource", "cpu"))
        (gpu if resource == "gpu" else cpu).append(command)

    completed: list[dict[str, object]] = []
    active: dict[concurrent.futures.Future, str] = {}
    gpu_active = False
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        while gpu or cpu or active:
            while len(active) < workers:
                selected = None
                resource = "cpu"
                if gpu and not gpu_active:
                    selected = gpu.popleft()
                    resource = "gpu"
                elif cpu:
                    selected = cpu.popleft()
                if selected is None:
                    break
                future = executor.submit(run_one, selected[0], selected[1])
                active[future] = resource
                gpu_active = gpu_active or resource == "gpu"
            if not active:
                continue
            done, _ = concurrent.futures.wait(
                active,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for future in done:
                resource = active.pop(future)
                if resource == "gpu":
                    gpu_active = False
                completed.append(future.result())
    return completed


def enforce_run_budget(*, started: float, max_wall_seconds: int) -> None:
    if max_wall_seconds <= 0:
        return
    elapsed = time.monotonic() - started
    if elapsed >= max_wall_seconds:
        raise TimeoutError(
            f"Pipeline wall-time budget exhausted: {elapsed:.1f}s >= "
            f"{max_wall_seconds}s"
        )


def stage_tracking_metrics(value: dict[str, object]) -> dict[str, float]:
    stage = str(value["stage"])
    metrics = {
        f"stage/{stage}/completed": float(value.get("status") == "completed"),
        f"stage/{stage}/wall_seconds": float(value.get("wall_seconds") or 0.0),
        f"stage/{stage}/peak_rss_mb": float(value.get("peak_rss_bytes") or 0.0)
        / (1024.0 * 1024.0),
    }
    peak_gpu = value.get("peak_gpu_memory_bytes")
    if peak_gpu is not None:
        metrics[f"stage/{stage}/peak_gpu_memory_mb"] = float(peak_gpu) / (
            1024.0 * 1024.0
        )
    return metrics


def final_tracking_metrics(store: RunStore, cfg: object) -> dict[str, float]:
    result = store.read_result()
    metrics = numeric_metrics(result, prefix="run")
    importance_path = store.path / "reports" / "feature_importance.csv"
    if importance_path.is_file():
        limit = int(cfg.tracking.underdeep.get("feature_importance_top_n", 20))
        with importance_path.open(encoding="utf-8", newline="") as source:
            rows = list(csv.DictReader(source))[:limit]
        for rank, row in enumerate(rows, start=1):
            raw_name = row.get("Feature Id") or row.get("Feature") or f"rank_{rank}"
            safe_name = "".join(
                char if char.isalnum() or char in "_-" else "_"
                for char in str(raw_name)
            )
            raw_value = row.get("Importances") or row.get("Importance")
            if raw_value is not None:
                metrics[f"feature_importance/{rank:02d}_{safe_name}"] = float(raw_value)
    return metrics


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
    tracker = UnderdeepTracker(
        artifact_dir=store.path,
        tracking_cfg=cfg.tracking.underdeep,
        run_name=(
            f"{cfg.tracking.underdeep.run_name_prefix}-{cfg.runtime.run_id}"
        ),
        description=(
            "ML Camp two-stage pipeline with temporal validation, natural "
            "candidate pools and SourceCost-aware ranking"
        ),
        parameters={
            "run_id": str(cfg.runtime.run_id),
            "experiment": str(cfg.experiment.name),
            "mode": str(cfg.runtime.mode),
            "scope": str(cfg.runtime.scope),
            "config_sha256": store.config_sha256,
        },
        tags=[
            "mla-camp",
            "recsys",
            "two-stage",
            str(cfg.runtime.mode),
            str(cfg.experiment.name),
        ],
    )
    started = time.monotonic()
    tracking_step = 0
    max_wall_seconds = int(cfg.pipeline.get("max_wall_seconds", 0))
    try:
        groups = execution_groups(cfg, commands)
        group_index = 0
        while group_index < len(groups):
            group = groups[group_index]
            enforce_run_budget(
                started=started,
                max_wall_seconds=max_wall_seconds,
            )
            if (
                int(cfg.pipeline.max_parallel_cg) > 1
                and group
                and group[0][0].startswith("generate_")
            ):
                candidate_block = []
                while (
                    group_index < len(groups)
                    and groups[group_index]
                    and groups[group_index][0][0].startswith("generate_")
                ):
                    candidate_block.extend(groups[group_index])
                    group_index += 1
                pending = pending_commands(
                    store,
                    candidate_block,
                    resume=bool(cfg.runtime.resume),
                )
                values = run_resource_aware_candidates(
                    cfg,
                    pending,
                    lambda stage, command: StageRunner(store).run(
                        stage,
                        command,
                        cwd=ROOT,
                    ),
                )
                for value in values:
                    tracking_step += 1
                    tracker.log(tracking_step, stage_tracking_metrics(value))
                continue
            pending = pending_commands(
                store, group, resume=bool(cfg.runtime.resume)
            )
            if len(pending) == 1:
                stage, command = pending[0]
                value = runner.run(stage, command, cwd=ROOT)
                tracking_step += 1
                tracker.log(tracking_step, stage_tracking_metrics(value))
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
                        value = future.result()
                        tracking_step += 1
                        tracker.log(tracking_step, stage_tracking_metrics(value))
            group_index += 1
    except (OSError, subprocess.CalledProcessError, TimeoutError) as error:
        store.finalize("failed", error=type(error).__name__)
        tracker.log_summary(final_tracking_metrics(store, cfg))
        tracker.close(error=type(error).__name__)
        return 1
    store.finalize("completed")
    tracker.log_summary(final_tracking_metrics(store, cfg))
    tracker.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
