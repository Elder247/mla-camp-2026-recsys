from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from omegaconf import DictConfig

from .artifacts import RunStore
from .config import REPOSITORY_ROOT, compose_config, parse_cli_dotlist


@dataclass(frozen=True)
class StageContext:
    cfg: DictConfig
    store: RunStore
    values: dict[str, str]


def load_stage_context(
    description: str,
    argv: Sequence[str] | None = None,
    *,
    extra_keys: Iterable[str] = (),
) -> StageContext:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("overrides", nargs="*", help="Hydra-style key=value arguments")
    args = parser.parse_args(argv)
    runtime, overrides = parse_cli_dotlist(
        args.overrides,
        extra_runtime_keys=extra_keys,
    )
    for required in ("experiment", "run_id"):
        if required not in runtime:
            parser.error(f"{required}=... is required")
    mode = runtime.get("mode", "offline")
    scope = runtime.get("scope", "full" if mode == "full" else "offline")
    cfg = compose_config(
        runtime["experiment"],
        run_id=runtime["run_id"],
        mode=mode,
        scope=scope,
        overrides=overrides,
    )
    for path in (Path(str(cfg.paths.root)), Path(str(cfg.paths.step2_root))):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
    store = RunStore.initialize(
        cfg,
        repo_root=REPOSITORY_ROOT,
        resume=bool(cfg.runtime.resume),
    )
    return StageContext(cfg=cfg, store=store, values=runtime)


def require_choice(context: StageContext, key: str, choices: Iterable[str]) -> str:
    allowed = set(choices)
    value = context.values.get(key)
    if value not in allowed:
        raise ValueError(f"{key} must be one of {sorted(allowed)}, got {value!r}")
    return str(value)


def run_data_dir(context: StageContext) -> Path:
    path = context.store.path / "data"
    path.mkdir(parents=True, exist_ok=True)
    return path
