from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

from omegaconf import DictConfig, OmegaConf


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPOSITORY_ROOT / "configs"
RUN_ID_RE = re.compile(r"^[0-9]{8}_[0-9]{4}_[a-z0-9][a-z0-9_-]{0,47}$")
RESERVED_CLI_KEYS = {"experiment", "run_id", "mode", "scope", "resume"}


def parse_cli_dotlist(values: Iterable[str]) -> tuple[dict[str, str], list[str]]:
    """Split Hydra-like ``key=value`` arguments into runtime keys and overrides."""

    runtime: dict[str, str] = {}
    overrides: list[str] = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected key=value override, got: {value!r}")
        key, raw = value.split("=", 1)
        if not key:
            raise ValueError(f"Empty override key in: {value!r}")
        if key in RESERVED_CLI_KEYS:
            runtime[key] = raw
        else:
            overrides.append(value)
    return runtime, overrides


def validate_run_id(run_id: str) -> None:
    if not RUN_ID_RE.fullmatch(run_id):
        raise ValueError(
            "run_id must match YYYYMMDD_HHMM_<short_name> and contain only "
            f"lowercase letters, digits, '_' or '-': {run_id!r}"
        )


def _load_yaml(path: Path) -> DictConfig:
    if not path.is_file():
        raise FileNotFoundError(path)
    loaded = OmegaConf.load(path)
    if not isinstance(loaded, DictConfig):
        raise TypeError(f"Config root must be a mapping: {path}")
    return loaded


def compose_config(
    experiment: str,
    *,
    run_id: str | None = None,
    mode: str | None = None,
    scope: str | None = None,
    overrides: Iterable[str] = (),
    config_root: Path = CONFIG_ROOT,
) -> DictConfig:
    """Compose and fully resolve one experiment configuration."""

    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", experiment):
        raise ValueError(f"Invalid experiment name: {experiment!r}")
    parts = [
        _load_yaml(config_root / "base.yaml"),
        _load_yaml(config_root / "paths.yaml"),
        _load_yaml(config_root / "splits" / "temporal.yaml"),
        _load_yaml(config_root / "experiments" / f"{experiment}.yaml"),
    ]
    if overrides:
        parts.append(OmegaConf.from_dotlist(list(overrides)))
    cfg = OmegaConf.merge(*parts)
    if run_id is not None:
        validate_run_id(run_id)
        cfg.runtime.run_id = run_id
    if mode is not None:
        cfg.runtime.mode = mode
    if scope is not None:
        cfg.runtime.scope = scope
    OmegaConf.resolve(cfg)
    validate_config(cfg)
    return cfg


def validate_config(cfg: DictConfig) -> None:
    root = Path(str(cfg.paths.root))
    if not root.is_absolute():
        raise ValueError(f"paths.root must be absolute: {root}")
    for key in ("runs", "cache", "python", "val_clicks", "test_clicks", "banner_index"):
        value = Path(str(cfg.paths[key]))
        if not value.is_absolute():
            raise ValueError(f"paths.{key} must be absolute: {value}")
    if str(cfg.runtime.scope) not in {"offline", "full"}:
        raise ValueError(f"runtime.scope must be offline or full: {cfg.runtime.scope}")
    if str(cfg.runtime.mode) not in {"smoke", "offline", "full"}:
        raise ValueError(f"runtime.mode must be smoke, offline or full: {cfg.runtime.mode}")
    if int(cfg.split.fit.end_exclusive) != int(cfg.split.holdout.start_inclusive):
        raise ValueError("Temporal fit/holdout boundaries must be contiguous")
    if int(cfg.split.fit.start_inclusive) >= int(cfg.split.fit.end_exclusive):
        raise ValueError("Temporal fit range is empty")
    if int(cfg.candidates.ranker_pool) <= 0:
        raise ValueError("candidates.ranker_pool must be positive")
    if int(cfg.candidates.union_max_candidates) < int(cfg.candidates.ranker_pool):
        raise ValueError("union_max_candidates must be >= ranker_pool")
    enabled = [
        name
        for name, item in cfg.candidates.generators.items()
        if bool(item.get("enabled", False))
    ]
    if not enabled:
        raise ValueError("At least one candidate generator must be enabled")
    stage_names = [str(stage.name) for stage in cfg.pipeline.stages]
    if len(stage_names) != len(set(stage_names)):
        raise ValueError(f"Pipeline stage names must be unique: {stage_names}")


def to_plain_dict(cfg: DictConfig) -> dict[str, Any]:
    value = OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)
    if not isinstance(value, dict):
        raise TypeError("Resolved config must be a mapping")
    return value


def config_fingerprint(cfg: DictConfig, *, include_runtime: bool = False) -> str:
    """Stable semantic fingerprint; run identity/resume flags are excluded by default."""

    value = to_plain_dict(cfg)
    if not include_runtime:
        runtime = dict(value.get("runtime") or {})
        for key in ("run_id", "resume", "mode"):
            runtime.pop(key, None)
        value["runtime"] = runtime
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

