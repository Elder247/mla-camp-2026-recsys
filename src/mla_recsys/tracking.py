from __future__ import annotations

import json
import logging
import os
import socket
from pathlib import Path
from typing import Any, Mapping

from omegaconf import DictConfig, OmegaConf

from .artifacts import atomic_write_json, mask_secrets, utc_now


LOGGER = logging.getLogger(__name__)


def underdeep_token_exists() -> bool:
    """Check credentials without ever opening or rendering the token."""

    return bool(
        os.environ.get("UNDERDEEP_TOKEN")
        or (Path.home() / ".underdeep" / "token").is_file()
    )


def _plain(value: Any) -> Any:
    if isinstance(value, DictConfig):
        return OmegaConf.to_container(value, resolve=True, enum_to_str=True)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def numeric_metrics(value: Any, *, prefix: str = "") -> dict[str, float]:
    """Flatten numeric run results to stable UnderDeep metric names."""

    output: dict[str, float] = {}
    if isinstance(value, Mapping):
        for key, item in value.items():
            name = f"{prefix}/{key}" if prefix else str(key)
            output.update(numeric_metrics(item, prefix=name))
    elif isinstance(value, bool):
        output[prefix] = float(value)
    elif isinstance(value, (int, float)) and prefix:
        output[prefix] = float(value)
    return output


class UnderdeepTracker:
    """Fail-open UnderDeep tracker with an always-on local JSONL backup."""

    def __init__(
        self,
        *,
        artifact_dir: Path,
        tracking_cfg: Mapping[str, Any] | DictConfig,
        run_name: str,
        description: str,
        parameters: Mapping[str, Any],
        tags: list[str],
    ) -> None:
        self.artifact_dir = artifact_dir
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.backup_path = artifact_dir / "underdeep_metrics.jsonl"
        self.run = None
        self.run_uid: str | None = None
        self.url: str | None = None
        self._closed = False
        cfg = _plain(tracking_cfg) or {}
        self.finish_timeout = int(cfg.get("finish_timeout_seconds", 30))
        self.enabled = bool(cfg.get("enabled", False))
        project = str(cfg.get("project") or "")
        experiment = str(cfg.get("experiment") or "")
        self._append_local(
            "init",
            {
                "run_name": run_name,
                "project": project,
                "experiment": experiment,
                "parameters": _plain(parameters),
                "tags": tags,
            },
        )
        if not self.enabled:
            LOGGER.info("UnderDeep is disabled by config; local backup remains enabled")
            return
        if not project or not experiment:
            LOGGER.warning("UnderDeep project/experiment is not configured")
            return
        if not underdeep_token_exists():
            LOGGER.warning("UnderDeep token is unavailable; local backup remains enabled")
            return
        try:
            import underdeep as U
            from underdeep.common.enums import ERunErrorPolicy

            client = U.Client(project=project, experiment=experiment)
            safe_parameters = {
                **_plain(parameters),
                "host": socket.gethostname(),
                "artifact_dir": str(artifact_dir),
            }
            self.run = client.init_run(
                name=run_name,
                description=description,
                parameters=safe_parameters,
                tags=tags,
                file_path=str(artifact_dir / "underdeep_metrics.buffer"),
                local_backup_path=str(self.backup_path),
                error_policy=ERunErrorPolicy.StopSendingData,
            )
            self.run_uid = str(self.run.uid)
            self.url = (
                getattr(self.run, "link", None)
                or getattr(self.run, "experiment_link", None)
            )
            info = {
                "version": 1,
                "project": project,
                "experiment": experiment,
                "run_uid": self.run_uid,
                "url": self.url,
                "started_at": utc_now(),
            }
            atomic_write_json(artifact_dir / "underdeep_run.json", info)
            LOGGER.info("UnderDeep run: %s", self.url)
        except Exception as error:  # tracker must never stop model computation
            LOGGER.warning("UnderDeep init failed (%s); using local backup", type(error).__name__)
            self.run = None

    def _append_local(self, event: str, payload: Mapping[str, Any]) -> None:
        value = {
            "created_at": utc_now(),
            "event": event,
            "payload": _plain(payload),
        }
        with self.backup_path.open("a", encoding="utf-8") as target:
            target.write(mask_secrets(json.dumps(value, ensure_ascii=False)) + "\n")

    def log(self, step: int, metrics: Mapping[str, float]) -> None:
        clean = {str(name): float(value) for name, value in metrics.items()}
        self._append_local("metrics", {"step": int(step), "metrics": clean})
        if self.run is None:
            return
        try:
            self.run.log(clean, step=int(step))
        except Exception as error:
            LOGGER.warning("UnderDeep live log failed: %s", type(error).__name__)

    def log_summary(self, metrics: Mapping[str, float]) -> None:
        clean = {str(name): float(value) for name, value in metrics.items()}
        self._append_local("summary", {"metrics": clean})
        if self.run is None:
            return
        try:
            self.run.log_summary(
                [{"name": name, "value": value} for name, value in clean.items()]
            )
        except Exception as error:
            LOGGER.warning("UnderDeep summary log failed: %s", type(error).__name__)

    def close(self, *, error: str | None = None) -> None:
        if self._closed:
            return
        self._closed = True
        self._append_local("finish", {"error": error})
        if self.run is None:
            return
        try:
            self.run.finish(error=error, timeout=self.finish_timeout)
        except Exception as finish_error:
            LOGGER.warning(
                "UnderDeep finish failed: %s", type(finish_error).__name__
            )
