from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from omegaconf import DictConfig, OmegaConf

from .config import config_fingerprint, to_plain_dict, validate_run_id


TOKEN_PATTERN = re.compile(r"(?:y[01]_|t[01]_|AQAD-)[A-Za-z0-9_\-]+")
MANIFEST_SUFFIX = ".manifest.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def mask_secrets(value: object) -> str:
    return TOKEN_PATTERN.sub("***", str(value))


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as target:
            target.write(text)
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2) + "\n")


@contextmanager
def atomic_output_path(path: Path):
    """Yield a same-directory temporary path and atomically replace on success."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    temp_path = Path(temporary)
    try:
        yield temp_path
        os.replace(temp_path, path)
    except BaseException:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass
        raise


def fingerprint_file(path: Path, *, sample_bytes: int = 1 << 20) -> dict[str, Any]:
    """Cheap deterministic identity using stat plus full/sampled SHA-256."""

    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        return {"path": str(resolved), "exists": False}
    stat = resolved.stat()
    digest = hashlib.sha256()
    with resolved.open("rb") as source:
        if stat.st_size <= sample_bytes * 2:
            digest.update(source.read())
            mode = "full_sha256"
        else:
            digest.update(source.read(sample_bytes))
            source.seek(-sample_bytes, os.SEEK_END)
            digest.update(source.read(sample_bytes))
            mode = "head_tail_sha256"
    return {
        "path": str(resolved),
        "exists": True,
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "hash_mode": mode,
        "sha256": digest.hexdigest(),
    }


def content_fingerprint(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _git(repo_root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_root), *args],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _probe_gpu(python_executable: Path, probe: str) -> dict[str, Any]:
    try:
        raw = subprocess.run(
            [str(python_executable), "-c", probe],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=30,
        ).stdout
        value = json.loads(raw)
        value["python_executable"] = str(python_executable)
        return value
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as error:
        return {
            "cuda_available": False,
            "gpus": [],
            "python_executable": str(python_executable),
            "probe_error": type(error).__name__,
        }


def collect_environment(repo_root: Path, python_executable: Path) -> dict[str, Any]:
    package_names = [
        "catboost",
        "hydra-core",
        "numpy",
        "omegaconf",
        "pandas",
        "polars",
        "pyarrow",
        "torch",
        "yql",
    ]
    packages: dict[str, str] = {}
    for name in package_names:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "missing-in-orchestrator"

    probe = (
        "import json\n"
        "try:\n"
        " import torch\n"
        " g=[{'index':i,'name':torch.cuda.get_device_name(i),"
        "'total_memory':torch.cuda.get_device_properties(i).total_memory} "
        "for i in range(torch.cuda.device_count())]\n"
        " print(json.dumps({'cuda_available':torch.cuda.is_available(),'gpus':g}))\n"
        "except Exception as e:\n"
        " print(json.dumps({'cuda_available':False,'gpus':[],'error':type(e).__name__}))\n"
    )
    gpu = _probe_gpu(python_executable, probe)
    # The shared model venv intentionally has a CPU torch wheel while CatBoost
    # uses the VM CUDA runtime. Probe the system interpreter as a read-only
    # fallback so the manifest still records the actual accelerator.
    system_python = shutil.which("python3")
    if not gpu.get("cuda_available") and system_python:
        fallback = _probe_gpu(Path(system_python), probe)
        if fallback.get("cuda_available"):
            gpu = fallback

    dirty = _git(repo_root, "status", "--porcelain")
    return {
        "created_at": utc_now(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python_executable": str(python_executable),
        "git": {
            "sha": _git(repo_root, "rev-parse", "HEAD"),
            "branch": _git(repo_root, "branch", "--show-current"),
            "dirty": bool(dirty) if dirty is not None else None,
        },
        "packages": packages,
        "compute": gpu,
    }


def output_manifest_path(output: Path) -> Path:
    return output.with_name(output.name + MANIFEST_SUFFIX)


def make_cache_key(
    *,
    stage: str,
    artifact_version: str,
    config_sha256: str,
    inputs: Iterable[dict[str, Any]],
) -> str:
    return content_fingerprint(
        {
            "stage": stage,
            "artifact_version": artifact_version,
            "config_sha256": config_sha256,
            "inputs": list(inputs),
        }
    )


def write_output_manifest(
    output: Path,
    *,
    stage: str,
    artifact_version: str,
    config_sha256: str,
    inputs: Iterable[dict[str, Any]],
    rows: int | None = None,
    schema: Any = None,
    scope: str | None = None,
) -> dict[str, Any]:
    input_list = list(inputs)
    value = {
        "version": 1,
        "stage": stage,
        "artifact_version": artifact_version,
        "scope": scope,
        "created_at": utc_now(),
        "cache_key": make_cache_key(
            stage=stage,
            artifact_version=artifact_version,
            config_sha256=config_sha256,
            inputs=input_list,
        ),
        "config_sha256": config_sha256,
        "inputs": input_list,
        "output": fingerprint_file(output),
        "rows": rows,
        "schema": schema,
    }
    atomic_write_json(output_manifest_path(output), value)
    return value


def validate_output_cache(
    output: Path,
    *,
    expected_cache_key: str,
    expected_schema: Any = None,
    expected_rows: int | None = None,
) -> tuple[bool, str]:
    manifest_path = output_manifest_path(output)
    if not output.is_file():
        return False, "output_missing"
    if not manifest_path.is_file():
        return False, "manifest_missing"
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False, "manifest_invalid"
    if value.get("cache_key") != expected_cache_key:
        return False, "cache_key_mismatch"
    if expected_schema is not None and value.get("schema") != expected_schema:
        return False, "schema_mismatch"
    if expected_rows is not None and value.get("rows") != expected_rows:
        return False, "row_count_mismatch"
    current = fingerprint_file(output)
    recorded = value.get("output") or {}
    for key in ("size_bytes", "sha256"):
        if current.get(key) != recorded.get(key):
            return False, f"output_{key}_mismatch"
    return True, "valid"


@dataclass
class RunStore:
    root: Path
    run_id: str
    config_sha256: str

    @property
    def path(self) -> Path:
        return self.root / self.run_id

    @classmethod
    def initialize(
        cls,
        cfg: DictConfig,
        *,
        repo_root: Path,
        resume: bool = True,
    ) -> "RunStore":
        run_id = str(cfg.runtime.run_id or "")
        validate_run_id(run_id)
        store = cls(Path(str(cfg.paths.runs)), run_id, config_fingerprint(cfg))
        config_path = store.path / "config.yaml"
        manifest_path = store.path / "manifest.json"
        if config_path.exists():
            if not resume:
                raise FileExistsError(f"Run already exists: {store.path}")
            if not manifest_path.is_file():
                raise RuntimeError(f"Existing run has no manifest: {store.path}")
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            if previous.get("config_sha256") != store.config_sha256:
                raise RuntimeError(
                    f"Run config fingerprint mismatch for {store.run_id}: "
                    f"{previous.get('config_sha256')} != {store.config_sha256}"
                )
            return store

        for relative in (
            "logs",
            "stages",
            "metrics",
            "candidates",
            "features",
            "models",
            "reports",
            "predictions",
        ):
            (store.path / relative).mkdir(parents=True, exist_ok=True)
        atomic_write_text(config_path, OmegaConf.to_yaml(cfg, resolve=True))
        input_keys = ("val_clicks", "test_clicks", "banner_index", "token_stats")
        inputs = [fingerprint_file(Path(str(cfg.paths[key]))) for key in input_keys]
        manifest = collect_environment(repo_root, Path(str(cfg.paths.python)))
        manifest.update(
            {
                "version": 1,
                "run_id": run_id,
                "experiment": str(cfg.experiment.name),
                "scope": str(cfg.runtime.scope),
                "config_sha256": store.config_sha256,
                "resolved_config_sha256": config_fingerprint(cfg, include_runtime=True),
                "inputs": inputs,
            }
        )
        atomic_write_json(manifest_path, manifest)
        atomic_write_json(
            store.path / "result.json",
            {
                "version": 1,
                "run_id": run_id,
                "status": "running",
                "started_at": utc_now(),
                "config_sha256": store.config_sha256,
                "stages": {},
            },
        )
        return store

    def read_result(self) -> dict[str, Any]:
        return json.loads((self.path / "result.json").read_text(encoding="utf-8"))

    def update_result(self, **updates: Any) -> dict[str, Any]:
        value = self.read_result()
        value.update(updates)
        atomic_write_json(self.path / "result.json", value)
        return value

    def record_stage(self, stage: str, value: dict[str, Any]) -> None:
        atomic_write_json(self.path / "stages" / f"{stage}.json", value)
        result = self.read_result()
        result.setdefault("stages", {})[stage] = value
        atomic_write_json(self.path / "result.json", result)
        timing_path = self.path / "reports" / "timing.csv"
        rows: list[dict[str, Any]] = []
        if timing_path.is_file():
            with timing_path.open(encoding="utf-8", newline="") as source:
                rows = list(csv.DictReader(source))
        rows = [row for row in rows if row.get("stage") != stage]
        rows.append(
            {
                "stage": stage,
                "status": value.get("status"),
                "wall_seconds": value.get("wall_seconds"),
                "peak_rss_bytes": value.get("peak_rss_bytes"),
            }
        )
        timing_path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=".timing.", dir=timing_path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="") as target:
                writer = csv.DictWriter(
                    target,
                    fieldnames=["stage", "status", "wall_seconds", "peak_rss_bytes"],
                )
                writer.writeheader()
                writer.writerows(rows)
                target.flush()
                os.fsync(target.fileno())
            os.replace(temporary, timing_path)
        except BaseException:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise

    def finalize(self, status: str, **updates: Any) -> dict[str, Any]:
        if status not in {"completed", "failed"}:
            raise ValueError(status)
        return self.update_result(status=status, finished_at=utc_now(), **updates)
