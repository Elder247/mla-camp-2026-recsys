#!/usr/bin/env python3
"""Create a ranker-only run by safely reusing immutable upstream artifacts."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mla_recsys.artifacts import RunStore, atomic_write_json, utc_now  # noqa: E402
from mla_recsys.config import compose_config, parse_cli_dotlist, to_plain_dict  # noqa: E402


UPSTREAM_DIRECTORIES = ("data", "counters", "candidates", "features")
UPSTREAM_METRIC_PREFIXES = ("generate_", "merge_", "features_")
UPSTREAM_METRIC_NAMES = {"data.json", "cache_parity.json"}
DOWNSTREAM_STAGES = {"train_ranker", "evaluate_run", "make_submission"}


def ranker_probe_semantics(cfg: object) -> dict:
    """Return the config projection that determines reusable feature rows."""
    value = to_plain_dict(cfg)
    value.pop("runtime", None)
    value.pop("ranker", None)
    # Submission ranking is produced after the reused feature contract.
    value.pop("submission", None)
    value.pop("promotion_gate", None)
    candidates = dict(value.get("candidates") or {})
    candidates.pop("reuse_run", None)
    value["candidates"] = candidates
    features = dict(value.get("features") or {})
    features.pop("reuse_run", None)
    features.pop("reuse_history_patch", None)
    value["features"] = features
    return value


def validate_donor(donor: Path, cfg: object) -> dict:
    config_path = donor / "config.yaml"
    result_path = donor / "result.json"
    if not config_path.is_file() or not result_path.is_file():
        raise FileNotFoundError(f"Donor run contract is incomplete: {donor}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") != "completed":
        raise ValueError(f"Donor run is not completed: {donor}")
    donor_cfg = OmegaConf.load(config_path)
    if ranker_probe_semantics(donor_cfg) != ranker_probe_semantics(cfg):
        raise ValueError("Donor upstream config differs from the ranker probe")
    for relative in UPSTREAM_DIRECTORIES:
        if not (donor / relative).is_dir():
            raise FileNotFoundError(f"Donor directory is absent: {donor / relative}")
    parity = donor / "metrics" / "cache_parity.json"
    if not parity.is_file() or not json.loads(parity.read_text(encoding="utf-8")).get("ok"):
        raise ValueError("Donor cache parity is absent or failed")
    return result


def materialize_tree(
    source: Path,
    target: Path,
    *,
    excluded_directory_names: frozenset[str] = frozenset(),
) -> tuple[int, int]:
    files = 0
    bytes_total = 0
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if excluded_directory_names.intersection(relative.parts):
            continue
        output = target / relative
        if path.is_dir():
            output.mkdir(parents=True, exist_ok=True)
            continue
        if not path.is_file():
            continue
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise FileExistsError(f"Ranker probe target already exists: {output}")
        try:
            os.link(path, output)
        except OSError:
            shutil.copy2(path, output)
        files += 1
        bytes_total += path.stat().st_size
    return files, bytes_total


def reusable_stage(stage: str, profile: str) -> bool:
    if profile == "ranker":
        return stage not in DOWNSTREAM_STAGES
    if profile == "history_features":
        return stage in {"prepare_data", "prepare_counters"} or stage.startswith(
            "generate_"
        )
    raise ValueError(f"Unknown reuse profile: {profile}")


def reusable_metric(name: str, profile: str) -> bool:
    if profile == "ranker":
        return name in UPSTREAM_METRIC_NAMES or name.startswith(
            UPSTREAM_METRIC_PREFIXES
        )
    if profile == "history_features":
        return name == "data.json" or name.startswith("generate_")
    raise ValueError(f"Unknown reuse profile: {profile}")


def materialize_ranker_probe(*, donor: Path, cfg: object, profile: str) -> dict:
    donor_result = validate_donor(donor, cfg)
    store = RunStore.initialize(cfg, repo_root=ROOT, resume=False)
    files = 0
    bytes_total = 0
    directories = (
        UPSTREAM_DIRECTORIES
        if profile == "ranker"
        else ("data", "counters", "candidates")
    )
    for relative in directories:
        excluded = frozenset({"merged"}) if relative == "candidates" and profile == "history_features" else frozenset()
        count, size = materialize_tree(
            donor / relative,
            store.path / relative,
            excluded_directory_names=excluded,
        )
        files += count
        bytes_total += size

    donor_metrics = donor / "metrics"
    for path in sorted(donor_metrics.glob("*.json")):
        if not reusable_metric(path.name, profile):
            continue
        target = store.path / "metrics" / path.name
        try:
            os.link(path, target)
        except OSError:
            shutil.copy2(path, target)
        files += 1
        bytes_total += path.stat().st_size

    reused_stages = []
    for stage_path in sorted((donor / "stages").glob("*.json")):
        stage = stage_path.stem
        if not reusable_stage(stage, profile):
            continue
        previous = json.loads(stage_path.read_text(encoding="utf-8"))
        if previous.get("status") != "completed":
            raise ValueError(f"Donor upstream stage is incomplete: {stage}")
        value = {
            "stage": stage,
            "status": "completed",
            "started_at": utc_now(),
            "finished_at": utc_now(),
            "wall_seconds": 0.0,
            "peak_rss_bytes": 0,
            "peak_rss_measurement": "cross_run_reuse",
            "peak_gpu_memory_bytes": None,
            "return_code": 0,
            "command": [],
            "log": None,
            "reused_from": str(donor),
            "donor_stage": previous,
        }
        store.record_stage(stage, value)
        reused_stages.append(stage)

    report = {
        "status": "completed",
        "created_at": utc_now(),
        "donor": str(donor),
        "donor_config_sha256": donor_result.get("config_sha256"),
        "files": files,
        "logical_bytes": bytes_total,
        "materialization": "hardlink_with_copy_fallback",
        "profile": profile,
        "directories": list(directories),
        "reused_stages": reused_stages,
    }
    atomic_write_json(store.path / "metrics" / "ranker_probe_reuse.json", report)
    print(json.dumps(report, indent=2))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Materialize a validated ranker-only run from a completed donor"
    )
    parser.add_argument("--donor", required=True, type=Path)
    parser.add_argument(
        "--profile",
        choices=("ranker", "history_features"),
        default="ranker",
    )
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()
    runtime, overrides = parse_cli_dotlist(args.overrides)
    for key in ("experiment", "run_id"):
        if key not in runtime:
            parser.error(f"{key}=... is required")
    mode = runtime.get("mode", "offline")
    scope = runtime.get("scope", "full" if mode == "full" else "offline")
    cfg = compose_config(
        runtime["experiment"],
        run_id=runtime["run_id"],
        mode=mode,
        scope=scope,
        overrides=overrides,
    )
    materialize_ranker_probe(
        donor=args.donor.resolve(), cfg=cfg, profile=args.profile
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
