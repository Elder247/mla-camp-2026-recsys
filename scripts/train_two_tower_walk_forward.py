#!/usr/bin/env python3
from __future__ import annotations

import argparse
import atexit
import json
import logging
import sys
from pathlib import Path

import pyarrow.parquet as pq
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def load_config(path: Path):
    cfg = OmegaConf.load(path)
    parent = cfg.get("extends")
    if parent:
        base = load_config((path.parent / str(parent)).resolve())
        cfg = OmegaConf.merge(base, cfg)
        del cfg["extends"]
    return cfg


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train predict-before-update weekly TwoTower and export OOF artifacts"
    )
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def selected_weeks(cfg: object) -> tuple[list[int], list[dict]]:
    rows = pq.read_table(Path(str(cfg.paths.week_stats_file))).to_pylist()
    rows.sort(key=lambda row: int(row["week_start"]))
    minimum = int(cfg.walk_forward.get("min_week_clicks", 1))
    rows = [row for row in rows if int(row["clicks"]) >= minimum]
    drop_first = int(cfg.walk_forward.get("drop_first_weeks", 0))
    drop_last = int(cfg.walk_forward.get("drop_last_weeks", 0))
    if drop_last:
        rows = rows[drop_first:-drop_last]
    else:
        rows = rows[drop_first:]
    maximum = int(cfg.walk_forward.get("max_weeks", 0))
    if maximum > 0:
        rows = rows[-maximum:]
    weeks = [int(row["week_start"]) for row in rows]
    return weeks, rows


def iter_request_rows(cfg: object):
    import yt.wrapper as yt
    from common.yt_data import make_client

    client = make_client()
    table = str(cfg.paths.weekly_train_table)
    columns = [
        "show_time",
        "uniq_id",
        "query",
        "region_id",
        "crypta_id_v2",
        "device",
        "age",
        "gender",
        "banner_id",
        "source_cost",
    ]
    path = yt.TablePath(table, columns=columns)
    for raw in client.read_table(path, unordered=False, enable_read_parallel=True):
        yield dict(raw)


def main() -> int:
    args = arguments()
    cfg = load_config(args.config.resolve())
    sys.path.insert(0, str(cfg.paths.step2_root))
    from two_tower_v2.data import YtTableSource, YtWeekTableSource
    from two_tower_v2.training import atomic_json, build_model, evaluate, resolve_device
    from two_tower_v2.walk_forward import (
        export_snapshot,
        extract_oof_requests,
        initialize_state,
        load_training_checkpoint,
        save_training_checkpoint,
        train_week,
        validate_week_sequence,
        walk_forward_events,
    )
    from mla_recsys.tracking import UnderdeepTracker, numeric_metrics

    artifact_dir = Path(str(cfg.paths.artifact_dir))
    artifact_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(artifact_dir / "train.log", encoding="utf-8"),
        ],
    )
    resolved = OmegaConf.to_yaml(cfg, resolve=True)
    resolved_path = artifact_dir / "config.resolved.yaml"
    if resolved_path.is_file() and resolved_path.read_text(encoding="utf-8") != resolved:
        raise RuntimeError("Walk-forward artifact config changed; use a new artifact path")
    resolved_path.write_text(resolved, encoding="utf-8")

    tracker = UnderdeepTracker(
        artifact_dir=artifact_dir,
        tracking_cfg=cfg.tracking.underdeep,
        run_name=str(cfg.tracking.underdeep.run_name),
        description=(
            "Predict-before-update weekly TwoTower v2 training for leakage-safe "
            "CatBoost OOF candidate generation"
        ),
        parameters={
            "solution": str(cfg.experiment.name),
            "weekly_train_table": str(cfg.paths.weekly_train_table),
            "requests_per_week": int(cfg.walk_forward.requests_per_week),
            "model": OmegaConf.to_container(cfg.model, resolve=True),
            "training": OmegaConf.to_container(cfg.training, resolve=True),
        },
        tags=["mla-camp", "recsys", "two-tower", "walk-forward", "yt-streaming"],
    )
    atexit.register(tracker.close)

    weeks, week_rows = selected_weeks(cfg)
    weeks = validate_week_sequence(weeks)
    schedule = walk_forward_events(weeks)
    atomic_json(artifact_dir / "schedule.json", {"weeks": week_rows, "events": schedule})

    oof_path = Path(str(cfg.paths.oof_requests_file))
    history_path = Path(str(cfg.paths.history_events_file))
    if not oof_path.is_file() or not history_path.is_file():
        report = extract_oof_requests(
            rows=iter_request_rows(cfg),
            weeks=weeks,
            requests_per_week=int(cfg.walk_forward.requests_per_week),
            output=oof_path,
            history_output=history_path,
        )
        atomic_json(artifact_dir / "oof_requests.json", report)

    device = resolve_device(str(cfg.runtime.device))
    checkpoint_dir = artifact_dir / "checkpoints"
    completed = sorted(checkpoint_dir.glob("after_week_*.pt"))
    if completed:
        model, optimizer, completed_index = load_training_checkpoint(
            completed[-1], cfg=cfg, device=device
        )
        next_index = completed_index + 1
    else:
        model, optimizer = initialize_state(cfg, device=device)
        next_index = 0

    manifest_path = artifact_dir / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file()
        else {
            "version": 1,
            "solution": "two_tower_v2_walk_forward",
            "status": "running",
            "weeks": weeks,
            "events": schedule,
            "snapshots": {},
            "training": {},
            "oof_requests": str(oof_path),
        }
    )
    manifest["status"] = "running"
    atomic_json(manifest_path, manifest)

    for index in range(next_index, len(weeks)):
        start = weeks[index]
        event = schedule[index]
        snapshot_dir = artifact_dir / "snapshots" / str(start)
        lifecycle = {
            **event,
            "trained_through_week_index": index - 1,
            "target_week_seen": False,
        }
        snapshot = export_snapshot(
            cfg=cfg,
            model=model,
            artifact_dir=snapshot_dir,
            device=device,
            lifecycle=lifecycle,
        )
        manifest["snapshots"][str(start)] = {
            "path": str(snapshot_dir),
            "lifecycle": lifecycle,
            "candidates": snapshot["candidates"],
        }
        atomic_json(manifest_path, manifest)

        source = YtWeekTableSource(
            str(cfg.paths.weekly_train_table),
            str(cfg.paths.proxy),
            start=start,
            end=start + 604800,
        )
        train_sample_fraction = float(
            cfg.walk_forward.get("train_sample_fraction", 1.0)
        )
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        metrics = train_week(
            cfg=cfg,
            model=model,
            optimizer=optimizer,
            rows=source.rows(
                sample_fraction=train_sample_fraction,
                sample_seed=int(cfg.training.seed) + index,
            ),
            week_index=index,
            device=device,
            tracker=tracker,
        )
        metrics.update(
            week_start=start,
            week_end_exclusive=start + 604800,
            train_sample_fraction=train_sample_fraction,
        )
        checkpoint = checkpoint_dir / f"after_week_{index:03d}.pt"
        save_training_checkpoint(
            checkpoint,
            cfg=cfg,
            model=model,
            optimizer=optimizer,
            completed_week_index=index,
            metrics=metrics,
        )
        atomic_json(artifact_dir / "metrics" / f"week_{index:03d}.json", metrics)
        manifest["training"][str(start)] = metrics
        atomic_json(manifest_path, manifest)
        tracker.log(
            (index + 1) * 1_000_000,
            numeric_metrics(metrics, prefix="week"),
        )

    final_path_key = cfg.walk_forward.get("final_artifact_path_key")
    final_dir = (
        Path(str(cfg.paths[str(final_path_key)]))
        if final_path_key
        else artifact_dir / "final"
    )
    final_lifecycle = {
        "predict_state": (
            "configured_full_quality_override"
            if final_path_key
            else f"after_week_{len(weeks) - 1}"
        ),
        "trained_through_week_index": (
            None if final_path_key else len(weeks) - 1
        ),
        "trained_on": (
            "full_prevalidation_raw_train"
            if final_path_key
            else "selected_walk_forward_weeks"
        ),
        "target_week_seen": False,
        "purpose": "validation_and_test",
    }
    final_model = model
    final_model_cfg = cfg
    if final_path_key:
        required = (
            final_dir / "model.pt",
            final_dir / "candidate_embeddings.npy",
            final_dir / "candidate_metadata.parquet",
            final_dir / "manifest.json",
        )
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Configured final artifact is incomplete: {missing}")
        checkpoint = torch.load(
            final_dir / "model.pt", map_location=device, weights_only=True
        )
        final_model_cfg = OmegaConf.create(checkpoint["config"])
        final_model = build_model(final_model_cfg).to(device)
        final_model.load_state_dict(checkpoint["state_dict"])
    else:
        export_snapshot(
            cfg=cfg,
            model=model,
            artifact_dir=final_dir,
            device=device,
            lifecycle=final_lifecycle,
        )
    validation = YtTableSource(
        str(cfg.paths.validation_table), str(cfg.paths.proxy)
    )
    health = evaluate(
        final_model,
        list(validation.rows()),
        cfg=final_model_cfg,
        device=device,
    )
    manifest.update(
        status="completed",
        final_artifact=str(final_dir),
        final_artifact_source=(
            "configured_full_quality_override" if final_path_key else "walk_forward_state"
        ),
        final_lifecycle=final_lifecycle,
        validation_health=health,
    )
    atomic_json(manifest_path, manifest)
    atomic_json(artifact_dir / "metrics.json", manifest)
    totals = {
        "training/weeks": float(len(weeks)),
        "training/examples_seen": float(
            sum(int(row["examples_seen"]) for row in manifest["training"].values())
        ),
        "training/seconds": float(
            sum(float(row["seconds"]) for row in manifest["training"].values())
        ),
        **numeric_metrics(health, prefix="validation"),
    }
    tracker.log_summary(totals)
    tracker.close()
    logging.info("walk-forward completed: %s", artifact_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
