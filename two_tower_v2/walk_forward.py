from __future__ import annotations

import hashlib
import heapq
import json
import logging
import os
import random
import resource
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from omegaconf import DictConfig, OmegaConf
from torch.nn import functional as F

from mla_recsys.counters import (
    COUNTER_EVENT_SCHEMA,
    scalar_key,
    stable_text_key,
    week_start,
)
from mla_recsys.data import write_request_parquet
from two_tower_v2.data import (
    YtWeekTableSource,
    batches,
    enrich_rows,
    pack_bags,
    prefetch_batches,
    shuffled_rows,
)
from two_tower_v2.training import (
    MODEL_FILENAME,
    all_cardinalities,
    atomic_json,
    bpe_limits,
    build_model,
    copy_tokenizer_artifact,
    export_candidates,
    load_bpe_tokenizer,
    numeric_feature_scale,
    retrieval_objective,
)


LOGGER = logging.getLogger(__name__)


def validate_week_sequence(values: Sequence[int]) -> list[int]:
    weeks = [int(value) for value in values]
    if not weeks:
        raise ValueError("walk-forward requires at least one week")
    if weeks != sorted(set(weeks)):
        raise ValueError("walk-forward weeks must be unique and increasing")
    if any(week_start(value) != value for value in weeks):
        raise ValueError("walk-forward boundaries must be Monday 00:00 UTC")
    if any(right - left != 604800 for left, right in zip(weeks, weeks[1:])):
        raise ValueError("walk-forward weeks must be contiguous")
    return weeks


def walk_forward_events(weeks: Sequence[int]) -> list[dict[str, Any]]:
    """Return the immutable predict-before-update lifecycle contract."""

    output = []
    for index, start in enumerate(validate_week_sequence(weeks)):
        output.append(
            {
                "week_index": index,
                "week_start": start,
                "week_end_exclusive": start + 604800,
                "predict_state": "random" if index == 0 else f"after_week_{index - 1}",
                "update_state": f"after_week_{index}",
                "order": ["predict", "freeze_pool", "attach_labels", "update"],
            }
        )
    return output


def initialize_state(
    cfg: DictConfig,
    *,
    device: torch.device,
) -> tuple[torch.nn.Module, torch.optim.Optimizer]:
    seed = int(cfg.training.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = build_model(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.training.learning_rate),
        weight_decay=float(cfg.training.weight_decay),
    )
    return model, optimizer


def train_week(
    *,
    cfg: DictConfig,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    rows: Iterable[dict[str, list[int]]],
    week_index: int,
    device: torch.device,
    tracker: Any | None = None,
) -> dict[str, Any]:
    model.train()
    cardinalities = all_cardinalities(cfg)
    tokenizer = load_bpe_tokenizer(cfg)
    limits = bpe_limits(cfg)
    objective = str(cfg.training.get("objective", "cross_entropy"))
    symmetric_weight = float(cfg.training.get("symmetric_weight", 0.0))
    started = time.perf_counter()
    examples_seen = 0
    steps = 0
    data_wait_seconds = 0.0
    train_seconds = 0.0
    last_loss = 0.0
    last_accuracy = 0.0
    iterator = iter(
        prefetch_batches(
            batches(
                shuffled_rows(
                    rows,
                    buffer_size=int(cfg.training.shuffle_buffer),
                    seed=int(cfg.training.seed) + week_index,
                ),
                int(cfg.training.batch_size),
            ),
            int(cfg.training.get("prefetch_batches", 0)),
        )
    )
    max_examples = int(cfg.walk_forward.get("max_examples_per_week", 0))
    while True:
        wait_started = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            break
        data_wait_seconds += time.perf_counter() - wait_started
        if max_examples > 0:
            remaining = max_examples - examples_seen
            if remaining <= 0:
                break
            batch = batch[:remaining]
        train_started = time.perf_counter()
        batch = enrich_rows(
            batch,
            cardinalities=cardinalities,
            tokenizer=tokenizer,
            bpe_limits=limits,
            source_cost_log1p_scale=float(
                cfg.get("numeric_features", {}).get("source_cost_log1p_scale", 1.0)
            ),
            product_price_log1p_scale=numeric_feature_scale(
                cfg, "product_price_log1p_scale"
            ),
        )
        bags = pack_bags(batch, cardinalities=cardinalities, device=device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            query = model.encode_query(bags)
            banner = model.encode_banner(bags)
            logits = query @ banner.T / float(cfg.training.temperature)
            loss, positive = retrieval_objective(
                logits,
                batch,
                objective=objective,
                symmetric_weight=symmetric_weight,
                sourcecost_weight_power=float(
                    cfg.training.get("sourcecost_weight_power", 0.0)
                ),
                sourcecost_weight_min=float(
                    cfg.training.get("sourcecost_weight_min", 1.0)
                ),
                sourcecost_weight_max=float(
                    cfg.training.get("sourcecost_weight_max", 1.0)
                ),
                logq_correction=str(
                    cfg.training.get("logq_correction", "none")
                ),
                logq_power=float(cfg.training.get("logq_power", 1.0)),
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(cfg.training.gradient_clip)
        )
        optimizer.step()
        train_seconds += time.perf_counter() - train_started
        examples_seen += len(batch)
        steps += 1
        last_loss = float(loss.detach().cpu())
        predicted = logits.argmax(dim=1)
        last_accuracy = float(
            positive[torch.arange(len(batch), device=device), predicted]
            .float()
            .mean()
            .cpu()
        )
        if steps % int(cfg.training.log_every) == 0:
            elapsed = time.perf_counter() - started
            live = {
                "train/week_index": float(week_index),
                "train/loss": last_loss,
                "train/in_batch_accuracy": last_accuracy,
                "train/examples_seen_in_week": float(examples_seen),
                "train/rows_per_second": examples_seen / max(elapsed, 1e-9),
                "train/data_wait_fraction": data_wait_seconds / max(elapsed, 1e-9),
            }
            LOGGER.info(
                "week=%s step=%s rows=%s loss=%.4f acc=%.4f rows/s=%.0f",
                week_index,
                steps,
                f"{examples_seen:,}",
                last_loss,
                last_accuracy,
                examples_seen / max(elapsed, 1e-9),
            )
            if tracker is not None:
                tracker.log(week_index * 1_000_000 + steps, live)
        if max_examples > 0 and examples_seen >= max_examples:
            break
    elapsed = time.perf_counter() - started
    return {
        "week_index": week_index,
        "steps": steps,
        "examples_seen": examples_seen,
        "seconds": elapsed,
        "rows_per_second": examples_seen / max(elapsed, 1e-9),
        "data_wait_seconds": data_wait_seconds,
        "train_seconds": train_seconds,
        "last_loss": last_loss,
        "last_accuracy": last_accuracy,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else 0,
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        * 1024,
    }


def save_training_checkpoint(
    path: Path,
    *,
    cfg: DictConfig,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    completed_week_index: int,
    metrics: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "version": 1,
            "config": OmegaConf.to_container(cfg, resolve=True),
            "state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "completed_week_index": int(completed_week_index),
            "metrics": dict(metrics),
        },
        temporary,
    )
    os.replace(temporary, path)


def load_training_checkpoint(
    path: Path,
    *,
    cfg: DictConfig,
    device: torch.device,
) -> tuple[torch.nn.Module, torch.optim.Optimizer, int]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model, optimizer = initialize_state(cfg, device=device)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return model, optimizer, int(checkpoint["completed_week_index"])


def export_snapshot(
    *,
    cfg: DictConfig,
    model: torch.nn.Module,
    artifact_dir: Path,
    device: torch.device,
    lifecycle: Mapping[str, Any],
) -> dict[str, Any]:
    if (artifact_dir / "manifest.json").is_file():
        return json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    if artifact_dir.exists() and any(artifact_dir.iterdir()):
        raise FileExistsError(f"Incomplete snapshot exists: {artifact_dir}")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = copy_tokenizer_artifact(cfg, artifact_dir)
    checkpoint = {
        "version": 2,
        "solution": str(cfg.experiment.name),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "state_dict": model.state_dict(),
        "training": dict(lifecycle),
    }
    temporary = artifact_dir / f"{MODEL_FILENAME}.tmp"
    torch.save(checkpoint, temporary)
    os.replace(temporary, artifact_dir / MODEL_FILENAME)
    candidates = export_candidates(
        cfg=cfg,
        model=model,
        artifact_dir=artifact_dir,
        device=device,
    )
    artifact_files = [
        "model.pt",
        "candidate_embeddings.npy",
        "candidate_metadata.parquet",
    ]
    if tokenizer_path is not None:
        artifact_files.append("tokenizer.json")
    manifest = {
        "version": 1,
        "solution": "two_tower_v2_walk_forward_snapshot",
        "lifecycle": dict(lifecycle),
        "candidates": candidates,
        "files": {
            name: {"bytes": (artifact_dir / name).stat().st_size}
            for name in artifact_files
        },
    }
    atomic_json(artifact_dir / "manifest.json", manifest)
    return manifest


def _request_key(row: Mapping[str, Any]) -> str:
    payload = "\x1f".join(
        [
            str(row.get("uniq_id") or ""),
            str(int(row["show_time"])),
            str(row.get("query") or ""),
        ]
    )
    return "oof:" + hashlib.sha1(payload.encode("utf-8")).hexdigest()


def extract_oof_requests(
    *,
    rows: Iterable[Mapping[str, Any]],
    weeks: Sequence[int],
    requests_per_week: int,
    output: Path,
    history_output: Path | None = None,
) -> dict[str, Any]:
    """Deterministically keep the smallest request hashes in every week."""

    if requests_per_week <= 0:
        raise ValueError("requests_per_week must be positive")
    selected: dict[int, dict[str, dict[str, Any]]] = {week: {} for week in weeks}
    heaps: dict[int, list[tuple[int, str]]] = {week: [] for week in weeks}
    history_writer = None
    history_temporary = None
    history_buffer: list[dict[str, Any]] = []
    history_rows = 0
    if history_output is not None:
        history_output.parent.mkdir(parents=True, exist_ok=True)
        history_temporary = history_output.with_suffix(history_output.suffix + ".tmp")
        history_temporary.unlink(missing_ok=True)
        history_writer = pq.ParquetWriter(
            history_temporary, COUNTER_EVENT_SCHEMA, compression="zstd"
        )

    def flush_history() -> None:
        nonlocal history_rows
        if history_writer is None or not history_buffer:
            return
        table = pa.Table.from_pylist(history_buffer, schema=COUNTER_EVENT_SCHEMA)
        history_writer.write_table(table)
        history_rows += table.num_rows
        history_buffer.clear()

    try:
        for raw in rows:
            timestamp = int(raw["show_time"])
            week = week_start(timestamp)
            if week not in selected:
                continue
            if history_writer is not None:
                history_buffer.append(
                    {
                        "show_time": timestamp,
                        "banner_id": int(raw["banner_id"]),
                        "group_id": None,
                        "domain": "",
                        "query_key": stable_text_key(raw.get("query")),
                        "region_key": scalar_key(raw.get("region_id")),
                        "user_key": scalar_key(raw.get("crypta_id_v2")),
                        "source_cost": float(raw.get("source_cost") or 0.0),
                    }
                )
                if len(history_buffer) >= 100_000:
                    flush_history()
            request_id = _request_key(raw)
            bucket = selected[week]
            if request_id in bucket:
                banner_id = int(raw["banner_id"])
                if banner_id not in bucket[request_id]["clicked_banner_ids"]:
                    bucket[request_id]["clicked_banner_ids"].append(banner_id)
                    bucket[request_id]["clicked_source_costs"].append(
                        float(raw.get("source_cost") or 0.0)
                    )
                continue
            priority = int(request_id[-16:], 16)
            if len(bucket) >= requests_per_week and priority >= -heaps[week][0][0]:
                continue
            if len(bucket) >= requests_per_week:
                _, removed = heapq.heappop(heaps[week])
                bucket.pop(removed, None)
            hit_log_id = int.from_bytes(
                hashlib.sha1(request_id.encode("utf-8")).digest()[:8], "little"
            )
            bucket[request_id] = {
                "request_id": request_id,
                "hit_log_id": hit_log_id,
                "show_time": timestamp,
                "query": str(raw.get("query") or ""),
                "region_id": raw.get("region_id"),
                "crypta_id_v2": raw.get("crypta_id_v2"),
                "device": raw.get("device"),
                "age": raw.get("age"),
                "gender": raw.get("gender"),
                "clicked_banner_ids": [int(raw["banner_id"])],
                "clicked_source_costs": [float(raw.get("source_cost") or 0.0)],
            }
            heapq.heappush(heaps[week], (-priority, request_id))
        flush_history()
    finally:
        if history_writer is not None:
            history_writer.close()
    if history_output is not None and history_temporary is not None:
        os.replace(history_temporary, history_output)
    materialized = sorted(
        (row for bucket in selected.values() for row in bucket.values()),
        key=lambda row: (int(row["show_time"]), str(row["request_id"])),
    )
    table = write_request_parquet(output, materialized)
    return {
        "weeks": list(weeks),
        "requests_per_week": requests_per_week,
        "requests": table.num_rows,
        "per_week": {
            str(week): len(selected[week])
            for week in weeks
        },
        "output": str(output),
        "history_output": str(history_output) if history_output else None,
        "history_rows": history_rows,
    }
