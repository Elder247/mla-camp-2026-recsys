from __future__ import annotations

import json
import logging
import os
import random
import resource
import shutil
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.nn import functional as F

from common.text import as_text, normalize, tokenize
from two_tower_v2.data import (
    YtTableSource,
    batches,
    enrich_rows,
    feature_bucket,
    pack_bags,
    prefetch_batches,
    shuffled_rows,
)
from two_tower_v2.model import TwoTowerV2


LOGGER = logging.getLogger(__name__)
MODEL_FILENAME = "model.pt"
EMBEDDINGS_FILENAME = "candidate_embeddings.npy"
METADATA_FILENAME = "candidate_metadata.parquet"
TOKENIZER_FILENAME = "tokenizer.json"


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(value)


def build_model(cfg: DictConfig | Mapping[str, Any]) -> TwoTowerV2:
    model = cfg["model"]
    return TwoTowerV2(
        query_cardinalities=dict(model["query_cardinalities"]),
        banner_cardinalities=dict(model["banner_cardinalities"]),
        embedding_policy=dict(model["embedding_policy"]),
        hidden_dim=int(model["hidden_dim"]),
        output_dim=int(model["output_dim"]),
        cross_layers=int(model["cross_layers"]),
        deep_layers=int(model["deep_layers"]),
        dropout=float(model["dropout"]),
        deep_residual=bool(model.get("deep_residual", False)),
    )


def all_cardinalities(cfg: DictConfig | Mapping[str, Any]) -> dict[str, int]:
    model = cfg["model"]
    return {
        **{str(k): int(v) for k, v in model["query_cardinalities"].items()},
        **{str(k): int(v) for k, v in model["banner_cardinalities"].items()},
    }


def load_bpe_tokenizer(
    cfg: DictConfig | Mapping[str, Any],
    *,
    artifact_dir: Path | None = None,
) -> Any | None:
    cardinalities = all_cardinalities(cfg)
    if not any(name.endswith("_bpe_ids") for name in cardinalities):
        return None
    from tokenizers import Tokenizer

    candidates = []
    if artifact_dir is not None:
        candidates.append(artifact_dir / TOKENIZER_FILENAME)
    configured = cfg.get("paths", {}).get("tokenizer_file")
    if configured:
        candidates.append(Path(str(configured)))
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise FileNotFoundError(
            "BPE fields are configured but tokenizer.json is unavailable"
        )
    return Tokenizer.from_file(str(path))


def bpe_limits(cfg: DictConfig | Mapping[str, Any]) -> dict[str, int]:
    configured = cfg.get("text_features", {}).get("limits", {})
    return {str(name): int(value) for name, value in dict(configured).items()}


def numeric_feature_scale(
    cfg: DictConfig | Mapping[str, Any], name: str
) -> float:
    return float(cfg.get("numeric_features", {}).get(name, 1.0))


def copy_tokenizer_artifact(
    cfg: DictConfig | Mapping[str, Any], artifact_dir: Path
) -> Path | None:
    tokenizer = load_bpe_tokenizer(cfg)
    if tokenizer is None:
        return None
    source = Path(str(cfg["paths"]["tokenizer_file"]))
    target = artifact_dir / TOKENIZER_FILENAME
    if source.resolve() != target.resolve():
        shutil.copy2(source, target)
    return target


def positive_ids(
    rows: list[dict[str, Any]], *, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return contiguous query and banner identities for one sampled batch."""

    query_ids: dict[tuple[Any, ...], int] = {}
    banner_ids: dict[tuple[str, int], int] = {}
    query_values = []
    banner_values = []
    for index, row in enumerate(rows):
        query_key = (
            tuple(int(value) for value in row.get("query_word_ids") or ()),
            tuple(int(value) for value in row.get("region_ids") or ()),
            (
                ("raw_user", int(row["crypta_id_v2"]))
                if int(row.get("crypta_id_v2") or 0) > 0
                else ("no_user", 0)
            ),
        )
        query_values.append(query_ids.setdefault(query_key, len(query_ids)))
        raw_banner_id = int(row.get("banner_id") or 0)
        banner = row.get("banner_id_ids") or ()
        banner_key = (
            ("raw", raw_banner_id)
            if raw_banner_id > 0
            else ("hashed", int(banner[0]) if banner else -(index + 1))
        )
        banner_values.append(
            banner_ids.setdefault(banner_key, len(banner_ids))
        )
    query_tensor = torch.tensor(query_values, dtype=torch.long, device=device)
    banner_tensor = torch.tensor(banner_values, dtype=torch.long, device=device)
    return query_tensor, banner_tensor


def positive_mask(
    rows: list[dict[str, Any]], *, device: torch.device
) -> torch.Tensor:
    """Mark repeated-banner and repeated-query clicks as genuine positives."""

    query_tensor, banner_tensor = positive_ids(rows, device=device)
    return (query_tensor[:, None] == query_tensor[None, :]) | (
        banner_tensor[:, None] == banner_tensor[None, :]
    )


def batch_frequency_logq(
    rows: list[dict[str, Any]], *, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate per-sample log Q from repeated identities in the current batch.

    The omitted ``-log(batch_size)`` term is constant within each softmax and
    therefore does not change the objective. Returning query and banner terms
    separately keeps the symmetric retrieval loss correct in both directions.
    """

    query_ids, banner_ids = positive_ids(rows, device=device)

    def repeated_log_count(values: torch.Tensor) -> torch.Tensor:
        _, inverse, counts = torch.unique(
            values,
            sorted=False,
            return_inverse=True,
            return_counts=True,
        )
        return counts[inverse].to(dtype=torch.float32).log()

    return repeated_log_count(query_ids), repeated_log_count(banner_ids)


def sourcecost_example_weights(
    rows: list[dict[str, Any]],
    *,
    power: float,
    minimum: float,
    maximum: float,
    device: torch.device,
) -> torch.Tensor:
    """Return bounded mean-one weights aligned with SourceCost Recall."""

    if power < 0.0:
        raise ValueError("sourcecost_weight_power must be non-negative")
    if minimum <= 0.0 or maximum < minimum:
        raise ValueError("invalid SourceCost weight bounds")
    if power == 0.0:
        return torch.ones(len(rows), dtype=torch.float32, device=device)
    values = torch.tensor(
        [np.log1p(max(0.0, float(row.get("source_cost") or 0.0))) for row in rows],
        dtype=torch.float32,
        device=device,
    )
    positive = values[values > 0]
    scale = positive.median() if positive.numel() else values.new_tensor(1.0)
    weights = (values / scale.clamp_min(1.0e-6)).clamp_min(1.0e-6).pow(power)
    weights = weights.clamp(min=minimum, max=maximum)
    return weights / weights.mean().clamp_min(1.0e-6)


def retrieval_objective(
    logits: torch.Tensor,
    rows: list[dict[str, Any]],
    *,
    objective: str,
    symmetric_weight: float,
    sourcecost_weight_power: float = 0.0,
    sourcecost_weight_min: float = 1.0,
    sourcecost_weight_max: float = 1.0,
    logq_correction: str = "none",
    logq_power: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    weights = sourcecost_example_weights(
        rows,
        power=sourcecost_weight_power,
        minimum=sourcecost_weight_min,
        maximum=sourcecost_weight_max,
        device=logits.device,
    )
    correction = str(logq_correction).lower()
    if correction not in {"none", "batch_frequency"}:
        raise ValueError(f"unknown logQ correction: {logq_correction}")
    if logq_power < 0.0:
        raise ValueError("logq_power must be non-negative")
    query_logits = logits.float()
    banner_logits = logits.T.float()
    if correction == "batch_frequency" and logq_power > 0.0:
        query_logq, banner_logq = batch_frequency_logq(
            rows, device=logits.device
        )
        query_logits = query_logits - logq_power * banner_logq.unsqueeze(0)
        banner_logits = banner_logits - logq_power * query_logq.unsqueeze(0)

    if objective == "cross_entropy":
        labels = torch.arange(len(rows), device=logits.device)
        mask = torch.eye(len(rows), dtype=torch.bool, device=logits.device)
        losses = F.cross_entropy(query_logits, labels, reduction="none")
        return (losses * weights).sum() / weights.sum(), mask
    if objective != "multi_positive":
        raise ValueError(f"unknown TwoTower objective: {objective}")
    if symmetric_weight < 0:
        raise ValueError("symmetric_weight must be non-negative")
    mask = positive_mask(rows, device=logits.device)

    def direction_loss(
        values: torch.Tensor,
        positives: torch.Tensor,
        row_weights: torch.Tensor,
    ) -> torch.Tensor:
        numerator = torch.logsumexp(
            values.float().masked_fill(~positives, float("-inf")), dim=1
        )
        denominator = torch.logsumexp(values.float(), dim=1)
        losses = denominator - numerator
        return (losses * row_weights).sum() / row_weights.sum()

    query_loss = direction_loss(query_logits, mask, weights)
    if symmetric_weight == 0:
        return query_loss, mask
    banner_loss = direction_loss(banner_logits, mask.T, weights)
    return (
        query_loss + symmetric_weight * banner_loss
    ) / (1.0 + symmetric_weight), mask


@torch.inference_mode()
def evaluate(
    model: TwoTowerV2,
    rows: list[dict[str, list[int]]],
    *,
    cfg: DictConfig,
    device: torch.device,
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    total_loss = 0.0
    correct = 0
    examples = 0
    cardinalities = all_cardinalities(cfg)
    tokenizer = load_bpe_tokenizer(cfg)
    objective = str(cfg.training.get("objective", "cross_entropy"))
    symmetric_weight = float(cfg.training.get("symmetric_weight", 0.0))
    batch_size = int(cfg.training.batch_size)
    for batch in batches(rows, batch_size):
        batch = enrich_rows(
            batch,
            cardinalities=cardinalities,
            tokenizer=tokenizer,
            bpe_limits=bpe_limits(cfg),
            source_cost_log1p_scale=float(
                cfg.get("numeric_features", {}).get("source_cost_log1p_scale", 1.0)
            ),
            product_price_log1p_scale=numeric_feature_scale(
                cfg, "product_price_log1p_scale"
            ),
        )
        bags = pack_bags(batch, cardinalities=cardinalities, device=device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            query = model.encode_query(bags)
            banner = model.encode_banner(bags)
            logits = query @ banner.T / float(cfg.training.temperature)
        loss, mask = retrieval_objective(
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
        total_loss += float(loss.cpu()) * len(batch)
        predicted = logits.argmax(dim=1)
        correct += int(mask[torch.arange(len(batch), device=device), predicted].sum().cpu())
        examples += len(batch)
    if was_training:
        model.train()
    return {
        "loss": total_loss / max(examples, 1),
        "in_batch_accuracy": correct / max(examples, 1),
        "examples": float(examples),
    }


def train_model(
    *,
    cfg: DictConfig,
    source: YtTableSource,
    validation: YtTableSource,
    artifact_dir: Path,
    device: torch.device,
    tracker: Any | None = None,
) -> tuple[TwoTowerV2, dict[str, Any]]:
    seed = int(cfg.training.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device)
    model = build_model(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.training.learning_rate),
        weight_decay=float(cfg.training.weight_decay),
    )
    cardinalities = all_cardinalities(cfg)
    tokenizer = load_bpe_tokenizer(cfg)
    limits = bpe_limits(cfg)
    validation_rows = list(validation.rows())
    objective = str(cfg.training.get("objective", "cross_entropy"))
    symmetric_weight = float(cfg.training.get("symmetric_weight", 0.0))
    started = time.perf_counter()
    data_wait_seconds = 0.0
    train_seconds = 0.0
    examples_seen = 0
    step = 0
    last_loss = 0.0
    last_accuracy = 0.0
    last_validation: dict[str, float] = {}
    next_validation = int(cfg.training.validate_every_rows)
    max_steps = int(cfg.training.max_steps)
    max_examples = int(cfg.training.max_examples)
    for epoch in range(int(cfg.training.epochs)):
        iterator = iter(
            prefetch_batches(
                batches(
                    shuffled_rows(
                        source.rows(),
                        buffer_size=int(cfg.training.shuffle_buffer),
                        seed=seed + epoch,
                    ),
                    int(cfg.training.batch_size),
                ),
                int(cfg.training.get("prefetch_batches", 0)),
            )
        )
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
                    cfg.get("numeric_features", {}).get(
                        "source_cost_log1p_scale", 1.0
                    )
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
                loss, mask = retrieval_objective(
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
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(cfg.training.gradient_clip)
                ).detach().cpu()
            )
            optimizer.step()
            train_seconds += time.perf_counter() - train_started
            step += 1
            examples_seen += len(batch)
            last_loss = float(loss.detach().cpu())
            predicted = logits.argmax(dim=1)
            last_accuracy = float(
                mask[torch.arange(len(batch), device=device), predicted]
                .float()
                .mean()
                .cpu()
            )
            if examples_seen >= next_validation:
                last_validation = evaluate(
                    model,
                    validation_rows,
                    cfg=cfg,
                    device=device,
                )
                LOGGER.info(
                    "validation rows=%s loss=%.4f acc=%.4f",
                    f"{examples_seen:,}",
                    last_validation["loss"],
                    last_validation["in_batch_accuracy"],
                )
                while next_validation <= examples_seen:
                    next_validation += int(cfg.training.validate_every_rows)
            if step % int(cfg.training.log_every) == 0:
                elapsed = time.perf_counter() - started
                live = {
                    "train/loss": last_loss,
                    "train/in_batch_accuracy": last_accuracy,
                    "train/examples_seen": float(examples_seen),
                    "train/rows_per_second": examples_seen / max(elapsed, 1e-9),
                    "train/data_wait_fraction": data_wait_seconds
                    / max(elapsed, 1e-9),
                }
                for name, value in last_validation.items():
                    live[f"validation/{name}"] = float(value)
                LOGGER.info(
                    "step=%s rows=%s loss=%.4f acc=%.4f grad=%.3f rows/s=%.0f wait=%.1f%%",
                    step,
                    f"{examples_seen:,}",
                    last_loss,
                    last_accuracy,
                    gradient_norm,
                    examples_seen / max(elapsed, 1e-9),
                    100 * data_wait_seconds / max(elapsed, 1e-9),
                )
                if tracker is not None:
                    tracker.log(step, live)
            if (max_steps > 0 and step >= max_steps) or (
                max_examples > 0 and examples_seen >= max_examples
            ):
                break
        if (max_steps > 0 and step >= max_steps) or (
            max_examples > 0 and examples_seen >= max_examples
        ):
            break
    if not last_validation:
        last_validation = evaluate(
            model,
            validation_rows,
            cfg=cfg,
            device=device,
        )
    elapsed = time.perf_counter() - started
    metrics = {
        "source": source.description,
        "source_rows": source.row_count,
        "validation_source": validation.description,
        "steps": step,
        "examples_seen": examples_seen,
        "seconds": elapsed,
        "rows_per_second": examples_seen / max(elapsed, 1e-9),
        "data_wait_seconds": data_wait_seconds,
        "data_wait_fraction": data_wait_seconds / max(elapsed, 1e-9),
        "train_seconds": train_seconds,
        "last_loss": last_loss,
        "last_accuracy": last_accuracy,
        "validation": last_validation,
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
    }
    checkpoint = {
        "version": 2,
        "solution": str(cfg.experiment.name),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "state_dict": model.state_dict(),
        "training": metrics,
    }
    temporary = artifact_dir / f"{MODEL_FILENAME}.tmp"
    torch.save(checkpoint, temporary)
    os.replace(temporary, artifact_dir / MODEL_FILENAME)
    return model, metrics


def _candidate_row(columns: dict[str, list[Any]], index: int) -> tuple[dict, dict]:
    banner_id = int(columns["BannerID"][index])
    group_id = int(columns["GroupExportID"][index] or 0)
    title = as_text(columns["BannerTitle"][index])
    text = as_text(columns["BannerText"][index])
    row = {
        "banner_id": banner_id,
        "banner_id_ids": [feature_bucket(str(banner_id))],
        "ad_group_id_ids": [feature_bucket(str(group_id))],
        "title_word_ids": [feature_bucket(token) for token in tokenize(title)[:32]],
        "text_word_ids": [feature_bucket(token) for token in tokenize(text)[:64]],
        "query_text": "",
        "title_text": normalize(title),
        "text_text": normalize(text),
        "source_cost": float(columns["SourceCost"][index] or 0.0),
        "product_price": float(columns["ProductPrice"][index] or 0.0),
        "banner_url": as_text(columns["BannerURL"][index]),
        "client_id_ids": [feature_bucket(str(columns["ClientID"][index] or 0))],
        "order_id_ids": [feature_bucket(str(columns["OrderID"][index] or 0))],
        "caesar_model_id_ids": [
            feature_bucket(str(columns["CaesarModelID"][index] or 0))
        ],
        "caesar_sku_id_ids": [
            feature_bucket(str(columns["CaesarSkuID"][index] or 0))
        ],
    }
    metadata = {
        "banner_id": banner_id,
        "ad_group_id": group_id,
        "title": title,
        "text": text,
        "url": as_text(columns["BannerURL"][index]),
        "source_cost": float(columns["SourceCost"][index] or 0.0),
    }
    return row, metadata


@torch.inference_mode()
def export_candidates(
    *,
    cfg: DictConfig,
    model: TwoTowerV2,
    artifact_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    index_file = Path(str(cfg.paths.index_file))
    parquet = pq.ParquetFile(index_file)
    total = parquet.metadata.num_rows
    max_rows = int(cfg.export.max_index_rows)
    if max_rows > 0:
        total = min(total, max_rows)
    banner_cardinalities = {
        str(k): int(v) for k, v in cfg.model.banner_cardinalities.items()
    }
    tokenizer = load_bpe_tokenizer(cfg, artifact_dir=artifact_dir)
    limits = bpe_limits(cfg)
    embeddings: list[np.ndarray] = []
    metadata_path = artifact_dir / METADATA_FILENAME
    temporary_metadata = metadata_path.with_suffix(".parquet.tmp")
    writer = None
    done = 0
    started = time.perf_counter()
    model.eval()
    try:
        for arrow_batch in parquet.iter_batches(batch_size=int(cfg.export.batch_size)):
            if done >= total:
                break
            columns = arrow_batch.to_pydict()
            keep = min(arrow_batch.num_rows, total - done)
            rows: list[dict[str, list[int]]] = []
            metadata = {
                "banner_id": [],
                "ad_group_id": [],
                "title": [],
                "text": [],
                "url": [],
                "source_cost": [],
            }
            for index in range(keep):
                row, item = _candidate_row(columns, index)
                rows.append(row)
                for name, value in item.items():
                    metadata[name].append(value)
            rows = enrich_rows(
                rows,
                cardinalities=banner_cardinalities,
                tokenizer=tokenizer,
                bpe_limits=limits,
                source_cost_log1p_scale=float(
                    cfg.get("numeric_features", {}).get(
                        "source_cost_log1p_scale", 1.0
                    )
                ),
                product_price_log1p_scale=numeric_feature_scale(
                    cfg, "product_price_log1p_scale"
                ),
            )
            bags = pack_bags(
                rows,
                cardinalities=banner_cardinalities,
                device=device,
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                vectors = model.encode_banner(bags)
            embeddings.append(vectors.float().cpu().numpy().astype(np.float16))
            table = pa.table(metadata)
            if writer is None:
                writer = pq.ParquetWriter(temporary_metadata, table.schema, compression="zstd")
            writer.write_table(table)
            done += keep
            if done % 100_000 < keep:
                LOGGER.info("candidate export %s/%s", f"{done:,}", f"{total:,}")
    finally:
        if writer is not None:
            writer.close()
    if done == 0:
        raise RuntimeError("candidate index is empty")
    os.replace(temporary_metadata, metadata_path)
    array = np.concatenate(embeddings)
    embeddings_path = artifact_dir / EMBEDDINGS_FILENAME
    temporary_embeddings = embeddings_path.with_suffix(".npy.tmp")
    with temporary_embeddings.open("wb") as target:
        np.save(target, array, allow_pickle=False)
    os.replace(temporary_embeddings, embeddings_path)
    return {
        "index_file": str(index_file),
        "index_size_bytes": index_file.stat().st_size,
        "candidates": done,
        "embedding_shape": list(array.shape),
        "embedding_dtype": str(array.dtype),
        "seconds": time.perf_counter() - started,
    }


def git_sha(root: Path) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() or None
