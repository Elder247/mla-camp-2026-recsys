#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import logging
import os
import random
import resource
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Leakage-safe temporal/full fine-tune of a completed TwoTower"
    )
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def load_config(path: Path):
    cfg = OmegaConf.load(path)
    parent = cfg.get("extends")
    if parent:
        base = load_config((path.parent / str(parent)).resolve())
        cfg = OmegaConf.merge(base, cfg)
        del cfg["extends"]
    return cfg


def select_rows(
    rows: list[dict[str, Any]], *, scope: str, boundary: int
) -> list[dict[str, Any]]:
    if scope == "temporal_fit":
        selected = [row for row in rows if int(row["show_time"]) < boundary]
    elif scope == "full":
        selected = list(rows)
    else:
        raise ValueError(f"unknown fine-tune scope: {scope}")
    return sorted(selected, key=lambda row: (int(row["show_time"]), int(row["hit_log_id"])))


def main() -> int:
    args = arguments()
    cfg = load_config(args.config.resolve())
    sys.path.insert(0, str(cfg.paths.step2_root))
    import pyarrow.parquet as pq

    from common.text import as_text, normalize, tokenize
    from mla_recsys.tracking import UnderdeepTracker, numeric_metrics
    from two_tower_v2.data import batches, enrich_rows, feature_bucket, pack_bags
    from two_tower_v2.training import (
        EMBEDDINGS_FILENAME,
        METADATA_FILENAME,
        MODEL_FILENAME,
        TOKENIZER_FILENAME,
        all_cardinalities,
        atomic_json,
        bpe_limits,
        build_model,
        export_candidates,
        git_sha,
        load_bpe_tokenizer,
        resolve_device,
        retrieval_objective,
    )

    base_artifact = Path(str(cfg.paths.base_artifact))
    artifact_dir = Path(str(cfg.paths.artifact_dir))
    if artifact_dir.exists() and any(artifact_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite fine-tune artifact: {artifact_dir}")
    required = [base_artifact / MODEL_FILENAME, base_artifact / "manifest.json"]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Base TwoTower artifact is incomplete: {missing}")
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
    (artifact_dir / "finetune.config.resolved.yaml").write_text(
        resolved, encoding="utf-8"
    )
    checkpoint = torch.load(
        base_artifact / MODEL_FILENAME, map_location="cpu", weights_only=True
    )
    model_cfg = OmegaConf.create(checkpoint["config"])
    tokenizer_source = base_artifact / TOKENIZER_FILENAME
    if tokenizer_source.is_file():
        shutil.copy2(tokenizer_source, artifact_dir / TOKENIZER_FILENAME)
        model_cfg.paths.tokenizer_file = str(artifact_dir / TOKENIZER_FILENAME)
    tokenizer = load_bpe_tokenizer(model_cfg, artifact_dir=artifact_dir)
    cardinalities = all_cardinalities(model_cfg)
    limits = bpe_limits(model_cfg)

    columns = [
        "HitLogID",
        "ShowTime",
        "SearchQuery",
        "RegionID",
        "BannerID",
        "GroupExportID",
        "BannerTitle",
        "BannerText",
        "SourceCost",
        "IsClick",
    ]
    data = pq.read_table(Path(str(cfg.paths.val_file)), columns=columns).to_pydict()
    rows = []
    for index, click in enumerate(data["IsClick"]):
        if int(click or 0) <= 0:
            continue
        query = as_text(data["SearchQuery"][index])
        title = as_text(data["BannerTitle"][index])
        text = as_text(data["BannerText"][index])
        region = int(data["RegionID"][index] or 0)
        banner = int(data["BannerID"][index])
        group = int(data["GroupExportID"][index] or 0)
        rows.append(
            {
                "hit_log_id": int(data["HitLogID"][index]),
                "show_time": int(data["ShowTime"][index]),
                "source_cost": float(data["SourceCost"][index] or 0.0),
                "query_word_ids": [
                    feature_bucket(token) for token in tokenize(query)[:32]
                ],
                "region_ids": [feature_bucket(str(region))],
                "banner_id_ids": [feature_bucket(str(banner))],
                "ad_group_id_ids": [feature_bucket(str(group))],
                "title_word_ids": [
                    feature_bucket(token) for token in tokenize(title)[:32]
                ],
                "text_word_ids": [
                    feature_bucket(token) for token in tokenize(text)[:64]
                ],
                "query_text": normalize(query),
                "title_text": normalize(title),
                "text_text": normalize(text),
            }
        )
    selected = select_rows(
        rows,
        scope=str(cfg.finetune.scope),
        boundary=int(cfg.finetune.temporal_boundary),
    )
    expected = int(cfg.finetune.expected_rows)
    if expected > 0 and len(selected) != expected:
        raise RuntimeError(f"Unexpected fine-tune rows: {len(selected)} != {expected}")

    seed = int(cfg.finetune.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = resolve_device(str(cfg.runtime.device))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device)
    model = build_model(model_cfg).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.finetune.learning_rate),
        weight_decay=float(cfg.finetune.weight_decay),
    )
    tracker = UnderdeepTracker(
        artifact_dir=artifact_dir,
        tracking_cfg=cfg.tracking.underdeep,
        run_name=str(cfg.tracking.underdeep.run_name),
        description="Leakage-safe chronological validation fine-tune of TwoTower",
        parameters={
            "base_artifact": str(base_artifact),
            "scope": str(cfg.finetune.scope),
            "rows": len(selected),
            "epochs": int(cfg.finetune.epochs),
        },
        tags=["mla-camp", "two-tower", "validation-finetune"],
    )
    started = time.perf_counter()
    step = 0
    examples_seen = 0
    last_loss = 0.0
    last_accuracy = 0.0
    model.train()
    try:
        for epoch in range(int(cfg.finetune.epochs)):
            epoch_rows = list(selected)
            if not bool(cfg.finetune.chronological):
                random.Random(seed + epoch).shuffle(epoch_rows)
            for raw_batch in batches(epoch_rows, int(cfg.finetune.batch_size)):
                batch = enrich_rows(
                    raw_batch,
                    cardinalities=cardinalities,
                    tokenizer=tokenizer,
                    bpe_limits=limits,
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
                    logits = query @ banner.T / float(cfg.finetune.temperature)
                    loss, positives = retrieval_objective(
                        logits,
                        batch,
                        objective=str(cfg.finetune.objective),
                        symmetric_weight=float(cfg.finetune.symmetric_weight),
                    )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(cfg.finetune.gradient_clip)
                )
                optimizer.step()
                step += 1
                examples_seen += len(batch)
                last_loss = float(loss.detach().cpu())
                predicted = logits.argmax(dim=1)
                last_accuracy = float(
                    positives[torch.arange(len(batch), device=device), predicted]
                    .float()
                    .mean()
                    .cpu()
                )
            tracker.log(
                epoch + 1,
                {
                    "finetune/epoch": float(epoch + 1),
                    "finetune/loss": last_loss,
                    "finetune/in_batch_accuracy": last_accuracy,
                    "finetune/examples_seen": float(examples_seen),
                },
            )
            logging.info(
                "epoch=%s rows=%s loss=%.4f acc=%.4f",
                epoch + 1,
                f"{examples_seen:,}",
                last_loss,
                last_accuracy,
            )
        training_seconds = time.perf_counter() - started
        metrics = {
            "base_artifact": str(base_artifact),
            "scope": str(cfg.finetune.scope),
            "rows": len(selected),
            "epochs": int(cfg.finetune.epochs),
            "steps": step,
            "examples_seen": examples_seen,
            "seconds": training_seconds,
            "last_loss": last_loss,
            "last_accuracy": last_accuracy,
            "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0,
            "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            * 1024,
        }
        output_checkpoint = {
            "version": 2,
            "solution": str(cfg.experiment.name),
            "config": OmegaConf.to_container(model_cfg, resolve=True),
            "state_dict": model.state_dict(),
            "training": metrics,
        }
        temporary = artifact_dir / f"{MODEL_FILENAME}.tmp"
        torch.save(output_checkpoint, temporary)
        os.replace(temporary, artifact_dir / MODEL_FILENAME)
        candidates = export_candidates(
            cfg=model_cfg,
            model=model,
            artifact_dir=artifact_dir,
            device=device,
        )
        files = [MODEL_FILENAME, EMBEDDINGS_FILENAME, METADATA_FILENAME]
        if (artifact_dir / TOKENIZER_FILENAME).is_file():
            files.append(TOKENIZER_FILENAME)
        report = {
            "version": 1,
            "solution": str(cfg.experiment.name),
            "git_sha": git_sha(ROOT),
            "base_model_sha256": hashlib.sha256(
                (base_artifact / MODEL_FILENAME).read_bytes()
            ).hexdigest(),
            "training": metrics,
            "candidates": candidates,
            "files": {
                name: {"bytes": (artifact_dir / name).stat().st_size}
                for name in files
            },
        }
        atomic_json(artifact_dir / "metrics.json", report)
        atomic_json(artifact_dir / "manifest.json", report)
        tracker.log_summary(numeric_metrics(report, prefix="finetune"))
        tracker.close()
    except Exception as error:
        tracker.close(error=type(error).__name__)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
