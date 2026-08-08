#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_config(path: Path):
    cfg = OmegaConf.load(path)
    parent = cfg.get("extends")
    if parent:
        base = load_config((path.parent / str(parent)).resolve())
        cfg = OmegaConf.merge(base, cfg)
        del cfg["extends"]
    return cfg


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train field-aware DCN TwoTower v2")
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = arguments()
    cfg = load_config(args.config.resolve())
    sys.path.insert(0, str(cfg.paths.step2_root))
    from two_tower_v2.data import YtTableSource
    from two_tower_v2.training import (
        atomic_json,
        export_candidates,
        git_sha,
        resolve_device,
        train_model,
    )

    artifact_dir = Path(str(cfg.paths.artifact_dir))
    if artifact_dir.exists() and any(artifact_dir.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite non-empty TwoTower artifact: {artifact_dir}"
        )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    log_path = artifact_dir / "train.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path, encoding="utf-8")],
    )
    resolved = OmegaConf.to_yaml(cfg, resolve=True)
    (artifact_dir / "config.resolved.yaml").write_text(resolved, encoding="utf-8")
    source = YtTableSource(str(cfg.paths.train_table), str(cfg.paths.proxy))
    validation = YtTableSource(str(cfg.paths.validation_table), str(cfg.paths.proxy))
    device = resolve_device(str(cfg.runtime.device))
    model, training = train_model(
        cfg=cfg,
        source=source,
        validation=validation,
        artifact_dir=artifact_dir,
        device=device,
    )
    candidates = export_candidates(
        cfg=cfg,
        model=model,
        artifact_dir=artifact_dir,
        device=device,
    )
    config_sha = hashlib.sha256(resolved.encode("utf-8")).hexdigest()
    manifest = {
        "version": 1,
        "solution": str(cfg.experiment.name),
        "config_sha256": config_sha,
        "git_sha": git_sha(ROOT),
        "training": training,
        "candidates": candidates,
        "files": {
            name: {
                "bytes": (artifact_dir / name).stat().st_size,
            }
            for name in ("model.pt", "candidate_embeddings.npy", "candidate_metadata.parquet")
        },
    }
    atomic_json(artifact_dir / "metrics.json", {"training": training, "candidates": candidates})
    atomic_json(artifact_dir / "manifest.json", manifest)
    logging.info("TwoTower v2 artifact completed: %s", artifact_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
