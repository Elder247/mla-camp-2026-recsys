#!/usr/bin/env python3
"""Re-encode a new candidate index with an existing TwoTower checkpoint."""
from __future__ import annotations

import argparse
import json
import logging
import os
import resource
import shutil
import sys
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-artifact", type=Path, required=True)
    parser.add_argument("--index-file", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def materialize_file(source: Path, target: Path) -> None:
    """Hard-link immutable files when possible and copy across filesystems."""
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def main() -> int:
    args = arguments()
    from two_tower_v2.training import (
        EMBEDDINGS_FILENAME,
        METADATA_FILENAME,
        MODEL_FILENAME,
        TOKENIZER_FILENAME,
        atomic_json,
        build_model,
        export_candidates,
        file_sha256,
        git_sha,
        resolve_device,
    )

    base_artifact = args.base_artifact.resolve()
    index_file = args.index_file.resolve()
    artifact_dir = args.artifact_dir.resolve()
    required = [base_artifact / MODEL_FILENAME, base_artifact / "manifest.json", index_file]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required input is missing: {missing}")
    if artifact_dir.exists():
        raise FileExistsError(f"Refusing to overwrite artifact: {artifact_dir}")

    temporary = artifact_dir.with_name(f".{artifact_dir.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"Temporary artifact already exists: {temporary}")
    temporary.mkdir(parents=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(temporary / "export.log", encoding="utf-8"),
        ],
    )

    started = time.perf_counter()
    checkpoint = torch.load(
        base_artifact / MODEL_FILENAME, map_location="cpu", weights_only=True
    )
    if int(checkpoint.get("version", 0)) != 2:
        raise ValueError(f"Unsupported checkpoint version: {checkpoint.get('version')}")
    model_cfg = OmegaConf.create(checkpoint["config"])
    model_cfg.paths.index_file = str(index_file)
    model_cfg.export.max_index_rows = 0

    tokenizer_source = base_artifact / TOKENIZER_FILENAME
    if tokenizer_source.is_file():
        materialize_file(tokenizer_source, temporary / TOKENIZER_FILENAME)
    inference_source = base_artifact / "inference_config.json"
    if inference_source.is_file():
        materialize_file(inference_source, temporary / inference_source.name)

    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model = build_model(model_cfg).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    candidate_report = export_candidates(
        cfg=model_cfg,
        model=model,
        artifact_dir=temporary,
        device=device,
    )
    materialize_file(base_artifact / MODEL_FILENAME, temporary / MODEL_FILENAME)

    files = [MODEL_FILENAME, EMBEDDINGS_FILENAME, METADATA_FILENAME, "export.log"]
    for optional in (TOKENIZER_FILENAME, "inference_config.json"):
        if (temporary / optional).is_file():
            files.append(optional)
    report = {
        "version": 1,
        "kind": "two_tower_candidate_reexport",
        "git_sha": git_sha(ROOT),
        "base_artifact": str(base_artifact),
        "base_manifest_sha256": file_sha256(base_artifact / "manifest.json"),
        "index_file": str(index_file),
        "index_sha256": file_sha256(index_file),
        "device": str(device),
        "seconds": time.perf_counter() - started,
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
        "candidates": candidate_report,
        "files": {
            name: {"bytes": (temporary / name).stat().st_size} for name in files
        },
    }
    atomic_json(temporary / "metrics.json", report)
    atomic_json(temporary / "manifest.json", report)
    os.replace(temporary, artifact_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
