#!/usr/bin/env python3
"""Attach a validated train-prior inference transform to a TwoTower artifact."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))



def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--prior-dir", type=Path, required=True)
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--unseen-count", type=float, default=1.0)
    parser.add_argument("--rerank-top-n", type=int, default=100)
    args = parser.parse_args()
    if not -1.0 <= args.alpha <= 1.0:
        raise ValueError("alpha must be in [-1, 1]")
    if args.unseen_count <= 0.0:
        raise ValueError("unseen-count must be positive")
    if args.rerank_top_n <= 0:
        raise ValueError("rerank-top-n must be positive")
    if not args.artifact_dir.is_dir():
        raise FileNotFoundError(f"artifact directory is missing: {args.artifact_dir}")
    target = args.artifact_dir / "inference_config.json"
    if target.exists():
        raise FileExistsError(f"refusing to overwrite inference config: {target}")
    prior_manifest_path = args.prior_dir / "manifest.json"
    prior_manifest = json.loads(prior_manifest_path.read_text(encoding="utf-8"))
    if prior_manifest.get("kind") != "global_banner_frequency":
        raise ValueError("unexpected prior kind")
    prior_file = args.prior_dir / str(prior_manifest["file"]["name"])
    if file_sha256(prior_file) != str(prior_manifest["file"]["sha256"]):
        raise ValueError("prior data SHA-256 mismatch")
    config = {
        "version": 1,
        "kind": "global_banner_logq_restore",
        "alpha": float(args.alpha),
        "unseen_count": float(args.unseen_count),
        "rerank_top_n": int(args.rerank_top_n),
        "prior_dir": str(args.prior_dir),
        "prior_manifest_sha256": file_sha256(prior_manifest_path),
        "prior_source": prior_manifest["source"],
        "selection": {
            "method": "early_temporal_tune_late_temporal_validation",
            "metric": "SourceCost Recall@50",
        },
    }
    atomic_json(target, config)
    root_manifest_path = args.artifact_dir / "manifest.json"
    if root_manifest_path.is_file():
        root_manifest = json.loads(root_manifest_path.read_text(encoding="utf-8"))
        root_manifest["inference"] = config
        root_manifest.setdefault("files", {})[target.name] = {
            "bytes": target.stat().st_size,
            "sha256": file_sha256(target),
        }
        atomic_json(root_manifest_path, root_manifest)
    print(json.dumps(config, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
