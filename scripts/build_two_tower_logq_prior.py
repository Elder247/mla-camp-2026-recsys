#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import logging
import os
import resource
import sys
import time
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an exact banner sampling prior for TwoTower logQ"
    )
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = arguments()
    cfg = OmegaConf.load(args.config.resolve())
    sys.path.insert(0, str(cfg.paths.step2_root))

    from two_tower_v2.data import YtTableSource
    from two_tower_v2.training import atomic_json, file_sha256, git_sha

    prior_dir = Path(str(cfg.paths.prior_dir))
    if prior_dir.exists() and any(prior_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite logQ prior: {prior_dir}")
    prior_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(prior_dir / "build.log", encoding="utf-8"),
        ],
    )
    resolved = OmegaConf.to_yaml(cfg, resolve=True)
    (prior_dir / "config.resolved.yaml").write_text(resolved, encoding="utf-8")

    source = YtTableSource(
        str(cfg.paths.train_table),
        str(cfg.paths.proxy),
        fields=("banner_id",),
    )
    counts: Counter[int] = Counter()
    rows_seen = 0
    invalid_rows = 0
    progress_every = int(cfg.build.progress_every_rows)
    started = time.perf_counter()
    for row in source.rows():
        rows_seen += 1
        banner_id = int(row.get("banner_id") or 0)
        if banner_id > 0:
            counts[banner_id] += 1
        else:
            invalid_rows += 1
        if progress_every > 0 and rows_seen % progress_every == 0:
            logging.info(
                "logQ prior rows=%s unique_items=%s",
                f"{rows_seen:,}",
                f"{len(counts):,}",
            )
    if rows_seen != source.row_count:
        raise RuntimeError(
            f"Incomplete logQ source read: {rows_seen} != {source.row_count}"
        )
    if not counts:
        raise RuntimeError("logQ prior has no positive banner ids")

    prior_name = str(cfg.build.output_name)
    prior_file = prior_dir / prior_name
    temporary = prior_file.with_suffix(prior_file.suffix + ".tmp")
    items = sorted(counts.items())
    table = pa.table(
        {
            "banner_id": pa.array(
                (banner_id for banner_id, _ in items), type=pa.uint64()
            ),
            "count": pa.array((count for _, count in items), type=pa.uint64()),
        }
    )
    pq.write_table(
        table,
        temporary,
        compression=str(cfg.build.compression),
        row_group_size=int(cfg.build.row_group_size),
    )
    os.replace(temporary, prior_file)
    elapsed = time.perf_counter() - started
    total_count = int(sum(counts.values()))
    metrics = {
        "rows_seen": rows_seen,
        "valid_rows": total_count,
        "invalid_rows": invalid_rows,
        "unique_items": len(counts),
        "seconds": elapsed,
        "rows_per_second": rows_seen / max(elapsed, 1.0e-9),
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        * 1024,
    }
    manifest = {
        "version": 1,
        "kind": "global_banner_frequency",
        "solution": str(cfg.experiment.name),
        "git_sha": git_sha(ROOT),
        "config_sha256": hashlib.sha256(resolved.encode("utf-8")).hexdigest(),
        "source": {
            "cluster": str(cfg.paths.proxy),
            "table": source.table,
            "row_count": source.row_count,
            "scope": str(cfg.build.scope),
        },
        "unique_items": len(counts),
        "total_count": total_count,
        "invalid_rows": invalid_rows,
        "file": {
            "name": prior_file.name,
            "bytes": prior_file.stat().st_size,
            "sha256": file_sha256(prior_file),
        },
        "metrics": metrics,
    }
    atomic_json(prior_dir / "metrics.json", metrics)
    atomic_json(prior_dir / "manifest.json", manifest)
    logging.info(
        "logQ prior completed rows=%s unique_items=%s seconds=%.1f",
        f"{rows_seen:,}",
        f"{len(counts):,}",
        elapsed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
