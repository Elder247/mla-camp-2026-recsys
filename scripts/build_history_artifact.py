#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.text import normalize  # noqa: E402


def text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def build(history_path: Path, index_path: Path, output_dir: Path, top_k: int) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    history = pq.read_table(history_path).to_pydict()
    aggregated: dict[tuple[str, str, int], dict[int, list[float | int]]] = defaultdict(dict)
    for row_index, banner_value in enumerate(history["banner_id"]):
        key_type = text(history["key_type"][row_index])
        query = normalize(text(history["search_query"][row_index]))
        region_id = int(history["region_id"][row_index] or 0)
        key = (key_type, query, region_id)
        banner_id = int(banner_value)
        stats = aggregated[key].setdefault(banner_id, [0, 0.0, 0])
        stats[0] += int(history["click_count"][row_index] or 0)
        stats[1] += float(history["source_cost_sum"][row_index] or 0.0)
        stats[2] = max(stats[2], int(history["last_show_time"][row_index] or 0))

    rankings: dict[tuple[str, str, int], list[tuple[int, int, float, int]]] = {}
    candidate_ids: set[int] = set()
    for key, by_banner in aggregated.items():
        rows = [
            (banner_id, int(stats[0]), float(stats[1]), int(stats[2]))
            for banner_id, stats in by_banner.items()
        ]
        rows.sort(key=lambda row: (-row[2], -row[1], -row[3], row[0]))
        rankings[key] = rows[:top_k]
        candidate_ids.update(row[0] for row in rankings[key])

    candidate_array = pa.array(sorted(candidate_ids), type=pa.int64())
    index = pq.read_table(
        index_path,
        columns=["BannerID", "BannerTitle", "BannerText", "BannerURL", "SourceCost"],
        filters=[("BannerID", "in", candidate_array.to_pylist())],
    ).to_pydict()
    candidates = {}
    for row_index, banner_value in enumerate(index["BannerID"]):
        banner_id = int(banner_value)
        if banner_id not in candidate_ids:
            continue
        candidates[banner_id] = {
            "title": text(index["BannerTitle"][row_index]),
            "text": text(index["BannerText"][row_index]),
            "url": text(index["BannerURL"][row_index]),
            "source_cost": float(index["SourceCost"][row_index] or 0.0),
        }

    model = {
        "version": 1,
        "rankings": rankings,
        "candidates": candidates,
        "metadata": {
            "keys": len(rankings),
            "candidate_ids": len(candidate_ids),
            "candidates_with_metadata": len(candidates),
            "top_k_per_key": top_k,
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "model.pkl.gz"
    with gzip.open(output_path, "wb", compresslevel=5) as target:
        pickle.dump(model, target, protocol=pickle.HIGHEST_PROTOCOL)
    print(model["metadata"])
    print(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build compact history generator artifact")
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=200)
    args = parser.parse_args()
    build(args.history, args.index, args.output_dir, args.top_k)


if __name__ == "__main__":
    main()
