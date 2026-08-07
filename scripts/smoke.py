#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mla_recsys.pipeline import MultiGeneratorPipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/baselines.json"))
    parser.add_argument("--query", default="купить авиабилеты москва")
    parser.add_argument("--region-id", type=int, default=213)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    pipeline = MultiGeneratorPipeline.from_config(args.config)
    example = {
        "request_id": "smoke",
        "query": args.query,
        "context": {"region_id": args.region_id},
    }
    rankings = pipeline.source_rankings(example)
    fused = pipeline.fuse(rankings)
    summary = {
        "query": args.query,
        "sources": {name: len(items) for name, items in rankings.items()},
        "union": len({int(item["banner_id"]) for items in rankings.values() for item in items}),
        "fused": len(fused),
        "top": [
            {
                "banner_id": item["banner_id"],
                "score": item["score"],
                "sources": item["sources"],
                "title": item.get("title", ""),
            }
            for item in fused[: args.top_k]
        ],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

