#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluate_candidates import load_examples, metric  # noqa: E402
from mla_recsys.fusion import fuse_rankings  # noqa: E402
from mla_recsys.pipeline import MultiGeneratorPipeline  # noqa: E402


def parse_grid(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item]


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune weighted RRF on cached source rankings")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--val-file", type=Path, required=True)
    parser.add_argument("--max-clicks", type=int, default=100)
    parser.add_argument("--tfidf", default="0.5,0.75,1.0")
    parser.add_argument("--two-tower", default="0.75,1.0,1.25")
    parser.add_argument("--history", default="0.75,1.0,1.25,1.5,2.0")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    examples = load_examples(args.val_file, args.max_clicks)
    pipeline = MultiGeneratorPipeline.from_config(args.config)
    quotas = {generator.name: generator.quota for generator in pipeline.generators}
    cached: list[tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]] = []
    for request_index, example in enumerate(examples, start=1):
        cached.append((pipeline.source_rankings(example), example["clicked"]))
        if request_index % 25 == 0:
            print(f"cached {request_index}/{len(examples)}", file=sys.stderr)

    results = []
    for tfidf_weight, tower_weight, history_weight in itertools.product(
        parse_grid(args.tfidf), parse_grid(args.two_tower), parse_grid(args.history)
    ):
        weights = {
            "tfidf": tfidf_weight,
            "two_tower_fps": tower_weight,
            "history": history_weight,
        }
        records = []
        for rankings, clicked in cached:
            fused = fuse_rankings(
                rankings,
                weights=weights,
                quotas=quotas,
                rrf_constant=float(pipeline.fusion.get("rrf_constant", 60.0)),
                max_candidates=500,
            )
            rank_map = {
                int(candidate["banner_id"]): rank
                for rank, candidate in enumerate(fused, start=1)
            }
            records.extend(
                {
                    "rank": rank_map.get(int(item["banner_id"]), 10**9),
                    "source_cost": float(item["source_cost"]),
                }
                for item in clicked
            )
        results.append(
            {
                "weights": weights,
                "metrics": {str(k): metric(records, k) for k in (50, 100, 500)},
            }
        )
    results.sort(
        key=lambda item: (
            -item["metrics"]["50"]["sourcecost_recall"],
            -item["metrics"]["50"]["recall"],
            -item["metrics"]["100"]["sourcecost_recall"],
        )
    )
    report = {"requests": len(examples), "top": results[: args.top]}
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
