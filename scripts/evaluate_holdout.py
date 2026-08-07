#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluate_candidates import load_examples, metric  # noqa: E402
from mla_recsys.pipeline import MultiGeneratorPipeline  # noqa: E402
from mla_recsys.ranker import load_ranker, rerank  # noqa: E402


def collect_records(
    clicked: list[dict[str, Any]], rank_map: dict[int, int]
) -> list[dict[str, Any]]:
    return [
        {
            "rank": rank_map.get(int(item["banner_id"]), 10**9),
            "source_cost": float(item["source_cost"]),
        }
        for item in clicked
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare RRF and CatBoost on untouched requests")
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--val-file", type=Path, required=True)
    parser.add_argument("--train-clicks", type=int, default=5000)
    parser.add_argument("--all-clicks", type=int, default=10000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    train_examples = load_examples(args.val_file, args.train_clicks)
    train_request_ids = {example["request_id"] for example in train_examples}
    all_examples = load_examples(args.val_file, args.all_clicks)
    holdout = [
        example for example in all_examples if example["request_id"] not in train_request_ids
    ]
    config_path = args.artifact_dir / "config.json"
    pipeline = MultiGeneratorPipeline.from_config(config_path)
    ranker = load_ranker(args.artifact_dir)
    if ranker is None:
        raise FileNotFoundError(f"No ranker in {args.artifact_dir}")

    records = {"rrf": [], "catboost": []}
    candidate_pool = int(ranker["metadata"].get("candidate_pool", 500))
    for request_index, example in enumerate(holdout, start=1):
        rankings = pipeline.source_rankings(example)
        fused = pipeline.fuse(
            rankings,
            max_candidates=max(candidate_pool, int(pipeline.fusion.get("max_candidates", 2200))),
        )
        rrf_rank_map = {
            int(candidate["banner_id"]): rank
            for rank, candidate in enumerate(fused, start=1)
        }
        reranked = rerank(ranker, example, fused[:candidate_pool])
        catboost_rank_map = {
            int(candidate["banner_id"]): rank
            for rank, candidate in enumerate(reranked, start=1)
        }
        records["rrf"].extend(collect_records(example["clicked"], rrf_rank_map))
        records["catboost"].extend(collect_records(example["clicked"], catboost_rank_map))
        if request_index % 100 == 0:
            print(f"holdout {request_index}/{len(holdout)}", file=sys.stderr)

    report = {
        "requests": len(holdout),
        "clicks": sum(len(example["clicked"]) for example in holdout),
        "excluded_train_request_ids": len(train_request_ids),
        "metrics": {
            model_name: {
                str(k): metric(model_records, k) for k in (1, 5, 10, 20, 50, 500)
            }
            for model_name, model_records in records.items()
        },
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
