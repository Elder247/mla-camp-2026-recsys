#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluate_candidates import load_examples  # noqa: E402
from mla_recsys.features import extract_feature_rows, feature_names  # noqa: E402
from mla_recsys.pipeline import MultiGeneratorPipeline  # noqa: E402


def is_validation(request_id: str) -> bool:
    digest = hashlib.sha1(request_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little") % 5 == 0


def add_group(
    target: dict[str, list[Any]],
    *,
    group_id: int,
    example: dict[str, Any],
    candidates: list[dict[str, Any]],
    generator_names: list[str],
) -> bool:
    clicked = {int(item["banner_id"]): item for item in example["clicked"]}
    labels = [
        1.0 + math.log1p(max(0.0, float(clicked[int(candidate["banner_id"])]["source_cost"])) / 1_000_000.0)
        if int(candidate["banner_id"]) in clicked
        else 0.0
        for candidate in candidates
    ]
    if not any(label > 0 for label in labels):
        return False
    rows = extract_feature_rows(example, candidates, generator_names)
    target["features"].extend(rows)
    target["labels"].extend(labels)
    target["groups"].extend([group_id] * len(rows))
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a simple CatBoost second-stage ranker")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--val-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-clicks", type=int, default=5000)
    parser.add_argument("--candidate-pool", type=int, default=500)
    parser.add_argument("--iterations", type=int, default=600)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.07)
    parser.add_argument("--task-type", choices=("CPU", "GPU"), default="GPU")
    args = parser.parse_args()

    from catboost import CatBoostRanker, Pool

    examples = load_examples(args.val_file, args.max_clicks)
    pipeline = MultiGeneratorPipeline.from_config(args.config)
    generator_names = [generator.name for generator in pipeline.generators]
    datasets = {
        "train": {"features": [], "labels": [], "groups": []},
        "validation": {"features": [], "labels": [], "groups": []},
    }
    stats = {
        "requests": len(examples),
        "train_groups": 0,
        "validation_groups": 0,
        "missed_groups": 0,
    }

    for request_index, example in enumerate(examples, start=1):
        rankings = pipeline.source_rankings(example)
        full_candidates = pipeline.fuse(
            rankings,
            max_candidates=max(args.candidate_pool, int(pipeline.fusion.get("max_candidates", 2000))),
        )
        clicked_ids = {int(item["banner_id"]) for item in example["clicked"]}
        selected = list(full_candidates[: args.candidate_pool])
        selected_ids = {int(candidate["banner_id"]) for candidate in selected}
        selected.extend(
            candidate
            for candidate in full_candidates[args.candidate_pool :]
            if int(candidate["banner_id"]) in clicked_ids
            and int(candidate["banner_id"]) not in selected_ids
        )
        split = "validation" if is_validation(example["request_id"]) else "train"
        added = add_group(
            datasets[split],
            group_id=request_index,
            example=example,
            candidates=selected,
            generator_names=generator_names,
        )
        if added:
            stats[f"{split}_groups"] += 1
        else:
            stats["missed_groups"] += 1
        if request_index % 100 == 0:
            print(f"features {request_index}/{len(examples)}", file=sys.stderr)

    if not stats["train_groups"] or not stats["validation_groups"]:
        raise RuntimeError(f"Not enough positive groups after split: {stats}")
    names = feature_names(generator_names)
    train_pool = Pool(
        datasets["train"]["features"],
        label=datasets["train"]["labels"],
        group_id=datasets["train"]["groups"],
        feature_names=names,
    )
    validation_pool = Pool(
        datasets["validation"]["features"],
        label=datasets["validation"]["labels"],
        group_id=datasets["validation"]["groups"],
        feature_names=names,
    )
    model = CatBoostRanker(
        loss_function="YetiRankPairwise",
        eval_metric="NDCG:top=50",
        iterations=args.iterations,
        depth=args.depth,
        learning_rate=args.learning_rate,
        l2_leaf_reg=5.0,
        random_seed=2026,
        task_type=args.task_type,
        devices="0" if args.task_type == "GPU" else None,
        verbose=50,
        allow_writing_files=False,
    )
    model.fit(
        train_pool,
        eval_set=validation_pool,
        early_stopping_rounds=75,
        use_best_model=True,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(args.output_dir / "ranker.cbm"))
    shutil.copy2(args.config, args.output_dir / "config.json")
    metadata = {
        "version": 1,
        "generator_names": generator_names,
        "feature_names": names,
        "candidate_pool": args.candidate_pool,
        "split": "sha1(request_id) % 5; 20% validation",
        "stats": stats,
        "best_iteration": model.get_best_iteration(),
        "best_score": model.get_best_score(),
    }
    (args.output_dir / "ranker.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
