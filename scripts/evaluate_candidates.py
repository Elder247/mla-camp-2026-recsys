#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mla_recsys.pipeline import MultiGeneratorPipeline  # noqa: E402


def load_examples(path: Path, max_clicks: int) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    columns = table.to_pydict()
    groups: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    click_count = 0
    for row_index in range(table.num_rows):
        if not int(columns.get("IsClick", [1] * table.num_rows)[row_index] or 0):
            continue
        if max_clicks > 0 and click_count >= max_clicks:
            break
        request_id = str(columns["SearchReqId"][row_index])
        if request_id not in groups:
            order.append(request_id)
            groups[request_id] = {
                "request_id": request_id,
                "query": columns["SearchQuery"][row_index] or "",
                "context": {
                    "region_id": columns.get("RegionID", [None] * table.num_rows)[row_index],
                    "device": columns.get("DetailedDeviceType", [None] * table.num_rows)[row_index],
                    "age": columns.get("Age", [None] * table.num_rows)[row_index],
                    "gender": columns.get("Gender", [None] * table.num_rows)[row_index],
                },
                "clicked": [],
            }
        banner_id = int(columns["BannerID"][row_index])
        clicked = groups[request_id]["clicked"]
        if any(item["banner_id"] == banner_id for item in clicked):
            continue
        clicked.append(
            {
                "banner_id": banner_id,
                "source_cost": float(columns.get("SourceCost", [0.0] * table.num_rows)[row_index] or 0.0),
            }
        )
        click_count += 1
    return [groups[request_id] for request_id in order]


def metric(records: list[dict[str, Any]], k: int) -> dict[str, Any]:
    total_cost = sum(item["source_cost"] for item in records)
    hits = [item for item in records if item["rank"] <= k]
    hit_cost = sum(item["source_cost"] for item in hits)
    return {
        "recall": len(hits) / len(records) if records else 0.0,
        "sourcecost_recall": hit_cost / total_cost if total_cost else 0.0,
        "hits": len(hits),
        "clicks": len(records),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Candidate recall and source complementarity")
    parser.add_argument("--config", default=str(ROOT / "configs/baselines.json"))
    parser.add_argument(
        "--val-file",
        default="/home/astrofimuk/workspace/tfidf_step1/data/val_clicks.parquet",
    )
    parser.add_argument("--max-clicks", type=int, default=100)
    parser.add_argument("--ks", default="50,100,500,1000,2000")
    parser.add_argument("--output")
    args = parser.parse_args()

    ks = sorted({int(value) for value in args.ks.split(",") if value})
    examples = load_examples(Path(args.val_file), args.max_clicks)
    pipeline = MultiGeneratorPipeline.from_config(args.config)
    source_records: dict[str, list[dict[str, Any]]] = {
        generator.name: [] for generator in pipeline.generators
    }
    source_records["fused"] = []
    membership = Counter()

    for request_index, example in enumerate(examples, start=1):
        rankings = pipeline.source_rankings(example)
        fused = pipeline.fuse(rankings)
        rank_maps = {
            name: {int(item["banner_id"]): rank for rank, item in enumerate(items, start=1)}
            for name, items in rankings.items()
        }
        rank_maps["fused"] = {
            int(item["banner_id"]): rank for rank, item in enumerate(fused, start=1)
        }
        for clicked in example["clicked"]:
            sources = tuple(
                sorted(name for name in rankings if clicked["banner_id"] in rank_maps[name])
            )
            membership[sources or ("miss",)] += 1
            for name, rank_map in rank_maps.items():
                source_records[name].append(
                    {
                        "rank": rank_map.get(clicked["banner_id"], 10**9),
                        "source_cost": clicked["source_cost"],
                    }
                )
        if request_index % 25 == 0:
            print(f"processed {request_index}/{len(examples)} requests", file=sys.stderr)

    report = {
        "requests": len(examples),
        "clicks": sum(len(example["clicked"]) for example in examples),
        "ks": ks,
        "metrics": {
            name: {str(k): metric(records, k) for k in ks}
            for name, records in source_records.items()
        },
        "source_membership": {
            "+".join(names): count for names, count in sorted(membership.items())
        },
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

