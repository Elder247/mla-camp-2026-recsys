#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mla_recsys.artifacts import atomic_write_json  # noqa: E402
from mla_recsys.candidate_cache import source_part_path  # noqa: E402
from mla_recsys.command import load_stage_context, require_choice  # noqa: E402
from mla_recsys.data import read_request_parquet  # noqa: E402
from mla_recsys.metrics import MISS_RANK, recall_metrics, records_from_found, truth_pairs  # noqa: E402


def source_found(
    *,
    run_path: Path,
    split: str,
    source: str,
    partitions: int,
    truth: dict[tuple[str, int], float],
) -> tuple[dict[tuple[str, int], int], dict[str, set[int]]]:
    import pyarrow.parquet as pq

    found: dict[tuple[str, int], int] = {}
    top50: dict[str, set[int]] = defaultdict(set)
    for partition in range(partitions):
        path = source_part_path(run_path, split, source, partition)
        if not path.is_file():
            raise FileNotFoundError(f"Missing source partition: {path}")
        rows = pq.read_table(
            path,
            columns=["request_id", "banner_id", "source_rank"],
        ).to_pylist()
        for row in rows:
            request_id = str(row["request_id"])
            banner_id = int(row["banner_id"])
            rank = int(row["source_rank"])
            if rank <= 50:
                top50[request_id].add(banner_id)
            pair = (request_id, banner_id)
            if pair in truth:
                found[pair] = min(found.get(pair, MISS_RANK), rank)
    return found, top50


def main() -> int:
    context = load_stage_context(
        "Evaluate one candidate source against a cached temporal baseline",
        extra_keys=("cg", "split", "baseline_run", "baseline_source"),
    )
    cfg = context.cfg
    source = require_choice(context, "cg", cfg.candidates.generators.keys())
    split = require_choice(context, "split", ("train", "holdout"))
    baseline_run = Path(context.values["baseline_run"])
    baseline_source = str(context.values["baseline_source"])
    requests = read_request_parquet(context.store.path / "data" / f"{split}_requests.parquet")
    truth = truth_pairs(requests)
    partitions = int(cfg.data.partition_count)
    current, current_top50 = source_found(
        run_path=context.store.path,
        split=split,
        source=source,
        partitions=partitions,
        truth=truth,
    )
    baseline, baseline_top50 = source_found(
        run_path=baseline_run,
        split=split,
        source=baseline_source,
        partitions=partitions,
        truth=truth,
    )
    cutoffs = [int(value) for value in cfg.evaluation.cutoffs]
    union = {
        pair: min(current.get(pair, MISS_RANK), baseline.get(pair, MISS_RANK))
        for pair in truth
        if pair in current or pair in baseline
    }
    total_cost = sum(truth.values())
    current_pairs = set(current)
    baseline_pairs = set(baseline)
    new_only = current_pairs - baseline_pairs
    baseline_only = baseline_pairs - current_pairs
    request_ids = {str(request["request_id"]) for request in requests}
    jaccards = []
    for request_id in request_ids:
        left = current_top50.get(request_id, set())
        right = baseline_top50.get(request_id, set())
        combined = left | right
        jaccards.append(len(left & right) / len(combined) if combined else 0.0)
    report = {
        "version": 1,
        "run_id": context.store.run_id,
        "split": split,
        "source": source,
        "baseline_run": str(baseline_run),
        "baseline_source": baseline_source,
        "requests": len(requests),
        "clicks": len(truth),
        "metrics": {
            "current": recall_metrics(records_from_found(truth, current), cutoffs),
            "baseline": recall_metrics(records_from_found(truth, baseline), cutoffs),
            "oracle_union": recall_metrics(records_from_found(truth, union), cutoffs),
        },
        "complementarity": {
            "new_only_hits": len(new_only),
            "new_only_sourcecost": sum(truth[pair] for pair in new_only),
            "new_only_sourcecost_share": (
                sum(truth[pair] for pair in new_only) / total_cost if total_cost else 0.0
            ),
            "baseline_only_hits": len(baseline_only),
            "baseline_only_sourcecost": sum(truth[pair] for pair in baseline_only),
            "baseline_only_sourcecost_share": (
                sum(truth[pair] for pair in baseline_only) / total_cost if total_cost else 0.0
            ),
            "mean_jaccard_top50": sum(jaccards) / len(jaccards) if jaccards else 0.0,
        },
    }
    output = context.store.path / "metrics" / f"retrieval_{split}_{source}.json"
    atomic_write_json(output, report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

