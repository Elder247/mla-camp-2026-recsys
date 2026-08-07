#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mla_recsys.artifacts import atomic_write_json  # noqa: E402
from mla_recsys.candidate_cache import enabled_sources, load_source  # noqa: E402
from mla_recsys.command import load_stage_context  # noqa: E402
from mla_recsys.data import read_request_parquet, request_example, stable_partition  # noqa: E402
from mla_recsys.fusion import fuse_rankings  # noqa: E402


def cached_top(
    run_path: Path, split: str, partition: int, *, top_k: int
) -> dict[str, list[int]]:
    path = run_path / "candidates" / split / "merged" / f"part-{partition:05d}.parquet"
    rows = pq.read_table(
        path,
        columns=["request_id", "banner_id", "pre_rank"],
        filters=[("pre_rank", "<=", top_k)],
    ).to_pylist()
    grouped: dict[str, list[tuple[int, int]]] = {}
    for row in rows:
        grouped.setdefault(str(row["request_id"]), []).append(
            (int(row["pre_rank"]), int(row["banner_id"]))
        )
    return {
        request_id: [banner_id for _, banner_id in sorted(values)]
        for request_id, values in grouped.items()
    }


def main() -> int:
    context = load_stage_context("Validate sampled cached and direct retrieval parity")
    cfg = context.cfg
    mode = str(cfg.runtime.mode)
    splits = ("full_train", "test") if mode == "full" else ("train", "holdout")
    requests_per_split = int(cfg.data.smoke_requests_per_split)
    top_k = int(cfg.evaluation.submission_top_k)
    sources = enabled_sources(cfg)
    specs = {source: load_source(cfg, source) for source in sources}
    weights = {source: float(cfg.candidates.generators[source].weight) for source in sources}
    quotas = {source: int(cfg.candidates.generators[source].quota) for source in sources}
    partitions = int(cfg.data.partition_count)
    mismatches = []
    checked = 0
    checked_by_split: dict[str, int] = {}
    for split in splits:
        cache_by_partition = {
            partition: cached_top(
                context.store.path, split, partition, top_k=top_k
            )
            for partition in range(partitions)
        }
        requests = read_request_parquet(
            context.store.path / "data" / f"{split}_requests.parquet"
        )[:requests_per_split]
        checked_by_split[split] = len(requests)
        for request in requests:
            rankings = {
                source: specs[source].generator.rank(request_example(request))
                for source in sources
            }
            direct = fuse_rankings(
                rankings,
                weights=weights,
                quotas=quotas,
                rrf_constant=float(cfg.candidates.rrf_constant),
                max_candidates=int(cfg.candidates.union_max_candidates),
            )
            direct_ids = [int(row["banner_id"]) for row in direct[:top_k]]
            partition = stable_partition(str(request["request_id"]), partitions)
            cached_ids = cache_by_partition[partition].get(str(request["request_id"]), [])
            checked += 1
            if direct_ids != cached_ids:
                mismatches.append(
                    {
                        "split": split,
                        "request_id": str(request["request_id"]),
                        "direct": direct_ids[:10],
                        "cached": cached_ids[:10],
                    }
                )
    report = {
        "mode": mode,
        "top_k": top_k,
        "requests_per_split_limit": requests_per_split,
        "checked_by_split": checked_by_split,
        "checked_requests": checked,
        "mismatch_count": len(mismatches),
        "mismatch_sample": mismatches[:5],
        "ok": not mismatches,
    }
    atomic_write_json(context.store.path / "metrics" / "cache_parity.json", report)
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
