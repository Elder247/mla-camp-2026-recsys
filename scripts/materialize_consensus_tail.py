#!/usr/bin/env python3
"""Fuse the tail of two accepted top-50 rankings by deterministic RRF consensus."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from mla_recsys.artifacts import (  # noqa: E402
    atomic_output_path,
    atomic_write_json,
    content_fingerprint,
    fingerprint_file,
    write_output_manifest,
)
from mla_recsys.data import read_request_parquet  # noqa: E402
from mla_recsys.submission import validate_submission  # noqa: E402
from scripts.tune_top50_ensemble import read_ranking  # noqa: E402


def consensus_tail(
    control: list[int],
    alternate: list[int],
    source_costs: dict[int, float] | None = None,
    *,
    preserve_top: int,
    alternate_weight: float,
    rrf_constant: float,
    source_cost_exponent: float = 0.0,
    source_cost_scale: float = 1_000_000.0,
) -> list[int]:
    if len(control) != 50 or len(set(control)) != 50:
        raise ValueError("control ranking must contain exactly 50 unique IDs")
    if len(alternate) != 50 or len(set(alternate)) != 50:
        raise ValueError("alternate ranking must contain exactly 50 unique IDs")
    if not 0 <= preserve_top < 50:
        raise ValueError("preserve_top must be in [0, 49]")
    if not 0.0 <= alternate_weight <= 1.0:
        raise ValueError("alternate_weight must be in [0, 1]")
    if rrf_constant < 0.0:
        raise ValueError("rrf_constant must be non-negative")
    if source_cost_exponent < 0.0:
        raise ValueError("source_cost_exponent must be non-negative")
    if source_cost_scale <= 0.0:
        raise ValueError("source_cost_scale must be positive")
    source_costs = source_costs or {}

    kept = list(control[:preserve_top])
    kept_set = set(kept)
    control_rank = {banner: rank for rank, banner in enumerate(control, start=1)}
    alternate_rank = {
        banner: rank for rank, banner in enumerate(alternate, start=1)
    }
    candidates = (set(control[preserve_top:]) | set(alternate[preserve_top:])) - kept_set
    scored = []
    for banner in candidates:
        score = 0.0
        if banner in control_rank:
            score += (1.0 - alternate_weight) / (
                rrf_constant + control_rank[banner]
            )
        if banner in alternate_rank:
            score += alternate_weight / (
                rrf_constant + alternate_rank[banner]
            )
        source_cost = max(0.0, float(source_costs.get(banner, 0.0)))
        score *= (1.0 + source_cost / source_cost_scale) ** source_cost_exponent
        best_rank = min(control_rank.get(banner, 10**9), alternate_rank.get(banner, 10**9))
        scored.append((score, best_rank, banner))
    scored.sort(key=lambda item: (-item[0], item[1], item[2]))
    result = kept + [banner for _, _, banner in scored[: 50 - preserve_top]]
    if len(result) != 50 or len(set(result)) != 50:
        raise RuntimeError("consensus tail did not produce 50 unique IDs")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--alternate", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--banner-index", type=Path, required=True)
    parser.add_argument("--preserve-top", type=int, default=10)
    parser.add_argument("--alternate-weight", type=float, default=0.5)
    parser.add_argument("--rrf-constant", type=float, default=30.0)
    parser.add_argument("--source-cost-exponent", type=float, default=0.0)
    parser.add_argument("--source-cost-scale", type=float, default=1_000_000.0)
    parser.add_argument("--scope", choices=("offline", "full"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    started = time.monotonic()
    control = read_ranking(args.control)
    alternate = read_ranking(args.alternate)
    requests = read_request_parquet(args.requests)
    index = pq.read_table(args.banner_index, columns=["BannerID", "SourceCost"])
    index_values = index.to_pydict()
    source_costs = {
        int(banner): float(cost or 0.0)
        for banner, cost in zip(index_values["BannerID"], index_values["SourceCost"])
    }
    rows = []
    changed = 0
    overlaps = []
    for request in requests:
        hit_log_id = int(request["hit_log_id"])
        control_order = control[hit_log_id]
        alternate_order = alternate[hit_log_id]
        banners = consensus_tail(
            control_order,
            alternate_order,
            source_costs,
            preserve_top=args.preserve_top,
            alternate_weight=args.alternate_weight,
            rrf_constant=args.rrf_constant,
            source_cost_exponent=args.source_cost_exponent,
            source_cost_scale=args.source_cost_scale,
        )
        changed += int(banners != control_order)
        overlaps.append(len(set(control_order) & set(alternate_order)))
        rows.append({"HitLogID": hit_log_id, "BannerID": banners})
    rows.sort(key=lambda row: int(row["HitLogID"]))

    schema = pa.schema(
        [
            pa.field("HitLogID", pa.uint64(), nullable=False),
            pa.field("BannerID", pa.list_(pa.uint64()), nullable=False),
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with atomic_output_path(args.output) as temporary:
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), temporary, compression="zstd")

    validation = validate_submission(
        args.output,
        expected_hitlog_ids={int(row["hit_log_id"]) for row in requests},
        valid_banner_ids=set(source_costs),
        top_k=50,
        allow_short=False,
    )
    parameters = {
        "control": str(args.control),
        "alternate": str(args.alternate),
        "preserve_top": args.preserve_top,
        "alternate_weight": args.alternate_weight,
        "rrf_constant": args.rrf_constant,
        "source_cost_exponent": args.source_cost_exponent,
        "source_cost_scale": args.source_cost_scale,
        "route": "preserve control prefix and fill tail by two-ranking RRF consensus",
    }
    write_output_manifest(
        args.output,
        stage="materialize_consensus_tail",
        artifact_version="accepted_top50_consensus_tail_v1",
        config_sha256=content_fingerprint(parameters),
        inputs=[fingerprint_file(args.control), fingerprint_file(args.alternate)],
        rows=len(rows),
        schema=str(schema),
        scope=args.scope,
    )
    report = {
        "status": "completed" if validation["ok"] else "validation_failed",
        "path": str(args.output),
        "parameters": parameters,
        "requests": len(rows),
        "changed_requests": changed,
        "changed_share": changed / len(rows),
        "mean_top50_overlap": sum(overlaps) / len(overlaps),
        "validation": validation,
        "artifact": fingerprint_file(args.output),
        "wall_seconds": time.monotonic() - started,
    }
    atomic_write_json(
        args.report or args.output.with_name(args.output.name + ".report.json"),
        report,
    )
    print(json.dumps(report, indent=2))
    return 0 if validation["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
