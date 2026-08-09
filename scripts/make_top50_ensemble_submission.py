#!/usr/bin/env python3
"""Materialize a strict submission from cached top-k rankings."""
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
from mla_recsys.rank_blend import value_geometric_from_base_order  # noqa: E402
from mla_recsys.submission import validate_submission  # noqa: E402
from scripts.tune_top50_ensemble import fuse_rankings, read_ranking  # noqa: E402


def ensemble_rows(
    requests: list[dict],
    sources: list[dict[int, list[int]]],
    weights: tuple[float, ...],
    *,
    rrf_constant: float,
    exponent: float,
    rerank_top_n: int,
    source_costs: dict[int, float],
) -> list[dict]:
    rows = []
    for request in requests:
        hit_log_id = int(request["hit_log_id"])
        base = fuse_rankings(
            [source.get(hit_log_id, []) for source in sources],
            weights,
            rrf_constant=rrf_constant,
            hit_log_id=hit_log_id,
            source_costs=source_costs,
        )
        ordered = value_geometric_from_base_order(
            base,
            source_cost_scale=1_000_000.0,
            exponent=exponent,
            rerank_top_n=rerank_top_n,
        )
        banners = [int(value[2]) for value in ordered[:50]]
        if len(banners) != 50 or len(set(banners)) != 50:
            raise RuntimeError(f"Invalid ensemble top-50 for HitLogID={hit_log_id}")
        rows.append({"HitLogID": hit_log_id, "BannerID": banners})
    return rows


def input_fingerprints(path: Path) -> list[dict]:
    if path.is_file():
        return [fingerprint_file(path)]
    parts = sorted(path.glob("part-*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No ranking parquet found at {path}")
    return [fingerprint_file(part) for part in parts]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--weight", type=float, action="append", required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--banner-index", type=Path, required=True)
    parser.add_argument("--candidate-top-k", type=int, default=100)
    parser.add_argument("--rrf-constant", type=float, required=True)
    parser.add_argument("--exponent", type=float, required=True)
    parser.add_argument("--rerank-top-n", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if len(args.input) != len(args.weight):
        parser.error("--input and --weight counts must match")
    if any(weight < 0.0 for weight in args.weight) or sum(args.weight) <= 0.0:
        parser.error("weights must be non-negative with a positive sum")
    total = sum(args.weight)
    weights = tuple(float(weight) / total for weight in args.weight)
    started = time.monotonic()
    sources = [
        read_ranking(path, candidate_top_k=args.candidate_top_k)
        for path in args.input
    ]
    requests = read_request_parquet(args.requests)
    index = pq.read_table(args.banner_index, columns=["BannerID", "SourceCost"])
    index_values = index.to_pydict()
    source_costs = {
        int(banner): float(cost or 0.0)
        for banner, cost in zip(index_values["BannerID"], index_values["SourceCost"])
    }
    rows = ensemble_rows(
        requests,
        sources,
        weights,
        rrf_constant=args.rrf_constant,
        exponent=args.exponent,
        rerank_top_n=args.rerank_top_n,
        source_costs=source_costs,
    )
    schema = pa.schema(
        [
            pa.field("HitLogID", pa.uint64(), nullable=False),
            pa.field("BannerID", pa.list_(pa.uint64()), nullable=False),
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with atomic_output_path(args.output) as temporary:
        pq.write_table(
            pa.Table.from_pylist(rows, schema=schema),
            temporary,
            compression="zstd",
        )
    parameters = {
        "input_paths": [str(path) for path in args.input],
        "weights": weights,
        "candidate_top_k": args.candidate_top_k,
        "rrf_constant": args.rrf_constant,
        "source_cost_scale": 1_000_000.0,
        "exponent": args.exponent,
        "rerank_top_n": args.rerank_top_n,
    }
    inputs = [
        fingerprint
        for path in args.input
        for fingerprint in input_fingerprints(path)
    ]
    write_output_manifest(
        args.output,
        stage="make_top50_ensemble_submission",
        artifact_version="cached_top50_rrf_geometry_v1",
        config_sha256=content_fingerprint(parameters),
        inputs=inputs,
        rows=len(rows),
        schema=str(schema),
        scope="full",
    )
    validation = validate_submission(
        args.output,
        expected_hitlog_ids={int(row["hit_log_id"]) for row in requests},
        valid_banner_ids=set(source_costs),
        top_k=50,
        allow_short=False,
    )
    report = {
        "status": "completed" if validation["ok"] else "validation_failed",
        "path": str(args.output),
        "parameters": parameters,
        "validation": validation,
        "artifact": fingerprint_file(args.output),
        "wall_seconds": time.monotonic() - started,
    }
    report_path = args.report or args.output.with_name(args.output.name + ".report.json")
    atomic_write_json(report_path, report)
    print(json.dumps(report, indent=2))
    return 0 if validation["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
