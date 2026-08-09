#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
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


def rerank_rows(
    values: list[tuple[float, int, int, int, float]],
    *,
    source_cost_scale: float,
    exponent: float,
    rerank_top_n: int,
) -> list[tuple[float, int, int, int, float]]:
    base = sorted(values, key=lambda value: (value[1], value[2]))
    return value_geometric_from_base_order(
        base,
        source_cost_scale=source_cost_scale,
        exponent=exponent,
        rerank_top_n=rerank_top_n,
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Materialize a validated direct TwoTower top-50 submission"
    )
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--source", default="two_tower_v2")
    parser.add_argument("--exponent", type=float, required=True)
    parser.add_argument("--rerank-top-n", type=int, required=True)
    parser.add_argument("--source-cost-scale", type=float, default=1_000_000.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    metadata = pq.read_table(
        args.artifact / "candidate_metadata.parquet",
        columns=["banner_id", "source_cost"],
    ).to_pydict()
    costs = {
        int(banner): float(cost or 0.0)
        for banner, cost in zip(metadata["banner_id"], metadata["source_cost"])
    }
    grouped: dict[str, list[tuple[float, int, int, int, float]]] = defaultdict(list)
    candidate_dir = args.run / "candidates" / "test" / args.source
    candidate_inputs = []
    for path in sorted(candidate_dir.glob("part-*.parquet")):
        candidate_inputs.append(fingerprint_file(path))
        rows = pq.read_table(
            path,
            columns=[
                "request_id",
                "hit_log_id",
                "banner_id",
                "source_rank",
                "source_score",
            ],
        ).to_pylist()
        for row in rows:
            banner = int(row["banner_id"])
            grouped[str(row["request_id"])].append(
                (
                    float(row["source_score"]),
                    int(row["source_rank"]),
                    banner,
                    int(row["hit_log_id"]),
                    costs.get(banner, 0.0),
                )
            )
    requests = read_request_parquet(args.run / "data" / "test_requests.parquet")
    rows = []
    for request in requests:
        request_id = str(request["request_id"])
        ordered = rerank_rows(
            grouped[request_id],
            source_cost_scale=args.source_cost_scale,
            exponent=args.exponent,
            rerank_top_n=args.rerank_top_n,
        )
        banners = [int(value[2]) for value in ordered[:50]]
        if len(banners) != 50 or len(set(banners)) != 50:
            raise RuntimeError(f"Invalid top-50 for request {request_id}")
        rows.append({"HitLogID": int(request["hit_log_id"]), "BannerID": banners})
    schema = pa.schema(
        [
            pa.field("HitLogID", pa.uint64(), nullable=False),
            pa.field("BannerID", pa.list_(pa.uint64()), nullable=False),
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with atomic_output_path(args.output) as temporary:
        pq.write_table(
            pa.Table.from_pylist(rows, schema=schema), temporary, compression="zstd"
        )
    parameters = {
        "source": args.source,
        "source_cost_scale": args.source_cost_scale,
        "exponent": args.exponent,
        "rerank_top_n": args.rerank_top_n,
    }
    write_output_manifest(
        args.output,
        stage="make_two_tower_submission",
        artifact_version="direct_two_tower_geometry_top50_v1",
        config_sha256=content_fingerprint(parameters),
        inputs=[*candidate_inputs, fingerprint_file(args.artifact / "model.pt")],
        rows=len(rows),
        schema=str(schema),
        scope="full",
    )
    validation = validate_submission(
        args.output,
        expected_hitlog_ids={int(row["hit_log_id"]) for row in requests},
        valid_banner_ids=set(costs),
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
    atomic_write_json(args.run / "metrics" / "two_tower_submission.json", report)
    print(json.dumps(report, indent=2))
    return 0 if validation["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
