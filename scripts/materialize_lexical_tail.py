#!/usr/bin/env python3
"""Inject one low-confidence lexical safety candidate at rank 50."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
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


def read_scored_lexical(path: Path, *, top_k: int) -> dict[int, list[tuple[int, float]]]:
    parts = sorted(path.glob("part-*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No lexical candidate parts at {path}")
    grouped: dict[int, list[tuple[int, int, float]]] = defaultdict(list)
    for part in parts:
        table = pq.read_table(
            part,
            columns=["hit_log_id", "banner_id", "source_rank", "source_score"],
            filters=[("source_rank", "<=", top_k)],
        )
        for row in table.to_pylist():
            grouped[int(row["hit_log_id"])].append(
                (
                    int(row["source_rank"]),
                    int(row["banner_id"]),
                    float(row["source_score"]),
                )
            )
    return {
        hit_log_id: [(banner_id, score) for _, banner_id, score in sorted(rows)]
        for hit_log_id, rows in grouped.items()
    }


def lexical_z(rows: list[tuple[int, float]], *, depth: int) -> float:
    scores = np.asarray([score for _, score in rows[:depth]], dtype=np.float64)
    if len(scores) < 2:
        return float("inf")
    return float((scores[0] - scores.mean()) / (scores.std() + 1.0e-6))


def routed_banners(
    control: list[int],
    lexical: list[tuple[int, float]],
    *,
    z_threshold: float,
    z_depth: int,
) -> tuple[list[int], bool, float]:
    if len(control) != 50 or len(set(control)) != 50:
        raise ValueError("control ranking must contain exactly 50 unique IDs")
    z_value = lexical_z(lexical, depth=z_depth)
    if z_value > z_threshold:
        return list(control), False, z_value
    existing = set(control)
    extra = next((banner_id for banner_id, _ in lexical if banner_id not in existing), None)
    if extra is None:
        return list(control), False, z_value
    return [*control[:49], int(extra)], True, z_value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--lexical", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--banner-index", type=Path, required=True)
    parser.add_argument("--z-threshold", type=float, required=True)
    parser.add_argument("--z-depth", type=int, default=10)
    parser.add_argument("--lexical-top-k", type=int, default=100)
    parser.add_argument("--scope", choices=("offline", "full"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    control = read_ranking(args.control)
    lexical = read_scored_lexical(args.lexical, top_k=args.lexical_top_k)
    requests = read_request_parquet(args.requests)
    rows = []
    changed = 0
    z_values = []
    for request in requests:
        hit_log_id = int(request["hit_log_id"])
        banners, used, z_value = routed_banners(
            control[hit_log_id],
            lexical.get(hit_log_id, []),
            z_threshold=args.z_threshold,
            z_depth=args.z_depth,
        )
        changed += int(used)
        z_values.append(z_value)
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
    index = pq.read_table(args.banner_index, columns=["BannerID"])
    validation = validate_submission(
        args.output,
        expected_hitlog_ids={int(row["hit_log_id"]) for row in requests},
        valid_banner_ids={int(value) for value in index["BannerID"]},
        top_k=50,
        allow_short=False,
    )
    parameters = {
        "control": str(args.control),
        "lexical": str(args.lexical),
        "z_threshold": args.z_threshold,
        "z_depth": args.z_depth,
        "lexical_top_k": args.lexical_top_k,
        "route": "replace rank 50 with first lexical ID absent from control",
    }
    inputs = [
        fingerprint_file(args.control),
        *(fingerprint_file(path) for path in sorted(args.lexical.glob("part-*.parquet"))),
        fingerprint_file(args.requests),
    ]
    write_output_manifest(
        args.output,
        stage="materialize_lexical_tail",
        artifact_version="tfidf_flat_confidence_tail_v1",
        config_sha256=content_fingerprint(parameters),
        inputs=inputs,
        rows=len(rows),
        schema=str(schema),
        scope=args.scope,
    )
    finite_z = [value for value in z_values if np.isfinite(value)]
    report = {
        "status": "completed" if validation["ok"] else "validation_failed",
        "path": str(args.output),
        "parameters": parameters,
        "requests": len(rows),
        "routed_requests": changed,
        "routed_share": changed / len(rows),
        "z_summary": {
            "minimum": min(finite_z),
            "median": float(np.median(finite_z)),
            "maximum": max(finite_z),
        },
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
