#!/usr/bin/env python3
"""Wait for two full runs and materialize their validated cross-pool top-50."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from omegaconf import OmegaConf

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
from scripts.tune_cross_pool_ensemble import fuse_orders, ranked_pool  # noqa: E402


def completed_run(path: Path) -> bool:
    result = path / "result.json"
    return result.is_file() and json.loads(result.read_text(encoding="utf-8")).get(
        "status"
    ) == "completed"


def submission_rows(
    requests: list[dict],
    old: dict[str, list[tuple[int, int, float]]],
    new: dict[str, list[tuple[int, int, float]]],
    *,
    new_weight: float,
    rrf_constant: float,
    exponent: float,
    rerank_top_n: int,
) -> list[dict]:
    if set(old) != set(new):
        raise ValueError("Full cross-pool runs cover different requests")
    rows = []
    for request in requests:
        request_id = str(request["request_id"])
        if request_id not in old:
            raise KeyError(f"Request is absent from ranked pools: {request_id}")
        base = fuse_orders(
            old[request_id],
            new[request_id],
            new_weight=new_weight,
            rrf_constant=rrf_constant,
        )
        ordered = value_geometric_from_base_order(
            base,
            source_cost_scale=1_000_000.0,
            exponent=exponent,
            rerank_top_n=rerank_top_n,
        )
        banners = [int(value[2]) for value in ordered[:50]]
        if len(banners) != 50 or len(set(banners)) != 50:
            raise RuntimeError(f"Invalid top-50 for request {request_id}")
        rows.append({"HitLogID": int(request["hit_log_id"]), "BannerID": banners})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-run", type=Path, required=True)
    parser.add_argument("--old-model-a-run", type=Path, required=True)
    parser.add_argument("--old-model-b-run", type=Path, required=True)
    parser.add_argument("--new-run", type=Path, required=True)
    parser.add_argument("--new-model-a-run", type=Path, required=True)
    parser.add_argument("--new-model-b-run", type=Path, required=True)
    parser.add_argument("--old-model-a-weight", type=float, default=0.5)
    parser.add_argument("--old-catboost-weight", type=float, default=0.5)
    parser.add_argument("--new-model-a-weight", type=float, default=0.65)
    parser.add_argument("--new-catboost-weight", type=float, default=0.6)
    parser.add_argument("--new-weight", type=float, default=0.4)
    parser.add_argument("--rrf-constant", type=float, default=10.0)
    parser.add_argument("--exponent", type=float, default=0.2)
    parser.add_argument("--rerank-top-n", type=int, default=75)
    parser.add_argument("--poll-seconds", type=int, default=20)
    parser.add_argument("--max-wait-seconds", type=int, default=7200)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    started = time.monotonic()
    while not completed_run(args.new_run):
        result_path = args.new_run / "result.json"
        if result_path.is_file():
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if result.get("status") == "failed":
                raise RuntimeError("New full run failed before cross-pool submission")
        if time.monotonic() - started >= args.max_wait_seconds:
            raise TimeoutError("Timed out waiting for new full run")
        time.sleep(max(1, args.poll_seconds))
    if not completed_run(args.old_run):
        raise RuntimeError("Old full donor is not completed")

    old = ranked_pool(
        run=args.old_run,
        model_a_run=args.old_model_a_run,
        model_b_run=args.old_model_b_run,
        model_a_weight=args.old_model_a_weight,
        catboost_weight=args.old_catboost_weight,
        exponent=0.0,
        rerank_top_n=750,
        split="test",
    )
    new = ranked_pool(
        run=args.new_run,
        model_a_run=args.new_model_a_run,
        model_b_run=args.new_model_b_run,
        model_a_weight=args.new_model_a_weight,
        catboost_weight=args.new_catboost_weight,
        exponent=0.0,
        rerank_top_n=750,
        split="test",
    )
    requests = read_request_parquet(args.new_run / "data/test_requests.parquet")
    rows = submission_rows(
        requests,
        old,
        new,
        new_weight=args.new_weight,
        rrf_constant=args.rrf_constant,
        exponent=args.exponent,
        rerank_top_n=args.rerank_top_n,
    )
    schema = pa.schema(
        [
            pa.field("HitLogID", pa.uint64(), nullable=False),
            pa.field("BannerID", pa.list_(pa.uint64()), nullable=False),
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with atomic_output_path(args.output) as temporary:
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), temporary, compression="zstd")

    parameters = {
        "old_run": str(args.old_run),
        "new_run": str(args.new_run),
        "old_model_a_weight": args.old_model_a_weight,
        "old_catboost_weight": args.old_catboost_weight,
        "new_model_a_weight": args.new_model_a_weight,
        "new_catboost_weight": args.new_catboost_weight,
        "new_weight": args.new_weight,
        "rrf_constant": args.rrf_constant,
        "source_cost_scale": 1_000_000.0,
        "exponent": args.exponent,
        "rerank_top_n": args.rerank_top_n,
    }
    inputs = [
        fingerprint_file(run / "models/catboost.cbm")
        for run in (
            args.old_model_a_run,
            args.old_model_b_run,
            args.new_model_a_run,
            args.new_model_b_run,
        )
    ]
    write_output_manifest(
        args.output,
        stage="make_cross_pool_submission",
        artifact_version="cross_pool_rrf_geometry_top50_v1",
        config_sha256=content_fingerprint(parameters),
        inputs=inputs,
        rows=len(rows),
        schema=str(schema),
        scope="full",
    )

    cfg = OmegaConf.load(args.new_run / "config.yaml")
    index = pq.read_table(str(cfg.paths.banner_index), columns=["BannerID"])
    validation = validate_submission(
        args.output,
        expected_hitlog_ids={int(row["hit_log_id"]) for row in requests},
        valid_banner_ids={int(value) for value in index["BannerID"].to_pylist()},
        top_k=int(cfg.evaluation.submission_top_k),
        allow_short=False,
    )
    report = {
        "status": "completed" if validation["ok"] else "validation_failed",
        "path": str(args.output),
        "rows": len(rows),
        "parameters": parameters,
        "validation": validation,
        "artifact": fingerprint_file(args.output),
        "wall_seconds": time.monotonic() - started,
    }
    atomic_write_json(args.new_run / "metrics/cross_pool_submission.json", report)
    atomic_write_json(
        args.new_run / "metrics/cross_pool_submission_validation.json", validation
    )
    print(json.dumps(report, indent=2))
    return 0 if validation["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
