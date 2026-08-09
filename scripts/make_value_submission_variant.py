#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import resource
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from make_submission import catboost_predictions  # noqa: E402
from mla_recsys.artifacts import (  # noqa: E402
    atomic_output_path,
    atomic_write_json,
    fingerprint_file,
    write_output_manifest,
)
from mla_recsys.config import config_fingerprint  # noqa: E402
from mla_recsys.data import read_request_parquet  # noqa: E402
from mla_recsys.submission import validate_submission  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    started = time.monotonic()
    parser = argparse.ArgumentParser(
        description="Create a non-overwriting SourceCost geometry submission variant"
    )
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--catboost-weight", type=float, required=True)
    parser.add_argument("--source-cost-scale", type=float, default=1_000_000.0)
    parser.add_argument("--exponent", type=float, required=True)
    parser.add_argument("--rerank-top-n", type=int, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise FileExistsError("Variant output/report already exists; refusing to overwrite")

    parameters = {
        "catboost_weight": args.catboost_weight,
        "source_cost_scale": args.source_cost_scale,
        "exponent": args.exponent,
        "rerank_top_n": args.rerank_top_n,
    }
    predictions, feature_paths = catboost_predictions(
        args.run,
        value_geometry=parameters,
    )
    requests = read_request_parquet(args.run / "data" / "test_requests.parquet")
    missing = [row["request_id"] for row in requests if row["request_id"] not in predictions]
    if missing:
        raise RuntimeError(f"Requests without predictions: {len(missing)}")
    rows = [
        {
            "HitLogID": int(predictions[str(request["request_id"])][0]),
            "BannerID": predictions[str(request["request_id"])][1],
        }
        for request in requests
    ]
    schema = pa.schema(
        [
            pa.field("HitLogID", pa.uint64(), nullable=False),
            pa.field("BannerID", pa.list_(pa.uint64()), nullable=False),
        ]
    )
    table = pa.Table.from_pylist(rows, schema=schema)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with atomic_output_path(args.output) as temporary:
        pq.write_table(table, temporary, compression="zstd")

    cfg = OmegaConf.load(args.run / "config.yaml")
    variant_cfg = OmegaConf.merge(
        cfg,
        {
            "submission": {
                "ranking": "value_geometry",
                "blend": {"catboost_weight": args.catboost_weight},
                "value_geometry": {
                    "source_cost_scale": args.source_cost_scale,
                    "exponent": args.exponent,
                    "rerank_top_n": args.rerank_top_n,
                },
            }
        },
    )
    inputs = [fingerprint_file(args.run / "models" / "catboost.cbm")]
    inputs.extend(fingerprint_file(path) for path in feature_paths)
    write_output_manifest(
        args.output,
        stage="make_value_submission_variant",
        artifact_version="value_geometry_batch_top50_v1",
        config_sha256=config_fingerprint(variant_cfg),
        inputs=inputs,
        rows=table.num_rows,
        schema=str(schema),
        scope="full",
    )

    banner_index = pq.read_table(cfg.paths.banner_index, columns=["BannerID"])
    validation = validate_submission(
        args.output,
        expected_hitlog_ids={int(row["hit_log_id"]) for row in requests},
        valid_banner_ids={int(value) for value in banner_index["BannerID"].to_pylist()},
        top_k=int(cfg.evaluation.submission_top_k),
        allow_short=bool(cfg.submission.allow_fewer_than_top_k),
    )
    report = {
        "run": str(args.run),
        "path": str(args.output),
        "sha256": sha256_file(args.output),
        "rows": table.num_rows,
        "parameters": parameters,
        "validation": validation,
        "wall_seconds": time.monotonic() - started,
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        * 1024,
    }
    atomic_write_json(args.report, report)
    print(json.dumps(report, indent=2))
    return 0 if validation["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
