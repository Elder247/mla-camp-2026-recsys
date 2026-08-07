#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from catboost import CatBoostRanker

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mla_recsys.artifacts import (  # noqa: E402
    atomic_output_path,
    atomic_write_json,
    fingerprint_file,
    write_output_manifest,
)
from mla_recsys.command import load_stage_context  # noqa: E402
from mla_recsys.config import config_fingerprint  # noqa: E402
from mla_recsys.data import read_request_parquet  # noqa: E402


def matrix(table: pa.Table, names: list[str]) -> np.ndarray:
    return np.column_stack(
        [table[name].combine_chunks().to_numpy(zero_copy_only=False) for name in names]
    ).astype(np.float32, copy=False)


def main() -> int:
    context = load_stage_context("Batch CatBoost inference and top-50 submission")
    cfg = context.cfg
    if str(cfg.runtime.scope) != "full":
        raise ValueError("make_submission requires scope=full")
    metadata = json.loads(
        (context.store.path / "models" / "catboost.json").read_text(encoding="utf-8")
    )
    names = list(metadata["feature_names"])
    model = CatBoostRanker()
    model_path = context.store.path / "models" / "catboost.cbm"
    model.load_model(str(model_path))
    predictions: dict[str, tuple[int, list[int]]] = {}
    feature_paths = sorted((context.store.path / "features" / "test").glob("part-*.parquet"))
    for path in feature_paths:
        table = pq.read_table(
            path,
            columns=["request_id", "hit_log_id", "banner_id", "pre_rank", *names],
        )
        if table.num_rows == 0:
            continue
        scores = model.predict(matrix(table, names))
        rows = table.select(["request_id", "hit_log_id", "banner_id", "pre_rank"]).to_pylist()
        grouped: dict[str, list[tuple[float, int, int, int]]] = {}
        for row, score in zip(rows, scores):
            grouped.setdefault(str(row["request_id"]), []).append(
                (
                    float(score),
                    int(row["pre_rank"]),
                    int(row["banner_id"]),
                    int(row["hit_log_id"]),
                )
            )
        for request_id, values in grouped.items():
            values.sort(key=lambda value: (-value[0], value[1], value[2]))
            predictions[request_id] = (values[0][3], [value[2] for value in values[:50]])

    expected = read_request_parquet(context.store.path / "data" / "test_requests.parquet")
    missing = [row["request_id"] for row in expected if row["request_id"] not in predictions]
    if missing:
        raise RuntimeError(f"Requests without predictions: {len(missing)}")
    rows = [
        {
            "HitLogID": int(predictions[str(request["request_id"])][0]),
            "BannerID": predictions[str(request["request_id"])][1],
        }
        for request in expected
    ]
    schema = pa.schema(
        [
            pa.field("HitLogID", pa.uint64(), nullable=False),
            pa.field("BannerID", pa.list_(pa.uint64()), nullable=False),
        ]
    )
    table = pa.Table.from_pylist(rows, schema=schema)
    output = context.store.path / "predictions" / "test_top50.parquet"
    with atomic_output_path(output) as temporary:
        pq.write_table(table, temporary, compression="zstd")
    inputs = [fingerprint_file(model_path)] + [fingerprint_file(path) for path in feature_paths]
    write_output_manifest(
        output,
        stage="make_submission",
        artifact_version="catboost_batch_top50_v1",
        config_sha256=config_fingerprint(cfg),
        inputs=inputs,
        rows=table.num_rows,
        schema=str(schema),
        scope="full",
    )
    report = {
        "path": str(output),
        "rows": table.num_rows,
        "min_items": min(len(row["BannerID"]) for row in rows),
        "max_items": max(len(row["BannerID"]) for row in rows),
    }
    atomic_write_json(context.store.path / "metrics" / "submission.json", report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

