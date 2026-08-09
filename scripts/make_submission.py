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
from mla_recsys.rank_blend import (  # noqa: E402
    rank_linear_order,
    rank_value_geometric_order,
    two_model_rank_linear_order,
    value_geometric_from_base_order,
)


def matrix(table: pa.Table, names: list[str]) -> np.ndarray:
    return np.column_stack(
        [table[name].combine_chunks().to_numpy(zero_copy_only=False) for name in names]
    ).astype(np.float32, copy=False)


def rrf_predictions(run_path: Path) -> tuple[dict[str, tuple[int, list[int]]], list[Path]]:
    predictions: dict[str, tuple[int, list[int]]] = {}
    paths = sorted((run_path / "candidates" / "test" / "merged").glob("part-*.parquet"))
    for path in paths:
        rows = pq.read_table(
            path,
            columns=["request_id", "hit_log_id", "banner_id", "pre_rank"],
        ).to_pylist()
        grouped: dict[str, list[tuple[int, int, int]]] = {}
        for row in rows:
            grouped.setdefault(str(row["request_id"]), []).append(
                (int(row["pre_rank"]), int(row["banner_id"]), int(row["hit_log_id"]))
            )
        for request_id, values in grouped.items():
            values.sort(key=lambda value: (value[0], value[1]))
            predictions[request_id] = (
                values[0][2],
                [value[1] for value in values[:50]],
            )
    return predictions, paths


def catboost_predictions(
    run_path: Path,
    *,
    blend_weight: float | None = None,
    value_geometry: dict[str, float | int] | None = None,
) -> tuple[dict[str, tuple[int, list[int]]], list[Path]]:
    metadata = json.loads(
        (run_path / "models" / "catboost.json").read_text(encoding="utf-8")
    )
    names = list(metadata["feature_names"])
    model = CatBoostRanker()
    model.load_model(str(run_path / "models" / "catboost.cbm"))
    predictions: dict[str, tuple[int, list[int]]] = {}
    paths = sorted((run_path / "features" / "test").glob("part-*.parquet"))
    for path in paths:
        columns = ["request_id", "hit_log_id", "banner_id", "pre_rank", *names]
        if value_geometry is not None:
            columns.append("source_cost_raw")
        table = pq.read_table(path, columns=list(dict.fromkeys(columns)))
        if table.num_rows == 0:
            continue
        scores = model.predict(matrix(table, names))
        selected = ["request_id", "hit_log_id", "banner_id", "pre_rank"]
        if value_geometry is not None:
            selected.append("source_cost_raw")
        rows = table.select(selected).to_pylist()
        grouped: dict[str, list[tuple]] = {}
        for row, score in zip(rows, scores):
            value = (
                float(score),
                int(row["pre_rank"]),
                int(row["banner_id"]),
                int(row["hit_log_id"]),
            )
            if value_geometry is not None:
                value = (*value, float(row["source_cost_raw"]))
            grouped.setdefault(str(row["request_id"]), []).append(value)
        for request_id, values in grouped.items():
            if value_geometry is not None:
                values = rank_value_geometric_order(
                    values,
                    catboost_weight=float(value_geometry["catboost_weight"]),
                    source_cost_scale=float(value_geometry["source_cost_scale"]),
                    exponent=float(value_geometry["exponent"]),
                    rerank_top_n=int(value_geometry["rerank_top_n"]),
                )
            elif blend_weight is None:
                values.sort(key=lambda value: (-value[0], value[1], value[2]))
            else:
                values = rank_linear_order(values, catboost_weight=blend_weight)
            predictions[request_id] = (
                values[0][3],
                [value[2] for value in values[:50]],
            )
    return predictions, paths


def model_ensemble_predictions(
    run_path: Path,
    *,
    model_a_path: Path,
    model_a_weight: float,
    catboost_weight: float,
    source_cost_scale: float,
    exponent: float,
    rerank_top_n: int,
) -> tuple[dict[str, tuple[int, list[int]]], list[Path]]:
    current_metadata = json.loads(
        (run_path / "models" / "catboost.json").read_text(encoding="utf-8")
    )
    external_metadata_path = model_a_path.with_name("catboost.json")
    external_metadata = json.loads(external_metadata_path.read_text(encoding="utf-8"))
    names = list(current_metadata["feature_names"])
    if names != list(external_metadata["feature_names"]):
        raise ValueError("Ensemble models use different ordered feature contracts")
    model_a = CatBoostRanker()
    model_a.load_model(str(model_a_path))
    model_b = CatBoostRanker()
    model_b.load_model(str(run_path / "models" / "catboost.cbm"))
    predictions: dict[str, tuple[int, list[int]]] = {}
    paths = sorted((run_path / "features" / "test").glob("part-*.parquet"))
    for path in paths:
        columns = list(
            dict.fromkeys(
                [
                    "request_id",
                    "hit_log_id",
                    "banner_id",
                    "pre_rank",
                    "source_cost_raw",
                    *names,
                ]
            )
        )
        table = pq.read_table(path, columns=columns)
        if table.num_rows == 0:
            continue
        values = matrix(table, names)
        scores_a = model_a.predict(values)
        scores_b = model_b.predict(values)
        rows = table.select(
            ["request_id", "hit_log_id", "banner_id", "pre_rank", "source_cost_raw"]
        ).to_pylist()
        grouped: dict[str, list[tuple]] = {}
        for row, score_a, score_b in zip(rows, scores_a, scores_b):
            grouped.setdefault(str(row["request_id"]), []).append(
                (
                    float(score_a),
                    float(score_b),
                    int(row["pre_rank"]),
                    int(row["banner_id"]),
                    int(row["hit_log_id"]),
                    float(row["source_cost_raw"]),
                )
            )
        for request_id, candidates in grouped.items():
            base = two_model_rank_linear_order(
                candidates,
                model_a_weight=model_a_weight,
                catboost_weight=catboost_weight,
            )
            geometric_base = [
                (0.0, value[2], value[3], value[4], value[5]) for value in base
            ]
            ordered = value_geometric_from_base_order(
                geometric_base,
                source_cost_scale=source_cost_scale,
                exponent=exponent,
                rerank_top_n=rerank_top_n,
            )
            predictions[request_id] = (
                ordered[0][3],
                [value[2] for value in ordered[:50]],
            )
    return predictions, [model_a_path, external_metadata_path, *paths]


def main() -> int:
    context = load_stage_context("Batch selected-ranker inference and top-50 submission")
    cfg = context.cfg
    if str(cfg.runtime.scope) != "full":
        raise ValueError("make_submission requires scope=full")
    ranking = str(cfg.submission.ranking)
    if ranking == "rrf":
        predictions, prediction_inputs = rrf_predictions(context.store.path)
    elif ranking == "catboost":
        predictions, prediction_inputs = catboost_predictions(context.store.path)
    elif ranking == "blend":
        predictions, prediction_inputs = catboost_predictions(
            context.store.path,
            blend_weight=float(cfg.submission.blend.catboost_weight),
        )
    elif ranking == "value_geometry":
        predictions, prediction_inputs = catboost_predictions(
            context.store.path,
            value_geometry={
                "catboost_weight": float(cfg.submission.blend.catboost_weight),
                "source_cost_scale": float(
                    cfg.submission.value_geometry.source_cost_scale
                ),
                "exponent": float(cfg.submission.value_geometry.exponent),
                "rerank_top_n": int(cfg.submission.value_geometry.rerank_top_n),
            },
        )
    elif ranking == "model_ensemble":
        ensemble = cfg.submission.model_ensemble
        if str(ensemble.method) != "rank_linear":
            raise ValueError("Only rank_linear model ensemble is supported")
        predictions, prediction_inputs = model_ensemble_predictions(
            context.store.path,
            model_a_path=Path(str(ensemble.model_a_path)),
            model_a_weight=float(ensemble.model_a_weight),
            catboost_weight=float(ensemble.catboost_weight),
            source_cost_scale=float(cfg.submission.value_geometry.source_cost_scale),
            exponent=float(cfg.submission.value_geometry.exponent),
            rerank_top_n=int(cfg.submission.value_geometry.rerank_top_n),
        )
    else:
        raise ValueError(f"Unsupported submission ranking: {ranking}")

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
    inputs = [fingerprint_file(path) for path in prediction_inputs]
    if ranking in {"catboost", "blend", "value_geometry", "model_ensemble"}:
        inputs.insert(0, fingerprint_file(context.store.path / "models" / "catboost.cbm"))
    write_output_manifest(
        output,
        stage="make_submission",
        artifact_version=f"{ranking}_batch_top50_v1",
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
        "ranking": ranking,
    }
    if ranking == "value_geometry":
        report["value_geometry"] = {
            "catboost_weight": float(cfg.submission.blend.catboost_weight),
            "source_cost_scale": float(
                cfg.submission.value_geometry.source_cost_scale
            ),
            "exponent": float(cfg.submission.value_geometry.exponent),
            "rerank_top_n": int(cfg.submission.value_geometry.rerank_top_n),
        }
    if ranking == "model_ensemble":
        report["model_ensemble"] = {
            "method": str(cfg.submission.model_ensemble.method),
            "model_a_path": str(cfg.submission.model_ensemble.model_a_path),
            "model_a_weight": float(cfg.submission.model_ensemble.model_a_weight),
            "catboost_weight": float(cfg.submission.model_ensemble.catboost_weight),
            "source_cost_scale": float(cfg.submission.value_geometry.source_cost_scale),
            "exponent": float(cfg.submission.value_geometry.exponent),
            "rerank_top_n": int(cfg.submission.value_geometry.rerank_top_n),
        }
    atomic_write_json(context.store.path / "metrics" / "submission.json", report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
