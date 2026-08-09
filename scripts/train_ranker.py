#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mla_recsys.artifacts import (  # noqa: E402
    atomic_write_json,
    fingerprint_file,
    write_output_manifest,
)
from mla_recsys.command import load_stage_context  # noqa: E402
from mla_recsys.config import config_fingerprint  # noqa: E402


def read_features(run_path: Path, split: str, columns: list[str]) -> pa.Table:
    paths = sorted((run_path / "features" / split).glob("part-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No feature parts for {split}")
    return pa.concat_tables([pq.read_table(path, columns=columns) for path in paths])


def matrix(table: pa.Table, names: list[str]) -> np.ndarray:
    return np.column_stack(
        [table[name].combine_chunks().to_numpy(zero_copy_only=False) for name in names]
    ).astype(np.float32, copy=False)


def positive_groups_only(table: pa.Table) -> pa.Table:
    return table.filter(pc.equal(table["group_has_positive"], True))


def filter_training_window(
    table: pa.Table,
    requests: pa.Table,
    cfg: object,
) -> tuple[pa.Table, dict[str, object]]:
    """Keep complete ranking groups from the most recent configured window."""
    days = float(cfg.ranker.get("training_window_days", 0.0))
    if days < 0.0:
        raise ValueError("ranker.training_window_days must be non-negative")
    if days == 0.0:
        return table, {
            "enabled": False,
            "days": 0.0,
            "rows_before": table.num_rows,
            "rows_after": table.num_rows,
        }
    if "request_id" not in table.column_names:
        raise ValueError("request_id is required for a ranker training window")
    required = {"request_id", "show_time"}
    if not required.issubset(requests.column_names):
        raise ValueError("request table must contain request_id and show_time")

    request_ids = requests["request_id"].combine_chunks().to_pylist()
    show_times = requests["show_time"].combine_chunks().to_pylist()
    timed = [
        (str(request_id), int(show_time))
        for request_id, show_time in zip(request_ids, show_times)
        if show_time is not None
    ]
    if not timed:
        raise ValueError("No timestamped requests are available for training window")
    maximum = max(show_time for _, show_time in timed)
    cutoff = maximum - int(days * 24 * 60 * 60)
    allowed = pa.array(
        [request_id for request_id, show_time in timed if show_time >= cutoff],
        type=pa.string(),
    )
    filtered = table.filter(pc.is_in(table["request_id"], value_set=allowed))
    if filtered.num_rows == 0:
        raise RuntimeError("Ranker training window removed every feature row")
    return filtered, {
        "enabled": True,
        "days": days,
        "cutoff_show_time": cutoff,
        "maximum_show_time": maximum,
        "requests_before": len(timed),
        "requests_after": len(allowed),
        "rows_before": table.num_rows,
        "rows_after": filtered.num_rows,
    }


def group_weight_array(
    table: pa.Table,
    cfg: object,
) -> tuple[np.ndarray | None, dict[str, object]]:
    """Return request-level SourceCost weights expanded to candidate rows.

    CatBoost requires every object in a ranking group to have the same group
    weight. Feature rows are emitted contiguously per request, so we compute
    one robust weight per contiguous group and repeat it for all candidates.
    """
    spec = cfg.ranker.get("group_weight", {})
    kind = str(spec.get("kind", "none"))
    if kind == "none":
        return None, {"kind": "none"}
    if kind != "source_cost":
        raise ValueError(f"Unsupported ranker.group_weight.kind: {kind}")
    if table.num_rows == 0:
        raise ValueError("Cannot build group weights for an empty table")

    cap_quantile = float(spec.get("cap_quantile", 1.0))
    power = float(spec.get("power", 1.0))
    minimum = float(spec.get("minimum", 1.0e-6))
    if not 0.0 < cap_quantile <= 1.0:
        raise ValueError("ranker.group_weight.cap_quantile must be in (0, 1]")
    if power <= 0.0:
        raise ValueError("ranker.group_weight.power must be positive")
    if minimum <= 0.0:
        raise ValueError("ranker.group_weight.minimum must be positive")

    group_ids = table["group_id"].combine_chunks().to_numpy(zero_copy_only=False)
    starts = np.r_[0, np.flatnonzero(group_ids[1:] != group_ids[:-1]) + 1]
    group_keys = group_ids[starts]
    if np.unique(group_keys).size != group_keys.size:
        raise ValueError("Ranking groups must be contiguous before weighting")
    lengths = np.diff(np.r_[starts, table.num_rows])
    source_cost = (
        table["label_raw_sc"].combine_chunks().to_numpy(zero_copy_only=False)
    )
    # SourceCost Recall sums every clicked banner, so multi-click requests use
    # the sum of their positive costs rather than only the largest click.
    raw_group_weights = np.add.reduceat(source_cost, starts).astype(
        np.float64, copy=False
    )
    cap = float(np.quantile(raw_group_weights, cap_quantile))
    transformed = np.power(
        np.maximum(np.minimum(raw_group_weights, cap), minimum), power
    )
    mean = float(transformed.mean())
    if not np.isfinite(mean) or mean <= 0.0:
        raise ValueError("SourceCost group weights have invalid mean")
    normalized = transformed / mean
    expanded = np.repeat(normalized, lengths).astype(np.float32, copy=False)
    return expanded, {
        "kind": kind,
        "groups": int(group_keys.size),
        "cap_quantile": cap_quantile,
        "cap_value": cap,
        "power": power,
        "minimum": minimum,
        "raw_min": float(raw_group_weights.min()),
        "raw_median": float(np.median(raw_group_weights)),
        "raw_max": float(raw_group_weights.max()),
        "normalized_min": float(normalized.min()),
        "normalized_max": float(normalized.max()),
        "normalized_mean": float(normalized.mean()),
    }


def label_spec(cfg: object) -> tuple[str, float]:
    kind = str(cfg.ranker.kind)
    if kind == "ranker_binary":
        return "label_binary", 1.0
    if kind == "ranker_logsc":
        return "label_logsc", 1.0
    if kind == "ranker_raw_sc_label":
        scale = float(cfg.ranker.raw_sc_scale)
        if scale <= 0.0:
            raise ValueError("ranker.raw_sc_scale must be positive")
        return "label_raw_sc", scale
    raise ValueError(f"Unsupported ranker.kind: {kind}")


def main() -> int:
    context = load_stage_context("Train CatBoost on cached natural-pool features")
    cfg = context.cfg
    if str(cfg.runtime.scope) == "full":
        train_split = "full_train"
        validation_split = None
    else:
        train_split = "train"
        validation_split = "holdout"

    from catboost import CatBoostRanker, Pool
    from mla_recsys.feature_cache import configured_feature_names

    names = configured_feature_names(cfg)
    label_column, label_scale = label_spec(cfg)
    training_window_enabled = float(cfg.ranker.get("training_window_days", 0.0)) > 0.0
    needed = list(
        dict.fromkeys(
            [
                "group_id",
                "group_has_positive",
                label_column,
                "label_raw_sc",
                *(["request_id"] if training_window_enabled else []),
                *names,
            ]
        )
    )
    train_all = read_features(context.store.path, train_split, needed)
    train_windowed, train_window_stats = filter_training_window(
        train_all,
        pq.read_table(
            context.store.path / "data" / f"{train_split}_requests.parquet",
            columns=["request_id", "show_time"],
        ),
        cfg,
    )
    train = positive_groups_only(train_windowed)
    if train.num_rows == 0:
        raise RuntimeError("No natural-pool positive groups in train")
    train_group_weight, train_group_weight_stats = group_weight_array(train, cfg)
    train_pool = Pool(
        matrix(train, names),
        label=train[label_column].combine_chunks().to_numpy(zero_copy_only=False)
        / label_scale,
        group_id=train["group_id"].combine_chunks().to_numpy(zero_copy_only=False),
        group_weight=train_group_weight,
        feature_names=names,
    )
    eval_pool = None
    validation_rows = 0
    validation_group_weight_stats = None
    if validation_split is not None:
        validation = positive_groups_only(
            read_features(context.store.path, validation_split, needed)
        )
        validation_rows = validation.num_rows
        validation_group_weight, validation_group_weight_stats = group_weight_array(
            validation, cfg
        )
        eval_pool = Pool(
            matrix(validation, names),
            label=validation[label_column].combine_chunks().to_numpy(zero_copy_only=False)
            / label_scale,
            group_id=validation["group_id"].combine_chunks().to_numpy(zero_copy_only=False),
            group_weight=validation_group_weight,
            feature_names=names,
        )
    iterations = (
        int(cfg.ranker.smoke_iterations)
        if str(cfg.runtime.mode) == "smoke"
        else int(cfg.ranker.iterations)
    )
    model = CatBoostRanker(
        loss_function=str(cfg.ranker.loss_function),
        eval_metric=str(cfg.ranker.eval_metric),
        iterations=iterations,
        depth=int(cfg.ranker.depth),
        learning_rate=float(cfg.ranker.learning_rate),
        l2_leaf_reg=float(cfg.ranker.l2_leaf_reg),
        random_seed=int(cfg.ranker.random_seed),
        task_type=str(cfg.ranker.task_type),
        devices=str(cfg.ranker.devices) if str(cfg.ranker.task_type) == "GPU" else None,
        verbose=max(1, iterations // 10),
        allow_writing_files=False,
    )
    fit_kwargs = {}
    if eval_pool is not None:
        fit_kwargs.update(
            eval_set=eval_pool,
            early_stopping_rounds=int(cfg.ranker.early_stopping_rounds),
            use_best_model=True,
        )
    model.fit(train_pool, **fit_kwargs)

    model_dir = context.store.path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "catboost.cbm"
    model.save_model(str(model_path))
    metadata = {
        "version": 2,
        "kind": str(cfg.ranker.kind),
        "feature_names": names,
        "candidate_pool": int(cfg.candidates.ranker_pool),
        "split": str(cfg.split.version),
        "scope": str(cfg.runtime.scope),
        "train_rows_all": train_all.num_rows,
        "train_rows_windowed": train_windowed.num_rows,
        "train_rows_fit": train.num_rows,
        "validation_rows_fit": validation_rows,
        "best_iteration": model.get_best_iteration(),
        "best_score": model.get_best_score(),
        "label_column": label_column,
        "label_scale": label_scale,
        "training_window": train_window_stats,
        "group_weight": {
            "train": train_group_weight_stats,
            "validation": validation_group_weight_stats,
        },
    }
    atomic_write_json(model_dir / "catboost.json", metadata)
    importance = model.get_feature_importance(
        type=str(cfg.ranker.importance.type),
        prettified=True,
    )
    importance_path = context.store.path / "reports" / "feature_importance.csv"
    importance.to_csv(importance_path, index=False)
    metadata["importance"] = {
        "type": str(cfg.ranker.importance.type),
        "features": len(names),
        "report": str(importance_path),
    }
    atomic_write_json(model_dir / "catboost.json", metadata)
    inputs = [
        fingerprint_file(path)
        for split in [train_split, validation_split]
        if split
        for path in sorted((context.store.path / "features" / split).glob("part-*.parquet"))
    ]
    write_output_manifest(
        model_path,
        stage="train_ranker",
        artifact_version=str(cfg.ranker.version),
        config_sha256=config_fingerprint(cfg),
        inputs=inputs,
        schema={"feature_names": names, "kind": str(cfg.ranker.kind)},
        scope=str(cfg.runtime.scope),
    )
    atomic_write_json(context.store.path / "metrics" / "ranker_train.json", metadata)
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
