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


def label_spec(cfg: object) -> tuple[str, float]:
    kind = str(cfg.ranker.kind)
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
    from mla_recsys.importance import first_complete_groups, permutation_importance

    names = configured_feature_names(cfg)
    label_column, label_scale = label_spec(cfg)
    needed = ["group_id", "group_has_positive", label_column, *names]
    train_all = read_features(context.store.path, train_split, needed)
    train = positive_groups_only(train_all)
    if train.num_rows == 0:
        raise RuntimeError("No natural-pool positive groups in train")
    train_pool = Pool(
        matrix(train, names),
        label=train[label_column].combine_chunks().to_numpy(zero_copy_only=False)
        / label_scale,
        group_id=train["group_id"].combine_chunks().to_numpy(zero_copy_only=False),
        feature_names=names,
    )
    eval_pool = None
    validation_rows = 0
    if validation_split is not None:
        validation = positive_groups_only(
            read_features(context.store.path, validation_split, needed)
        )
        validation_rows = validation.num_rows
        eval_pool = Pool(
            matrix(validation, names),
            label=validation[label_column].combine_chunks().to_numpy(zero_copy_only=False)
            / label_scale,
            group_id=validation["group_id"].combine_chunks().to_numpy(zero_copy_only=False),
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
        "train_rows_fit": train.num_rows,
        "validation_rows_fit": validation_rows,
        "best_iteration": model.get_best_iteration(),
        "best_score": model.get_best_score(),
        "label_column": label_column,
        "label_scale": label_scale,
    }
    atomic_write_json(model_dir / "catboost.json", metadata)
    importance = model.get_feature_importance(
        type="PredictionValuesChange",
        prettified=True,
    )
    importance_path = context.store.path / "reports" / "feature_importance.csv"
    importance.to_csv(importance_path, index=False)
    importance_records = importance.to_dict(orient="records")
    importance_name_key = "Feature Id"
    if importance_records and importance_name_key not in importance_records[0]:
        importance_name_key = next(iter(importance_records[0]))
    importance_value_key = (
        next(key for key in importance_records[0] if key != importance_name_key)
        if importance_records
        else "Importances"
    )

    importance_table = validation if validation_split is not None else train
    sample_limit = int(cfg.ranker.importance.permutation_sample_rows)
    sample_group_ids = importance_table["group_id"].combine_chunks().to_numpy(
        zero_copy_only=False
    )
    sample_indices = first_complete_groups(sample_group_ids, sample_limit)
    sampled = importance_table.take(pa.array(sample_indices))
    sampled_matrix = matrix(sampled, names)
    sampled_labels = (
        sampled[label_column].combine_chunks().to_numpy(zero_copy_only=False)
        / label_scale
    )
    sampled_groups = sampled["group_id"].combine_chunks().to_numpy(zero_copy_only=False)
    standard_by_name = {
        str(row[importance_name_key]): float(row[importance_value_key])
        for row in importance_records
    }
    permutation_top = int(cfg.ranker.importance.get("permutation_top_features", 40))
    selected = sorted(
        range(len(names)),
        key=lambda index: -standard_by_name.get(names[index], 0.0),
    )[:permutation_top]
    permutation_baseline, permutation_rows = permutation_importance(
        model,
        sampled_matrix,
        sampled_labels,
        sampled_groups,
        names,
        feature_indices=selected,
        repeats=int(cfg.ranker.importance.permutation_repeats),
        top_k=50,
        seed=int(cfg.ranker.random_seed),
    )
    import pandas as pd

    pd.DataFrame(permutation_rows).to_csv(
        context.store.path / "reports" / "feature_importance_permutation.csv",
        index=False,
    )
    shap_rows = min(int(cfg.ranker.importance.shap_sample_rows), sampled.num_rows)
    shap_pool = Pool(
        sampled_matrix[:shap_rows],
        feature_names=names,
    )
    shap_values = np.asarray(
        model.get_feature_importance(type="ShapValues", data=shap_pool)
    )[:, :-1]
    shap_summary = sorted(
        (
            {"feature": name, "mean_abs_shap": float(value)}
            for name, value in zip(names, np.mean(np.abs(shap_values), axis=0))
        ),
        key=lambda row: -row["mean_abs_shap"],
    )[: int(cfg.ranker.importance.shap_top_k)]
    pd.DataFrame(shap_summary).to_csv(
        context.store.path / "reports" / "feature_importance_shap_top20.csv",
        index=False,
    )
    metadata["importance"] = {
        "permutation_metric": "sourcecost_capture@50_within_natural_positive_groups",
        "permutation_baseline": permutation_baseline,
        "permutation_sample_rows": sampled.num_rows,
        "permutation_features": len(selected),
        "shap_sample_rows": shap_rows,
        "shap_top_k": int(cfg.ranker.importance.shap_top_k),
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
