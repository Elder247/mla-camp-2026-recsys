from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def topk_value_capture(
    scores: np.ndarray,
    labels: np.ndarray,
    group_ids: np.ndarray,
    *,
    top_k: int,
) -> float:
    numerator = 0.0
    denominator = float(labels.sum())
    if denominator <= 0.0:
        return 0.0
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and group_ids[end] == group_ids[start]:
            end += 1
        order = np.argsort(-scores[start:end], kind="stable")[:top_k]
        numerator += float(labels[start:end][order].sum())
        start = end
    return numerator / denominator


def permutation_importance(
    model: object,
    matrix: np.ndarray,
    labels: np.ndarray,
    group_ids: np.ndarray,
    feature_names: Sequence[str],
    *,
    feature_indices: Sequence[int],
    repeats: int,
    top_k: int,
    seed: int,
) -> tuple[float, list[dict[str, float | str]]]:
    baseline_scores = np.asarray(model.predict(matrix), dtype=np.float64)
    baseline = topk_value_capture(
        baseline_scores, labels, group_ids, top_k=top_k
    )
    rng = np.random.default_rng(seed)
    output = []
    for index in feature_indices:
        original = matrix[:, index].copy()
        values = []
        for _ in range(int(repeats)):
            matrix[:, index] = original[rng.permutation(len(original))]
            score = topk_value_capture(
                np.asarray(model.predict(matrix), dtype=np.float64),
                labels,
                group_ids,
                top_k=top_k,
            )
            values.append(baseline - score)
        matrix[:, index] = original
        output.append(
            {
                "feature": str(feature_names[index]),
                "baseline_sourcecost_capture": baseline,
                "permutation_drop_mean": float(np.mean(values)),
                "permutation_drop_std": float(np.std(values)),
            }
        )
    output.sort(key=lambda row: -float(row["permutation_drop_mean"]))
    return baseline, output


def first_complete_groups(group_ids: np.ndarray, max_rows: int) -> np.ndarray:
    if len(group_ids) <= max_rows:
        return np.arange(len(group_ids), dtype=np.int64)
    end = int(max_rows)
    while end < len(group_ids) and group_ids[end] == group_ids[end - 1]:
        end += 1
    return np.arange(end, dtype=np.int64)
