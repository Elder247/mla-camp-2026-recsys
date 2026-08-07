from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from common.text import normalize, tokenize


BASE_FEATURES = [
    "rrf_score",
    "source_count",
    "query_char_count",
    "query_token_count",
    "title_char_count",
    "title_token_count",
    "text_char_count",
    "text_token_count",
    "title_overlap",
    "text_overlap",
    "any_overlap",
    "title_jaccard",
    "text_jaccard",
    "exact_phrase_title",
    "source_cost_log1p",
    "history_click_count_log1p",
    "history_source_cost_log1p",
    "history_query_present",
    "history_region_present",
]


def feature_names(generator_names: Sequence[str]) -> list[str]:
    result = list(BASE_FEATURES)
    for source in generator_names:
        result.extend(
            [
                f"{source}__present",
                f"{source}__reciprocal_rank",
                f"{source}__log_rank",
                f"{source}__score",
                f"{source}__score_z",
                f"{source}__score_minmax",
            ]
        )
    return result


def _overlap(left: set[str], right: set[str]) -> tuple[float, float]:
    intersection = len(left & right)
    union = len(left | right)
    return float(intersection), float(intersection / union if union else 0.0)


def extract_feature_rows(
    example: dict[str, Any],
    candidates: Sequence[dict[str, Any]],
    generator_names: Sequence[str],
) -> list[list[float]]:
    names = feature_names(generator_names)
    score_stats: dict[str, tuple[float, float, float, float]] = {}
    for source in generator_names:
        values = [
            float(candidate["retrieval"][source]["score"])
            for candidate in candidates
            if source in candidate.get("retrieval", {})
            and candidate["retrieval"][source].get("score") is not None
        ]
        if values:
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / len(values)
            score_stats[source] = (mean, math.sqrt(variance), min(values), max(values))
        else:
            score_stats[source] = (0.0, 0.0, 0.0, 0.0)

    query = normalize(example.get("query"))
    query_tokens = set(tokenize(query))
    rows = []
    for candidate in candidates:
        title = normalize(candidate.get("title"))
        body = normalize(candidate.get("text"))
        title_tokens = set(tokenize(title))
        body_tokens = set(tokenize(body))
        title_overlap, title_jaccard = _overlap(query_tokens, title_tokens)
        text_overlap, text_jaccard = _overlap(query_tokens, body_tokens)
        history_retrieval = candidate.get("retrieval", {}).get("history", {})
        history_contributions = history_retrieval.get("contributions") or {}
        history_sources = history_contributions.get("history") or {}
        values: dict[str, float] = {
            "rrf_score": float(candidate.get("rrf_score", 0.0)),
            "source_count": float(candidate.get("source_count", 0)),
            "query_char_count": float(len(query)),
            "query_token_count": float(len(query_tokens)),
            "title_char_count": float(len(title)),
            "title_token_count": float(len(title_tokens)),
            "text_char_count": float(len(body)),
            "text_token_count": float(len(body_tokens)),
            "title_overlap": title_overlap,
            "text_overlap": text_overlap,
            "any_overlap": float(len(query_tokens & (title_tokens | body_tokens))),
            "title_jaccard": title_jaccard,
            "text_jaccard": text_jaccard,
            "exact_phrase_title": float(bool(query) and query in title),
            "source_cost_log1p": math.log1p(max(0.0, float(candidate.get("source_cost", 0.0)))),
            "history_click_count_log1p": math.log1p(
                max(0.0, float(history_contributions.get("click_count", 0.0)))
            ),
            "history_source_cost_log1p": math.log1p(
                max(0.0, float(history_contributions.get("source_cost_sum", 0.0)))
            ),
            "history_query_present": float("query" in history_sources),
            "history_region_present": float("query_region" in history_sources),
        }
        retrieval = candidate.get("retrieval", {})
        for source in generator_names:
            source_data = retrieval.get(source)
            present = source_data is not None
            source_score = (
                float(source_data["score"])
                if present and source_data.get("score") is not None
                else 0.0
            )
            mean, std, minimum, maximum = score_stats[source]
            values[f"{source}__present"] = float(present)
            values[f"{source}__reciprocal_rank"] = (
                float(source_data["reciprocal_rank"]) if present else 0.0
            )
            values[f"{source}__log_rank"] = (
                math.log1p(float(source_data["rank"])) if present else 0.0
            )
            values[f"{source}__score"] = source_score
            values[f"{source}__score_z"] = (
                (source_score - mean) / std if present and std > 1e-12 else 0.0
            )
            values[f"{source}__score_minmax"] = (
                (source_score - minimum) / (maximum - minimum)
                if present and maximum > minimum
                else 0.0
            )
        rows.append([values[name] for name in names])
    return rows
