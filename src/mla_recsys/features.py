from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from .text import normalize, tokenize


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

V2_GROUP_FEATURES: dict[str, list[str]] = {
    "retrieval_provenance_v2": [
        "retrieval_min_rank",
        "retrieval_mean_rank",
        "lexical_only",
        "neural_only",
        "history_only",
        "history_and_neural",
        "history_and_lexical",
    ],
    "request_context_v2": [
        "region_id_numeric",
        "region_missing",
        "user_present",
        "device_hash_bucket",
        "device_missing",
        "age_numeric",
        "age_missing",
        "age_bucket",
        "gender_numeric",
        "gender_missing",
        "query_digit_share",
        "query_cyrillic_share",
        "query_latin_share",
    ],
    "candidate_static_v2": [
        "source_cost_raw",
        "source_cost_missing",
        "product_price_raw",
        "product_price_log1p",
        "product_price_missing",
        "group_hash_bucket",
        "group_missing",
        "client_hash_bucket",
        "client_missing",
        "domain_hash_bucket",
        "domain_missing",
        "url_char_count",
        "url_token_count",
        "url_digit_share",
        "title_digit_share",
        "text_digit_share",
        "title_missing",
        "text_missing",
        "group_banner_count_log1p",
        "domain_banner_count_log1p",
        "group_source_cost_mean",
        "domain_source_cost_mean",
        "source_cost_vs_group_mean",
        "source_cost_vs_domain_mean",
    ],
    "text_match_v2": [
        "url_overlap",
        "url_jaccard",
        "query_title_coverage",
        "query_text_coverage",
        "query_url_coverage",
        "title_prefix_match",
        "text_prefix_match",
        "url_prefix_match",
        "title_text_overlap_gap",
        "query_title_char_trigram_jaccard",
    ],
    "cross_features_v1": [
        "query_pop_x_banner_pop",
        "region_banner_minus_banner_pop",
        "user_banner_minus_banner_pop",
        "user_group_minus_group_pop",
        "source_cost_x_banner_sc_avg",
    ],
}


def feature_names(
    generator_names: Sequence[str],
    *,
    version: str = "feature_v1",
    enabled_groups: Sequence[str] = (),
    counter_families: Sequence[str] = (),
    counter_windows_days: Sequence[int] = (),
) -> list[str]:
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
    if version == "feature_v1":
        return result
    for group in enabled_groups:
        result.extend(V2_GROUP_FEATURES.get(str(group), []))
    if "weekly_counters_v1" in enabled_groups:
        for family in counter_families:
            for days in counter_windows_days:
                label = "all" if int(days) == 0 else f"{int(days)}d"
                prefix = f"counter__{family}__{label}"
                result.extend(
                    [
                        f"{prefix}__clicks_log1p",
                        f"{prefix}__sc_sum_log1p",
                        f"{prefix}__sc_avg",
                        f"{prefix}__age_days",
                        f"{prefix}__present",
                    ]
                )
    if len(result) != len(set(result)):
        raise ValueError("Feature names must be unique")
    return result


def _overlap(left: set[str], right: set[str]) -> tuple[float, float]:
    intersection = len(left & right)
    union = len(left | right)
    return float(intersection), float(intersection / union if union else 0.0)


def _share(text: str, pattern: str) -> float:
    return float(len(re.findall(pattern, text)) / len(text)) if text else 0.0


def _hash_bucket(value: Any, buckets: int = 1_000_003) -> float:
    text = str(value or "")
    if not text:
        return 0.0
    digest = hashlib.sha1(text.encode("utf-8")).digest()
    return float(int.from_bytes(digest[:8], "little") % buckets + 1)


def _trigrams(text: str) -> set[str]:
    compact = f"  {text} "
    return {compact[index : index + 3] for index in range(max(0, len(compact) - 2))}


def _ratio(value: float, denominator: float) -> float:
    return float(value / denominator) if abs(denominator) > 1e-12 else 0.0


def extract_feature_rows(
    example: dict[str, Any],
    candidates: Sequence[dict[str, Any]],
    generator_names: Sequence[str],
    *,
    version: str = "feature_v1",
    enabled_groups: Sequence[str] = (),
    counter_families: Sequence[str] = (),
    counter_windows_days: Sequence[int] = (),
) -> list[list[float]]:
    names = feature_names(
        generator_names,
        version=version,
        enabled_groups=enabled_groups,
        counter_families=counter_families,
        counter_windows_days=counter_windows_days,
    )
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
    context = example.get("context") or {}
    rows = []
    for candidate in candidates:
        title = normalize(candidate.get("title"))
        body = normalize(candidate.get("text"))
        url = normalize(candidate.get("url"))
        title_tokens = set(tokenize(title))
        body_tokens = set(tokenize(body))
        url_tokens = set(tokenize(url))
        title_overlap, title_jaccard = _overlap(query_tokens, title_tokens)
        text_overlap, text_jaccard = _overlap(query_tokens, body_tokens)
        url_overlap, url_jaccard = _overlap(query_tokens, url_tokens)
        history_retrieval = candidate.get("retrieval", {}).get("history", {})
        history_contributions = history_retrieval.get("contributions") or {}
        history_sources = history_contributions.get("history") or {}
        history_click_count = float(
            candidate.get(
                "history_click_count",
                history_contributions.get("click_count", 0.0),
            )
            or 0.0
        )
        history_source_cost_sum = float(
            candidate.get(
                "history_source_cost_sum",
                history_contributions.get("source_cost_sum", 0.0),
            )
            or 0.0
        )
        history_query_present = bool(
            candidate.get("history_query_present", "query" in history_sources)
        )
        history_region_present = bool(
            candidate.get("history_region_present", "query_region" in history_sources)
        )
        source_cost = float(candidate.get("source_cost", 0.0) or 0.0)
        product_price = float(candidate.get("product_price", 0.0) or 0.0)
        group_mean = float(candidate.get("group_source_cost_mean", 0.0) or 0.0)
        domain_mean = float(candidate.get("domain_source_cost_mean", 0.0) or 0.0)
        retrieval = candidate.get("retrieval", {})
        ranks = [float(item["rank"]) for item in retrieval.values()]
        lexical = any("tfidf" in name or "bm25" in name for name in retrieval)
        neural = any("tower" in name or "neural" in name for name in retrieval)
        historical = any(
            "history" in name or "pop" in name for name in retrieval
        )
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
            "source_cost_log1p": math.log1p(max(0.0, source_cost)),
            "history_click_count_log1p": math.log1p(
                max(0.0, history_click_count)
            ),
            "history_source_cost_log1p": math.log1p(
                max(0.0, history_source_cost_sum)
            ),
            "history_query_present": float(history_query_present),
            "history_region_present": float(history_region_present),
            "retrieval_min_rank": min(ranks) if ranks else 0.0,
            "retrieval_mean_rank": sum(ranks) / len(ranks) if ranks else 0.0,
            "lexical_only": float(lexical and not neural and not historical),
            "neural_only": float(neural and not lexical and not historical),
            "history_only": float(historical and not lexical and not neural),
            "history_and_neural": float(historical and neural),
            "history_and_lexical": float(historical and lexical),
            "region_id_numeric": float(context.get("region_id") or 0),
            "region_missing": float(context.get("region_id") is None),
            "user_present": float(context.get("crypta_id_v2") is not None),
            "device_hash_bucket": _hash_bucket(context.get("device")),
            "device_missing": float(not context.get("device")),
            "age_numeric": float(context.get("age") or 0),
            "age_missing": float(context.get("age") is None),
            "age_bucket": float(int(context.get("age") or 0) // 10),
            "gender_numeric": float(context.get("gender") or 0),
            "gender_missing": float(context.get("gender") is None),
            "query_digit_share": _share(query, r"[0-9]"),
            "query_cyrillic_share": _share(query, r"[а-яё]"),
            "query_latin_share": _share(query, r"[a-z]"),
            "source_cost_raw": source_cost,
            "source_cost_missing": float(candidate.get("source_cost") is None),
            "product_price_raw": product_price,
            "product_price_log1p": math.log1p(max(0.0, product_price)),
            "product_price_missing": float(candidate.get("product_price") is None),
            "group_hash_bucket": _hash_bucket(candidate.get("group_id")),
            "group_missing": float(candidate.get("group_id") is None),
            "client_hash_bucket": _hash_bucket(candidate.get("client_id")),
            "client_missing": float(candidate.get("client_id") is None),
            "domain_hash_bucket": _hash_bucket(candidate.get("domain")),
            "domain_missing": float(not candidate.get("domain")),
            "url_char_count": float(len(url)),
            "url_token_count": float(len(url_tokens)),
            "url_digit_share": _share(url, r"[0-9]"),
            "title_digit_share": _share(title, r"[0-9]"),
            "text_digit_share": _share(body, r"[0-9]"),
            "title_missing": float(not title),
            "text_missing": float(not body),
            "group_banner_count_log1p": math.log1p(
                float(candidate.get("group_banner_count", 0) or 0)
            ),
            "domain_banner_count_log1p": math.log1p(
                float(candidate.get("domain_banner_count", 0) or 0)
            ),
            "group_source_cost_mean": group_mean,
            "domain_source_cost_mean": domain_mean,
            "source_cost_vs_group_mean": _ratio(source_cost, group_mean),
            "source_cost_vs_domain_mean": _ratio(source_cost, domain_mean),
            "url_overlap": url_overlap,
            "url_jaccard": url_jaccard,
            "query_title_coverage": _ratio(title_overlap, len(query_tokens)),
            "query_text_coverage": _ratio(text_overlap, len(query_tokens)),
            "query_url_coverage": _ratio(url_overlap, len(query_tokens)),
            "title_prefix_match": float(bool(query) and title.startswith(query)),
            "text_prefix_match": float(bool(query) and body.startswith(query)),
            "url_prefix_match": float(bool(query) and url.startswith(query)),
            "title_text_overlap_gap": title_overlap - text_overlap,
            "query_title_char_trigram_jaccard": _overlap(
                _trigrams(query), _trigrams(title)
            )[1],
        }
        values.update(
            (str(name), float(value))
            for name, value in (candidate.get("counter_features") or {}).items()
        )
        query_pop = values.get("counter__query__all__clicks_log1p", 0.0)
        banner_pop = values.get("counter__banner__all__clicks_log1p", 0.0)
        group_pop = values.get("counter__group__all__clicks_log1p", 0.0)
        values.update(
            {
                "query_pop_x_banner_pop": query_pop * banner_pop,
                "region_banner_minus_banner_pop": values.get(
                    "counter__region_banner__all__clicks_log1p", 0.0
                )
                - banner_pop,
                "user_banner_minus_banner_pop": values.get(
                    "counter__user_banner__all__clicks_log1p", 0.0
                )
                - banner_pop,
                "user_group_minus_group_pop": values.get(
                    "counter__user_group__all__clicks_log1p", 0.0
                )
                - group_pop,
                "source_cost_x_banner_sc_avg": source_cost
                * values.get("counter__banner__all__sc_avg", 0.0),
            }
        )
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
        rows.append([float(values.get(name, 0.0)) for name in names])
    return rows
