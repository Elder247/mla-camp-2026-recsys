#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterable

import pyarrow.parquet as pq


TOKEN_RE = re.compile(r"[0-9a-zа-яё]+", re.IGNORECASE)


def raw(value: object) -> str:
    return "" if value is None else str(value)


def trim_lower(value: object) -> str:
    return raw(value).strip().lower()


def nfkc(value: object) -> str:
    return unicodedata.normalize("NFKC", raw(value))


def whitespace(value: object) -> str:
    return " ".join(nfkc(value).lower().split())


def project_normalize(value: object) -> str:
    return " ".join(TOKEN_RE.findall(nfkc(value).lower().replace("ё", "е")))


def token_canonical(value: object) -> str:
    return " ".join(sorted(project_normalize(value).split()))


NORMALIZERS: dict[str, Callable[[object], str]] = {
    "raw": raw,
    "trim_lower": trim_lower,
    "nfkc": nfkc,
    "whitespace": whitespace,
    "project": project_normalize,
    "token_canonical": token_canonical,
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Leakage-safe exact-query history audit")
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--fallback", type=Path, required=True)
    parser.add_argument(
        "--prior-requests",
        type=Path,
        help="Additional labelled requests that strictly precede --requests",
    )
    parser.add_argument("--test-requests", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefixes", default="10,20,30,40,50")
    parser.add_argument("--exact-ks", default="10,50,100")
    return parser.parse_args()


def request_rows(path: Path) -> list[dict]:
    return pq.read_table(path).to_pylist()


def prediction_map(path: Path) -> dict[int, list[int]]:
    rows = pq.read_table(path, columns=["HitLogID", "BannerID"]).to_pylist()
    return {int(row["HitLogID"]): [int(value) for value in row["BannerID"]] for row in rows}


def scan_history(path: Path, target_forms: dict[str, set[str]]) -> tuple[dict[str, dict[str, dict[int, list[float]]]], dict]:
    retained: dict[str, dict[str, dict[int, list[float]]]] = {
        name: defaultdict(dict) for name in NORMALIZERS
    }
    observed_forms: dict[str, set[str]] = {name: set() for name in NORMALIZERS}
    maximum_time = 0
    raw_query_count = 0
    parquet = pq.ParquetFile(path)
    columns = ["key_type", "search_query", "banner_id", "click_count", "source_cost_sum", "last_show_time"]
    for batch in parquet.iter_batches(batch_size=131_072, columns=columns):
        for row in batch.to_pylist():
            if row["key_type"] != "query":
                continue
            query = raw(row["search_query"])
            raw_query_count += 1
            banner_id = int(row["banner_id"])
            click_count = int(row["click_count"] or 0)
            source_cost_sum = float(row["source_cost_sum"] or 0.0)
            last_show_time = int(row["last_show_time"] or 0)
            maximum_time = max(maximum_time, last_show_time)
            for name, normalizer in NORMALIZERS.items():
                key = normalizer(query)
                observed_forms[name].add(key)
                if key not in target_forms[name]:
                    continue
                stats = retained[name][key].setdefault(banner_id, [0.0, 0.0, 0.0])
                stats[0] += click_count
                stats[1] += source_cost_sum
                stats[2] = max(stats[2], last_show_time)
    metadata = {
        "query_rows_scanned": raw_query_count,
        "maximum_last_show_time": maximum_time,
        "unique_history_keys": {name: len(values) for name, values in observed_forms.items()},
        "observed_forms": observed_forms,
    }
    return retained, metadata


def add_prior_requests(
    history: dict[str, dict[str, dict[int, list[float]]]],
    observed_forms: dict[str, set[str]],
    target_forms: dict[str, set[str]],
    rows: list[dict],
) -> dict:
    maximum_time = 0
    added_clicks = 0
    for row in rows:
        show_time = int(row["show_time"] or 0)
        maximum_time = max(maximum_time, show_time)
        for name, normalizer in NORMALIZERS.items():
            key = normalizer(row["query"])
            observed_forms[name].add(key)
            if key not in target_forms[name]:
                continue
            by_banner = history[name][key]
            for banner_value, cost_value in zip(
                row["clicked_banner_ids"], row["clicked_source_costs"]
            ):
                banner_id = int(banner_value)
                stats = by_banner.setdefault(banner_id, [0.0, 0.0, 0.0])
                stats[0] += 1.0
                stats[1] += float(cost_value)
                stats[2] = max(stats[2], show_time)
                added_clicks += 1
    return {
        "requests": len(rows),
        "clicks_added_across_normalizations": added_clicks,
        "maximum_show_time": maximum_time,
    }


def order_rows(by_banner: dict[int, list[float]], mode: str) -> list[tuple[int, list[float]]]:
    rows = list(by_banner.items())
    if mode == "source_cost_sum":
        return sorted(rows, key=lambda row: (-row[1][1], -row[1][0], -row[1][2], row[0]))
    if mode == "click_count":
        return sorted(rows, key=lambda row: (-row[1][0], -row[1][1], -row[1][2], row[0]))
    if mode == "recency":
        return sorted(rows, key=lambda row: (-row[1][2], -row[1][0], -row[1][1], row[0]))
    if mode == "count_cost025":
        def score(row: tuple[int, list[float]]) -> float:
            count, cost_sum, _ = row[1]
            average_cost = cost_sum / max(count, 1.0)
            return math.log1p(count) + 0.25 * math.log1p(average_cost)
        return sorted(rows, key=lambda row: (-score(row), -row[1][1], row[0]))
    if mode == "rrf":
        ranks: dict[int, float] = defaultdict(float)
        sorters = (
            lambda row: (-row[1][1], -row[1][0], row[0]),
            lambda row: (-row[1][0], -row[1][1], row[0]),
            lambda row: (-row[1][2], -row[1][0], row[0]),
        )
        for sorter in sorters:
            for rank, row in enumerate(sorted(rows, key=sorter), start=1):
                ranks[row[0]] += 1.0 / (20.0 + rank)
        return sorted(rows, key=lambda row: (-ranks[row[0]], row[0]))
    raise ValueError(mode)


def rankings(history: dict[str, dict[str, dict[int, list[float]]]]) -> dict[str, dict[str, dict[str, list[tuple[int, list[float]]]]]]:
    result = {}
    for normalization, by_query in history.items():
        result[normalization] = {}
        for mode in ("source_cost_sum", "click_count", "count_cost025", "recency", "rrf"):
            result[normalization][mode] = {
                query: order_rows(by_banner, mode) for query, by_banner in by_query.items()
            }
    return result


def deduplicated_prefix(rows: Iterable[tuple[int, list[float]]], limit: int, cutoff: int | None) -> list[int]:
    if limit <= 0:
        return []
    output = []
    seen = set()
    for banner_id, stats in rows:
        if cutoff is not None and int(stats[2]) >= cutoff:
            continue
        if banner_id in seen:
            continue
        output.append(banner_id)
        seen.add(banner_id)
        if len(output) >= limit:
            break
    return output


def fill(prefix: list[int], fallback: list[int], limit: int = 50) -> list[int]:
    output = []
    seen = set()
    for banner_id in prefix + fallback:
        if banner_id in seen:
            continue
        output.append(banner_id)
        seen.add(banner_id)
        if len(output) == limit:
            break
    return output


def metric(rows: list[dict], predictions: dict[int, list[int]], k: int) -> dict[str, float]:
    hits = total = 0
    hit_cost = total_cost = 0.0
    for row in rows:
        predicted = set(predictions[int(row["hit_log_id"])][:k])
        for banner_id, cost in zip(row["clicked_banner_ids"], row["clicked_source_costs"]):
            total += 1
            total_cost += float(cost)
            if int(banner_id) in predicted:
                hits += 1
                hit_cost += float(cost)
    return {
        "recall": hits / total if total else 0.0,
        "source_cost_recall": hit_cost / total_cost if total_cost else 0.0,
        "hits": hits,
        "targets": total,
        "hit_source_cost": hit_cost,
        "total_source_cost": total_cost,
    }


def compare(rows: list[dict], candidate: dict[int, list[int]], baseline: dict[int, list[int]]) -> dict:
    new_hits = reverse_hits = 0
    new_cost = reverse_cost = 0.0
    for row in rows:
        hit_log_id = int(row["hit_log_id"])
        left = set(candidate[hit_log_id])
        right = set(baseline[hit_log_id])
        for banner_id, cost in zip(row["clicked_banner_ids"], row["clicked_source_costs"]):
            banner_id = int(banner_id)
            if banner_id in left and banner_id not in right:
                new_hits += 1
                new_cost += float(cost)
            if banner_id in right and banner_id not in left:
                reverse_hits += 1
                reverse_cost += float(cost)
    candidate_union = {banner for values in candidate.values() for banner in values}
    baseline_union = {banner for values in baseline.values() for banner in values}
    return {
        "new_hits": new_hits,
        "reverse_only_hits": reverse_hits,
        "new_hit_source_cost": new_cost,
        "reverse_only_source_cost": reverse_cost,
        "new_candidate_ids_vs_baseline_union": len(candidate_union - baseline_union),
        "candidate_union_ids": len(candidate_union),
        "baseline_union_ids": len(baseline_union),
    }


def main() -> int:
    args = arguments()
    requests = request_rows(args.requests)
    test_requests = request_rows(args.test_requests) if args.test_requests else []
    all_requests = requests + test_requests
    target_forms = {
        name: {normalizer(row["query"]) for row in all_requests}
        for name, normalizer in NORMALIZERS.items()
    }
    history, history_meta = scan_history(args.history, target_forms)
    observed_forms = history_meta.pop("observed_forms")
    prior_rows = request_rows(args.prior_requests) if args.prior_requests else []
    prior_meta = add_prior_requests(
        history, observed_forms, target_forms, prior_rows
    ) if prior_rows else {"requests": 0, "clicks_added_across_normalizations": 0, "maximum_show_time": 0}
    coverage = {}
    for name, normalizer in NORMALIZERS.items():
        coverage[name] = {}
        for label, rows in (("temporal", requests), ("test", test_requests)):
            matched = sum(normalizer(row["query"]) in observed_forms[name] for row in rows)
            coverage[name][label] = {
                "matched_requests": matched,
                "requests": len(rows),
                "share": matched / len(rows) if rows else 0.0,
            }

    ordered = rankings(history)
    baseline = prediction_map(args.fallback)
    ordered_requests = sorted(requests, key=lambda row: int(row["show_time"] or 0))
    midpoint = len(ordered_requests) // 2
    splits = {"early": ordered_requests[:midpoint], "late": ordered_requests[midpoint:], "full": ordered_requests}
    prefixes = [int(value) for value in args.prefixes.split(",")]
    exact_ks = [int(value) for value in args.exact_ks.split(",")]
    screens = []

    for normalization in ("raw", "project"):
        normalizer = NORMALIZERS[normalization]
        for mode in ("source_cost_sum", "click_count", "count_cost025", "recency", "rrf"):
            exact_lists = ordered[normalization][mode]
            for prefix_size in prefixes:
                combined = {}
                exact_only = {}
                lengths = []
                for row in requests:
                    hit_log_id = int(row["hit_log_id"])
                    key = normalizer(row["query"])
                    exact = deduplicated_prefix(
                        exact_lists.get(key, ()), max(max(exact_ks), prefix_size), int(row["show_time"] or 0)
                    )
                    lengths.append(len(exact))
                    exact_only[hit_log_id] = exact
                    combined[hit_log_id] = fill(exact[:prefix_size], baseline[hit_log_id])
                result = {
                    "normalization": normalization,
                    "mode": mode,
                    "prefix": prefix_size,
                    "coverage": sum(length > 0 for length in lengths) / len(lengths),
                    "mean_exact_length": statistics.fmean(lengths),
                    "median_exact_length": statistics.median(lengths),
                    "splits": {name: metric(rows, combined, 50) for name, rows in splits.items()},
                }
                if prefix_size == 30:
                    result["comparison_to_fallback"] = compare(requests, combined, baseline)
                    result["exact_only"] = {
                        str(k): metric(requests, exact_only, k) for k in exact_ks
                    }
                screens.append(result)

    baseline_metrics = {name: metric(rows, baseline, 50) for name, rows in splits.items()}
    minimum_request_time = min(int(row["show_time"] or 0) for row in requests)
    maximum_available_history_time = max(
        history_meta["maximum_last_show_time"], prior_meta["maximum_show_time"]
    )
    report = {
        "inputs": {"history": str(args.history), "requests": str(args.requests), "fallback": str(args.fallback)},
        "history": history_meta,
        "prior_requests": prior_meta,
        "leakage_contract": {
            "maximum_history_last_show_time": history_meta["maximum_last_show_time"],
            "maximum_prior_request_show_time": prior_meta["maximum_show_time"],
            "maximum_available_history_time": maximum_available_history_time,
            "minimum_temporal_request_show_time": minimum_request_time,
            "all_available_history_precedes_temporal_requests": maximum_available_history_time < minimum_request_time,
        },
        "coverage": coverage,
        "baseline": baseline_metrics,
        "screens": screens,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
