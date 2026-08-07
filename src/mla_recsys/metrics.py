from __future__ import annotations

from collections.abc import Iterable
from typing import Any


MISS_RANK = 2**31 - 1


def recall_metrics(
    records: Iterable[dict[str, Any]], cutoffs: Iterable[int]
) -> dict[str, dict[str, Any]]:
    values = list(records)
    total_cost = sum(float(item["source_cost"]) for item in values)
    result = {}
    for cutoff in cutoffs:
        hits = [item for item in values if int(item["rank"]) <= int(cutoff)]
        hit_cost = sum(float(item["source_cost"]) for item in hits)
        result[str(cutoff)] = {
            "recall": len(hits) / len(values) if values else 0.0,
            "sourcecost_recall": hit_cost / total_cost if total_cost else 0.0,
            "hits": len(hits),
            "clicks": len(values),
            "sourcecost_hit": hit_cost,
            "sourcecost_total": total_cost,
        }
    return result


def truth_pairs(requests: Iterable[dict[str, Any]]) -> dict[tuple[str, int], float]:
    result = {}
    for request in requests:
        for banner_id, source_cost in zip(
            request["clicked_banner_ids"], request["clicked_source_costs"]
        ):
            result[(str(request["request_id"]), int(banner_id))] = float(source_cost)
    return result


def records_from_found(
    truth: dict[tuple[str, int], float],
    found: dict[tuple[str, int], int],
) -> list[dict[str, Any]]:
    return [
        {"rank": int(found.get(pair, MISS_RANK)), "source_cost": source_cost}
        for pair, source_cost in truth.items()
    ]

