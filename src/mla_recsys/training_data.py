from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def freeze_natural_pool(rows: Iterable[dict[str, Any]], pool_size: int) -> list[dict[str, Any]]:
    if pool_size <= 0:
        raise ValueError("pool_size must be positive")
    ordered = sorted(
        rows,
        key=lambda row: (int(row["pre_rank"]), int(row["banner_id"])),
    )
    return ordered[:pool_size]


def attach_targets(
    natural_pool: Iterable[dict[str, Any]],
    clicked: dict[int, float],
) -> list[dict[str, Any]]:
    """Join labels after cutoff; this function never changes pool membership."""

    result = []
    for raw in natural_pool:
        row = dict(raw)
        banner_id = int(row["banner_id"])
        source_cost = float(clicked.get(banner_id, 0.0))
        row["is_positive"] = banner_id in clicked
        row["target_source_cost"] = source_cost
        result.append(row)
    return result

