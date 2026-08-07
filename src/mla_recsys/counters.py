from __future__ import annotations

from collections.abc import Iterable
from typing import Any


WEEK_SECONDS = 7 * 24 * 60 * 60
MONDAY_OFFSET = 4 * 24 * 60 * 60


def week_start(timestamp: int) -> int:
    return ((int(timestamp) - MONDAY_OFFSET) // WEEK_SECONDS) * WEEK_SECONDS + MONDAY_OFFSET


def past_only_snapshot(
    rows: Iterable[dict[str, Any]],
    *,
    row_timestamp: int,
    timestamp_key: str = "show_time",
) -> list[dict[str, Any]]:
    """Return events in weeks strictly before the row's calendar week."""

    cutoff = week_start(row_timestamp)
    return [row for row in rows if week_start(int(row[timestamp_key])) < cutoff]


def validate_scope(scope: str, artifact_scope: str) -> None:
    if scope != artifact_scope:
        raise ValueError(f"Counter scope mismatch: requested={scope}, artifact={artifact_scope}")

