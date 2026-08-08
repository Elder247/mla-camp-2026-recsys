from __future__ import annotations

import bisect
import hashlib
import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pyarrow as pa
import pyarrow.parquet as pq

from common.text import normalize


WEEK_SECONDS = 7 * 24 * 60 * 60
MONDAY_OFFSET = 4 * 24 * 60 * 60

COUNTER_EVENT_SCHEMA = pa.schema(
    [
        pa.field("show_time", pa.uint64(), nullable=False),
        pa.field("banner_id", pa.uint64(), nullable=False),
        pa.field("group_id", pa.int64(), nullable=True),
        pa.field("domain", pa.string(), nullable=False),
        pa.field("query_key", pa.string(), nullable=False),
        pa.field("region_key", pa.string(), nullable=False),
        pa.field("user_key", pa.string(), nullable=False),
        pa.field("source_cost", pa.float64(), nullable=False),
    ]
)


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


def stable_text_key(value: Any) -> str:
    normalized = normalize(value)
    if not normalized:
        return ""
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:20]


def scalar_key(value: Any) -> str:
    return "" if value is None else str(value)


def url_domain(value: Any) -> str:
    text = str(value or "").strip().lower()
    if not text:
        return ""
    candidate = text if "://" in text else f"//{text}"
    try:
        return (urlsplit(candidate).hostname or "").removeprefix("www.")
    except ValueError:
        return ""


@dataclass(frozen=True)
class CounterStats:
    clicks: float = 0.0
    source_cost_sum: float = 0.0
    source_cost_avg: float = 0.0
    age_days: float = 0.0
    present: float = 0.0


class CounterLookup:
    """Small exact-ASOF lookup over click events from the configured fit scope.

    Every query uses timestamps strictly less than the row timestamp. Holdout and
    test naturally see a frozen fit state because their artifacts contain only
    train and full_train events respectively.
    """

    def __init__(self, events: Sequence[dict[str, Any]]) -> None:
        grouped: dict[tuple[str, str], list[tuple[int, float]]] = defaultdict(list)
        for event in events:
            timestamp = int(event["show_time"])
            cost = float(event["source_cost"])
            banner = scalar_key(event["banner_id"])
            group = scalar_key(event.get("group_id"))
            domain = str(event.get("domain") or "")
            query = str(event.get("query_key") or "")
            region = str(event.get("region_key") or "")
            user = str(event.get("user_key") or "")
            keys = {
                "banner": banner,
                "group": group,
                "domain": domain,
                "query": query,
                "region": region,
                "user": user,
                "query_banner": f"{query}|{banner}" if query else "",
                "region_banner": f"{region}|{banner}" if region else "",
                "user_banner": f"{user}|{banner}" if user else "",
                "user_group": f"{user}|{group}" if user and group else "",
            }
            for family, key in keys.items():
                if key:
                    grouped[(family, key)].append((timestamp, cost))

        self.timestamps: dict[tuple[str, str], list[int]] = {}
        self.prefix_cost: dict[tuple[str, str], list[float]] = {}
        for key, values in grouped.items():
            values.sort()
            self.timestamps[key] = [item[0] for item in values]
            prefix = [0.0]
            for _, cost in values:
                prefix.append(prefix[-1] + cost)
            self.prefix_cost[key] = prefix

    @classmethod
    def from_parquet(cls, path: Path) -> "CounterLookup":
        return cls(pq.read_table(path, schema=COUNTER_EVENT_SCHEMA).to_pylist())

    @staticmethod
    def entity_keys(request: dict[str, Any], candidate: dict[str, Any]) -> dict[str, str]:
        banner = scalar_key(candidate.get("banner_id"))
        group = scalar_key(candidate.get("group_id"))
        domain = str(candidate.get("domain") or "")
        query = stable_text_key(request.get("query"))
        region = scalar_key(request.get("region_id"))
        user = scalar_key(request.get("crypta_id_v2"))
        return {
            "banner": banner,
            "group": group,
            "domain": domain,
            "query": query,
            "region": region,
            "user": user,
            "query_banner": f"{query}|{banner}" if query else "",
            "region_banner": f"{region}|{banner}" if region else "",
            "user_banner": f"{user}|{banner}" if user else "",
            "user_group": f"{user}|{group}" if user and group else "",
        }

    def stats(
        self,
        family: str,
        key: str,
        *,
        row_timestamp: int | None,
        window_days: int,
        frozen_cutoff: int | None = None,
    ) -> CounterStats:
        if not key:
            return CounterStats()
        timestamps = self.timestamps.get((family, key), [])
        if not timestamps:
            return CounterStats()
        # bisect_left is the leakage boundary: events at the same timestamp are
        # excluded. Test rows use the persisted scope cutoff.
        cutoff = int(frozen_cutoff if row_timestamp is None else row_timestamp)
        right = bisect.bisect_left(timestamps, cutoff)
        if right == 0:
            return CounterStats()
        if int(window_days) > 0:
            start = cutoff - int(window_days) * 24 * 60 * 60
            left = bisect.bisect_left(timestamps, start, 0, right)
        else:
            left = 0
        count = right - left
        if count <= 0:
            return CounterStats()
        costs = self.prefix_cost[(family, key)]
        total = costs[right] - costs[left]
        return CounterStats(
            clicks=float(count),
            source_cost_sum=float(total),
            source_cost_avg=float(total / count),
            age_days=float((cutoff - timestamps[right - 1]) / (24 * 60 * 60)),
            present=1.0,
        )


def counter_feature_values(
    lookup: CounterLookup,
    request: dict[str, Any],
    candidate: dict[str, Any],
    *,
    families: Sequence[str],
    windows_days: Sequence[int],
    frozen_cutoff: int,
) -> dict[str, float]:
    keys = lookup.entity_keys(request, candidate)
    values: dict[str, float] = {}
    for family in families:
        for days in windows_days:
            label = "all" if int(days) == 0 else f"{int(days)}d"
            stats = lookup.stats(
                family,
                keys.get(family, ""),
                row_timestamp=request.get("show_time"),
                window_days=int(days),
                frozen_cutoff=frozen_cutoff,
            )
            prefix = f"counter__{family}__{label}"
            values[f"{prefix}__clicks_log1p"] = math.log1p(stats.clicks)
            values[f"{prefix}__sc_sum_log1p"] = math.log1p(
                max(0.0, stats.source_cost_sum)
            )
            values[f"{prefix}__sc_avg"] = stats.source_cost_avg
            values[f"{prefix}__age_days"] = stats.age_days
            values[f"{prefix}__present"] = stats.present
    return values
