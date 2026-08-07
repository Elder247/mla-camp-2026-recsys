from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mla_recsys.counters import past_only_snapshot, validate_scope, week_start  # noqa: E402
from mla_recsys.data import stable_partition, temporal_split  # noqa: E402
from mla_recsys.training_data import attach_targets, freeze_natural_pool  # noqa: E402


class TemporalContractTest(unittest.TestCase):
    def test_request_split_is_temporal_and_group_atomic(self) -> None:
        fit, holdout = temporal_split(
            [
                {"request_id": "a", "show_time": 9, "show_time_min": 9, "show_time_max": 9},
                {"request_id": "b", "show_time": 10, "show_time_min": 10, "show_time_max": 10},
            ],
            boundary=10,
        )
        self.assertEqual([row["request_id"] for row in fit], ["a"])
        self.assertEqual([row["request_id"] for row in holdout], ["b"])
        with self.assertRaises(ValueError):
            temporal_split(
                [{"request_id": "cross", "show_time": 9, "show_time_min": 9, "show_time_max": 10}],
                boundary=10,
            )

    def test_positive_after_pool_is_not_injected(self) -> None:
        rows = [
            {"banner_id": banner_id, "pre_rank": rank}
            for rank, banner_id in enumerate((10, 20, 30), start=1)
        ]
        natural = freeze_natural_pool(rows, 2)
        labeled = attach_targets(natural, {30: 1_000_000.0})
        self.assertEqual([row["banner_id"] for row in labeled], [10, 20])
        self.assertFalse(any(row["is_positive"] for row in labeled))

    def test_weekly_snapshot_excludes_current_and_future_week(self) -> None:
        row_time = 1_800_000_000
        start = week_start(row_time)
        rows = [
            {"show_time": start - 1, "name": "past"},
            {"show_time": start, "name": "current"},
            {"show_time": start + 604800, "name": "future"},
        ]
        self.assertEqual(
            [row["name"] for row in past_only_snapshot(rows, row_timestamp=row_time)],
            ["past"],
        )
        with self.assertRaises(ValueError):
            validate_scope("offline", "full")

    def test_partition_is_stable(self) -> None:
        self.assertEqual(stable_partition("request", 32), stable_partition("request", 32))


if __name__ == "__main__":
    unittest.main()

