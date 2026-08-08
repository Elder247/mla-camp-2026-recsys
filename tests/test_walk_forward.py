from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mla_recsys.counters import week_start  # noqa: E402
from mla_recsys.data import read_request_parquet  # noqa: E402
import pyarrow.parquet as pq
from two_tower_v2.walk_forward import (  # noqa: E402
    extract_oof_requests,
    validate_week_sequence,
    walk_forward_events,
)


class WalkForwardTest(unittest.TestCase):
    def test_contract_predicts_before_same_week_update(self) -> None:
        first = week_start(1_780_000_000)
        weeks = [first, first + 604800, first + 2 * 604800]
        events = walk_forward_events(weeks)
        self.assertEqual(events[0]["predict_state"], "random")
        self.assertEqual(events[1]["predict_state"], "after_week_0")
        self.assertEqual(
            events[2]["order"],
            ["predict", "freeze_pool", "attach_labels", "update"],
        )
        self.assertEqual(validate_week_sequence(weeks), weeks)
        with self.assertRaises(ValueError):
            validate_week_sequence([weeks[1], weeks[0]])

    def test_oof_sampling_is_bounded_and_keeps_multi_click_targets(self) -> None:
        first = week_start(1_780_000_000)
        rows = [
            {
                "uniq_id": 1,
                "show_time": first + 10,
                "query": "coffee",
                "region_id": 1,
                "banner_id": 10,
                "source_cost": 100.0,
            },
            {
                "uniq_id": 1,
                "show_time": first + 10,
                "query": "coffee",
                "region_id": 1,
                "banner_id": 11,
                "source_cost": 200.0,
            },
            {
                "uniq_id": 2,
                "show_time": first + 20,
                "query": "tea",
                "region_id": 2,
                "banner_id": 12,
                "source_cost": 300.0,
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "oof.parquet"
            history_output = Path(directory) / "history.parquet"
            report = extract_oof_requests(
                rows=rows,
                weeks=[first],
                requests_per_week=2,
                output=output,
                history_output=history_output,
            )
            materialized = read_request_parquet(output)
            history = pq.read_table(history_output)
        self.assertEqual(report["requests"], 2)
        self.assertEqual(report["history_rows"], 3)
        self.assertEqual(history.num_rows, 3)
        self.assertEqual(history["show_time"].to_pylist(), [first + 10, first + 10, first + 20])
        coffee = next(row for row in materialized if row["query"] == "coffee")
        self.assertEqual(coffee["clicked_banner_ids"], [10, 11])
        self.assertEqual(coffee["clicked_source_costs"], [100.0, 200.0])


if __name__ == "__main__":
    unittest.main()
