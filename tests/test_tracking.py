from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from mla_recsys.tracking import UnderdeepTracker, numeric_metrics


class TrackingTest(unittest.TestCase):
    def test_disabled_tracker_always_writes_local_backup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tracker = UnderdeepTracker(
                artifact_dir=Path(directory),
                tracking_cfg={"enabled": False},
                run_name="unit-test",
                description="unit",
                parameters={"mode": "smoke"},
                tags=["test"],
            )
            tracker.log(3, {"stage/wall_seconds": 1.25})
            tracker.log_summary({"primary/sourcecost_recall@50": 0.61})
            tracker.close()
            rows = [
                json.loads(line)
                for line in (Path(directory) / "underdeep_metrics.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
        self.assertEqual(
            [row["event"] for row in rows],
            ["init", "metrics", "summary", "finish"],
        )

    def test_numeric_metrics_flattens_nested_results(self) -> None:
        self.assertEqual(
            numeric_metrics({"a": {"b": 2}, "ok": True}, prefix="run"),
            {"run/a/b": 2.0, "run/ok": 1.0},
        )


if __name__ == "__main__":
    unittest.main()
