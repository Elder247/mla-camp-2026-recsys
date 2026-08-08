from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from generators.history import rank  # noqa: E402


class HistoryRankTest(unittest.TestCase):
    def test_region_and_query_backoff_are_deduplicated(self) -> None:
        model = {
            "rankings": {
                ("query_region", "купить кофе", 213): [
                    (10, 2, 200.0, 4),
                    (20, 1, 100.0, 3),
                ],
                ("query", "купить кофе", 0): [
                    (20, 3, 300.0, 5),
                    (30, 1, 50.0, 2),
                ],
            },
            "candidates": {
                10: {"title": "A"},
                20: {"title": "B"},
                30: {"title": "C"},
            },
        }
        result = rank(
            model=model,
            example={"query": "Купить кофе", "context": {"region_id": 213}},
            features={},
            top_k=10,
        )
        self.assertEqual([item["banner_id"] for item in result], [20, 10, 30])
        self.assertEqual(result[0]["title"], "B")
        self.assertEqual(
            set(result[0]["contributions"]["history"]),
            {"query", "query_region"},
        )

    def test_rankings_without_index_metadata_are_rejected(self) -> None:
        model = {
            "rankings": {
                ("query", "кофе", 0): [
                    (10, 3, 300.0, 5),
                    (999, 2, 200.0, 4),
                ]
            },
            "candidates": {10: {"title": "known"}},
        }
        result = rank(
            model=model,
            example={"query": "Кофе", "context": {"region_id": 213}},
            features={},
            top_k=10,
        )
        self.assertEqual([item["banner_id"] for item in result], [10])


if __name__ == "__main__":
    unittest.main()
