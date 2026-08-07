from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mla_recsys.features import extract_feature_rows, feature_names  # noqa: E402


class FeatureTest(unittest.TestCase):
    def test_feature_order_and_history_values(self) -> None:
        generators = ["tfidf", "history"]
        candidates = [
            {
                "banner_id": 1,
                "title": "Купить кофе в Москве",
                "text": "Доставка",
                "source_cost": 100.0,
                "rrf_score": 0.1,
                "source_count": 2,
                "retrieval": {
                    "tfidf": {"rank": 1, "reciprocal_rank": 1.0, "score": 3.0},
                    "history": {
                        "rank": 2,
                        "reciprocal_rank": 0.5,
                        "score": 0.2,
                        "contributions": {
                            "history": {"query_region": {"rank": 1}},
                            "click_count": 4,
                            "source_cost_sum": 500.0,
                        },
                    },
                },
            }
        ]
        names = feature_names(generators)
        row = extract_feature_rows(
            {"query": "купить кофе", "context": {"region_id": 213}},
            candidates,
            generators,
        )[0]
        values = dict(zip(names, row))
        self.assertEqual(len(row), len(names))
        self.assertEqual(values["title_overlap"], 2.0)
        self.assertEqual(values["history_region_present"], 1.0)
        self.assertEqual(values["history__present"], 1.0)


if __name__ == "__main__":
    unittest.main()
