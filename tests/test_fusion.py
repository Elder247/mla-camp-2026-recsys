from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mla_recsys.fusion import fuse_rankings  # noqa: E402


class FusionTest(unittest.TestCase):
    def test_deduplicates_and_preserves_provenance(self) -> None:
        rankings = {
            "lexical": [
                {"banner_id": 1, "score": 10.0, "title": "one"},
                {"banner_id": 2, "score": 9.0, "title": "two"},
            ],
            "semantic": [
                {"banner_id": 2, "score": 0.9, "title": "two"},
                {"banner_id": 3, "score": 0.8, "title": "three"},
            ],
        }
        result = fuse_rankings(rankings, rrf_constant=10.0, max_candidates=10)
        self.assertEqual([item["banner_id"] for item in result], [2, 1, 3])
        shared = result[0]
        self.assertEqual(shared["sources"], ["lexical", "semantic"])
        self.assertEqual(shared["source_count"], 2)
        self.assertEqual(shared["retrieval"]["lexical"]["rank"], 2)
        self.assertEqual(shared["retrieval"]["semantic"]["rank"], 1)

    def test_quota_is_applied_after_source_deduplication(self) -> None:
        rankings = {
            "source": [
                {"banner_id": 1, "score": 3.0},
                {"banner_id": 1, "score": 2.0},
                {"banner_id": 2, "score": 1.0},
            ]
        }
        result = fuse_rankings(rankings, quotas={"source": 2}, max_candidates=10)
        self.assertEqual([item["banner_id"] for item in result], [1, 2])

    def test_rejects_invalid_settings(self) -> None:
        with self.assertRaises(ValueError):
            fuse_rankings({}, rrf_constant=0)
        with self.assertRaises(ValueError):
            fuse_rankings({"x": []}, max_candidates=0)

    def test_cached_source_rank_is_preserved(self) -> None:
        result = fuse_rankings(
            {"source": [{"banner_id": 7, "score": 1.0, "_source_rank": 3}]},
            rrf_constant=10.0,
            max_candidates=10,
        )
        self.assertEqual(result[0]["retrieval"]["source"]["rank"], 3)
        self.assertAlmostEqual(result[0]["rrf_score"], 1.0 / 13.0)


if __name__ == "__main__":
    unittest.main()
