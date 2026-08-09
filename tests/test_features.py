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

    def test_v2_feature_order_raw_value_and_counters(self) -> None:
        enabled = [
            "retrieval_provenance_v2",
            "candidate_static_v2",
            "weekly_counters_v1",
            "cross_features_v1",
        ]
        names = feature_names(
            ["tfidf"],
            version="feature_v2",
            enabled_groups=enabled,
            counter_families=["banner"],
            counter_windows_days=[0],
        )
        candidate = {
            "banner_id": 1,
            "title": "кофе",
            "text": "",
            "url": "example.test",
            "source_cost": 1234567.0,
            "rrf_score": 0.1,
            "source_count": 1,
            "retrieval": {
                "tfidf": {"rank": 1, "reciprocal_rank": 1.0, "score": 3.0}
            },
            "counter_features": {
                "counter__banner__all__clicks_log1p": 2.0,
                "counter__banner__all__sc_sum_log1p": 3.0,
                "counter__banner__all__sc_avg": 4.0,
                "counter__banner__all__age_days": 5.0,
                "counter__banner__all__present": 1.0,
            },
        }
        row = extract_feature_rows(
            {"query": "кофе", "context": {}},
            [candidate],
            ["tfidf"],
            version="feature_v2",
            enabled_groups=enabled,
            counter_families=["banner"],
            counter_windows_days=[0],
        )[0]
        values = dict(zip(names, row))
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(values["source_cost_raw"], 1234567.0)
        self.assertEqual(values["counter__banner__all__present"], 1.0)
        self.assertEqual(values["source_cost_x_banner_sc_avg"], 4938268.0)

    def test_temporal_history_root_aggregates_feed_legacy_history_features(self) -> None:
        generators = ["history_query_sc_oof"]
        candidate = {
            "banner_id": 1,
            "title": "кофе",
            "text": "",
            "source_cost": 100.0,
            "rrf_score": 0.1,
            "source_count": 1,
            "history_click_count": 4,
            "history_source_cost_sum": 500.0,
            "history_query_present": True,
            "history_region_present": False,
            "retrieval": {
                "history_query_sc_oof": {
                    "rank": 1,
                    "reciprocal_rank": 1.0,
                    "score": 500.0,
                }
            },
        }
        names = feature_names(generators)
        row = extract_feature_rows(
            {"query": "кофе", "context": {}},
            [candidate],
            generators,
        )[0]
        values = dict(zip(names, row))
        self.assertAlmostEqual(values["history_click_count_log1p"], 1.6094379)
        self.assertAlmostEqual(values["history_source_cost_log1p"], 6.2166061)
        self.assertEqual(values["history_query_present"], 1.0)
        self.assertEqual(values["history_region_present"], 0.0)


if __name__ == "__main__":
    unittest.main()
