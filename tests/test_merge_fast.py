from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mla_recsys.candidate_cache import SOURCE_SCHEMA, source_part_path  # noqa: E402
from mla_recsys.config import compose_config  # noqa: E402
from mla_recsys.fusion import fuse_rankings  # noqa: E402
from mla_recsys.merge import merge_partition  # noqa: E402


def source_row(
    banner_id: int,
    rank: int,
    score: float,
    *,
    clicks: int = 0,
    source_cost: float = 0.0,
    query: bool = False,
    region: bool = False,
) -> dict:
    return {
        "request_id": "r1",
        "hit_log_id": 11,
        "banner_id": banner_id,
        "source_rank": rank,
        "source_score": score,
        "history_click_count": clicks,
        "history_source_cost_sum": source_cost,
        "history_query_present": query,
        "history_region_present": region,
    }


class FastMergeTest(unittest.TestCase):
    def test_matches_reference_rrf_and_provenance(self) -> None:
        cfg = compose_config(
            "i0_reproduce",
            mode="offline",
            overrides=[
                "data.partition_count=1",
                "candidates.union_max_candidates=3",
                "candidates.ranker_pool=3",
            ],
        )
        rows = {
            "tfidf_v1": [source_row(1, 1, 3.0), source_row(2, 2, 2.0)],
            "two_tower_fps_v1": [
                source_row(2, 1, 0.9),
                source_row(3, 2, 0.8),
            ],
            "history_legacy_v1": [
                source_row(
                    3,
                    1,
                    4.0,
                    clicks=4,
                    source_cost=500.0,
                    query=True,
                    region=True,
                )
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            run_path = Path(directory)
            for source, values in rows.items():
                path = source_part_path(run_path, "train", source, 0)
                path.parent.mkdir(parents=True, exist_ok=True)
                pq.write_table(pa.Table.from_pylist(values, schema=SOURCE_SCHEMA), path)
            table = merge_partition(
                cfg=cfg,
                run_path=run_path,
                split="train",
                partition=0,
                requests=[{"request_id": "r1", "hit_log_id": 11}],
            )

        rankings = {
            source: [
                {
                    "banner_id": row["banner_id"],
                    "score": row["source_score"],
                    "_source_rank": row["source_rank"],
                }
                for row in values
            ]
            for source, values in rows.items()
        }
        reference = fuse_rankings(
            rankings,
            weights={
                source: float(cfg.candidates.generators[source].weight)
                for source in rankings
            },
            quotas={
                source: int(cfg.candidates.generators[source].quota)
                for source in rankings
            },
            rrf_constant=float(cfg.candidates.rrf_constant),
            max_candidates=3,
        )
        actual = table.to_pylist()
        self.assertEqual(
            [row["banner_id"] for row in actual],
            [row["banner_id"] for row in reference],
        )
        for row, expected in zip(actual, reference):
            self.assertAlmostEqual(row["rrf_score"], expected["rrf_score"])
            self.assertEqual(row["source_count"], expected["source_count"])
        banner_three = next(row for row in actual if row["banner_id"] == 3)
        self.assertEqual(banner_three["history_click_count"], 4)
        self.assertEqual(banner_three["history_source_cost_sum"], 500.0)
        self.assertTrue(banner_three["history_query_present"])
        self.assertTrue(banner_three["history_region_present"])


if __name__ == "__main__":
    unittest.main()
