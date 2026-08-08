from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mla_recsys.submission import validate_submission  # noqa: E402
from scripts.make_submission import rrf_predictions  # noqa: E402


class SubmissionTest(unittest.TestCase):
    def test_rrf_predictions_follow_pre_rank(self) -> None:
        schema = pa.schema(
            [
                pa.field("request_id", pa.string(), nullable=False),
                pa.field("hit_log_id", pa.uint64(), nullable=False),
                pa.field("banner_id", pa.uint64(), nullable=False),
                pa.field("pre_rank", pa.int32(), nullable=False),
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            run_path = Path(directory)
            merged = run_path / "candidates" / "test" / "merged"
            merged.mkdir(parents=True)
            pq.write_table(
                pa.Table.from_pylist(
                    [
                        {"request_id": "a", "hit_log_id": 11, "banner_id": 30, "pre_rank": 2},
                        {"request_id": "a", "hit_log_id": 11, "banner_id": 20, "pre_rank": 1},
                        {"request_id": "b", "hit_log_id": 12, "banner_id": 40, "pre_rank": 1},
                    ],
                    schema=schema,
                ),
                merged / "part-00000.parquet",
            )
            predictions, inputs = rrf_predictions(run_path)
            self.assertEqual(predictions["a"], (11, [20, 30]))
            self.assertEqual(predictions["b"], (12, [40]))
            self.assertEqual(len(inputs), 1)

    def test_strict_contract(self) -> None:
        schema = pa.schema(
            [
                pa.field("HitLogID", pa.uint64(), nullable=False),
                pa.field("BannerID", pa.list_(pa.uint64()), nullable=False),
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "submission.parquet"
            pq.write_table(
                pa.Table.from_pylist(
                    [
                        {"HitLogID": 1, "BannerID": [10, 20]},
                        {"HitLogID": 2, "BannerID": [20, 30]},
                    ],
                    schema=schema,
                ),
                path,
            )
            report = validate_submission(
                path,
                expected_hitlog_ids={1, 2},
                valid_banner_ids={10, 20, 30},
                top_k=2,
                allow_short=False,
            )
            self.assertTrue(report["ok"], report)
            bad = validate_submission(
                path,
                expected_hitlog_ids={1, 2},
                valid_banner_ids={10, 20},
                top_k=2,
                allow_short=False,
            )
            self.assertFalse(bad["ok"])


if __name__ == "__main__":
    unittest.main()
