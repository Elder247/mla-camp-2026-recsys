from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from continue_v3_walk_forward import retrieval_gate  # noqa: E402


def metrics(current50: float, baseline50: float, current500: float, baseline500: float, union500: float) -> dict:
    def row(value: float) -> dict:
        return {"sourcecost_recall": value}

    return {
        "current": {"50": row(current50), "500": row(current500)},
        "baseline": {"50": row(baseline50), "500": row(baseline500)},
        "oracle_union": {"500": row(union500)},
    }


class V3WalkForwardGateTest(unittest.TestCase):
    def test_accepts_quality_and_complementarity_gain(self) -> None:
        result = retrieval_gate(
            metrics(0.55, 0.54, 0.70, 0.705, 0.715),
            min_sc50_gain=0.002,
            max_sc500_loss=0.01,
            min_union_sc500_gain=0.002,
        )
        self.assertTrue(result["accepted"])

    def test_rejects_non_complementary_source(self) -> None:
        result = retrieval_gate(
            metrics(0.55, 0.54, 0.70, 0.705, 0.706),
            min_sc50_gain=0.002,
            max_sc500_loss=0.01,
            min_union_sc500_gain=0.002,
        )
        self.assertFalse(result["accepted"])
        self.assertFalse(result["checks"]["union_sc500_gain"])


if __name__ == "__main__":
    unittest.main()
