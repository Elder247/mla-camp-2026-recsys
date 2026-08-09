from __future__ import annotations

from scripts.continue_logq_fullfit import gate, robust_best


def row(early: float, late: float, full: float, weight: float) -> dict:
    return {
        "weights": [weight, 1.0 - weight],
        "tune_metrics": {"50": {"sourcecost_recall": early}},
        "validation_metrics": {"50": {"sourcecost_recall": late}},
        "full_metrics": {"50": {"sourcecost_recall": full}},
    }


def test_robust_best_prefers_the_stronger_weak_half() -> None:
    report = {
        "refined_results": [
            row(0.72, 0.60, 0.66, 0.2),
            row(0.68, 0.65, 0.665, 0.4),
        ]
    }

    assert robust_best(report)["weights"] == [0.4, 0.6]


def test_gate_requires_gain_on_early_late_and_full() -> None:
    baseline = {"refined_results": [row(0.60, 0.61, 0.605, 1.0)]}
    accepted = {"refined_results": [row(0.602, 0.613, 0.608, 0.5)]}
    late_regression = {"refined_results": [row(0.62, 0.609, 0.615, 0.5)]}

    assert gate(accepted, baseline, 0.001)["accepted"] is True
    rejected = gate(late_regression, baseline, 0.001)
    assert rejected["accepted"] is False
    assert rejected["gains"]["late"] < 0.0
