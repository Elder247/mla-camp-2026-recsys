from __future__ import annotations

from scripts.continue_v7_top50_ensemble import selected_parameters


def test_selected_parameters_keeps_honest_split_metrics() -> None:
    report = {
        "best": {
            "weights": [0.2, 0.3, 0.5],
            "rrf_constant": 10,
            "exponent": 0.2,
            "rerank_top_n": 75,
            "tune_metrics": {"50": {"sourcecost_recall": 0.68}},
            "validation_metrics": {"50": {"sourcecost_recall": 0.66}},
            "full_metrics": {"50": {"sourcecost_recall": 0.67}},
        }
    }

    selected = selected_parameters(report)

    assert selected["weights"] == (0.2, 0.3, 0.5)
    assert selected["validation_sc50"] == 0.66
