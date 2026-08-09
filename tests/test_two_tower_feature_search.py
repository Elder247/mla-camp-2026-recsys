from __future__ import annotations

import pytest

from scripts.continue_two_tower_feature_search import select_trial


def test_select_trial_uses_sc50_with_sc500_safety_floor() -> None:
    trials = [
        {
            "name": "unsafe",
            "metrics": {
                "sourcecost_recall_at_50": 0.60,
                "recall_at_50": 0.50,
                "sourcecost_recall_at_500": 0.64,
                "oracle_sourcecost_recall_at_50": 0.70,
                "new_only_sourcecost_share": 0.10,
            },
        },
        {
            "name": "winner",
            "metrics": {
                "sourcecost_recall_at_50": 0.58,
                "recall_at_50": 0.49,
                "sourcecost_recall_at_500": 0.68,
                "oracle_sourcecost_recall_at_50": 0.64,
                "new_only_sourcecost_share": 0.03,
            },
        },
        {
            "name": "runner_up",
            "metrics": {
                "sourcecost_recall_at_50": 0.57,
                "recall_at_50": 0.51,
                "sourcecost_recall_at_500": 0.69,
                "oracle_sourcecost_recall_at_50": 0.62,
                "new_only_sourcecost_share": 0.04,
            },
        },
    ]

    selected = select_trial(trials, minimum_sourcecost_recall_at_500=0.665)

    assert selected["name"] == "winner"


def test_select_trial_rejects_all_unsafe_trials() -> None:
    with pytest.raises(RuntimeError, match="safety floor"):
        select_trial(
            [
                {
                    "name": "unsafe",
                    "metrics": {
                        "sourcecost_recall_at_50": 0.60,
                        "recall_at_50": 0.50,
                        "sourcecost_recall_at_500": 0.64,
                        "oracle_sourcecost_recall_at_50": 0.70,
                        "new_only_sourcecost_share": 0.10,
                    },
                }
            ],
            minimum_sourcecost_recall_at_500=0.665,
        )


def test_select_trial_prefers_complementarity_before_single_tower_score() -> None:
    trials = [
        {
            "name": "strong_single",
            "metrics": {
                "sourcecost_recall_at_50": 0.59,
                "recall_at_50": 0.50,
                "sourcecost_recall_at_500": 0.68,
                "oracle_sourcecost_recall_at_50": 0.63,
                "new_only_sourcecost_share": 0.01,
            },
        },
        {
            "name": "complementary",
            "metrics": {
                "sourcecost_recall_at_50": 0.56,
                "recall_at_50": 0.49,
                "sourcecost_recall_at_500": 0.69,
                "oracle_sourcecost_recall_at_50": 0.66,
                "new_only_sourcecost_share": 0.04,
            },
        },
    ]

    selected = select_trial(trials, minimum_sourcecost_recall_at_500=0.665)

    assert selected["name"] == "complementary"
