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
            },
        },
        {
            "name": "winner",
            "metrics": {
                "sourcecost_recall_at_50": 0.58,
                "recall_at_50": 0.49,
                "sourcecost_recall_at_500": 0.68,
            },
        },
        {
            "name": "runner_up",
            "metrics": {
                "sourcecost_recall_at_50": 0.57,
                "recall_at_50": 0.51,
                "sourcecost_recall_at_500": 0.69,
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
                    },
                }
            ],
            minimum_sourcecost_recall_at_500=0.665,
        )
