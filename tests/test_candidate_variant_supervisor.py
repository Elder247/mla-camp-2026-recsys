from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from scripts.continue_candidate_variant import full_command, select_variant


def metrics(value: float) -> dict:
    return {"50": {"sourcecost_recall": value, "recall": value - 0.1}}


def test_select_variant_inverts_qrmse_weight_for_external_yeti() -> None:
    single = {
        "best": {
            "catboost_weight": 0.6,
            "exponent": 0.1,
            "rerank_top_n": 75,
            "metrics": metrics(0.65),
        }
    }
    ensemble = {
        "geometry": {
            "base": {"model_a_weight": 0.65, "catboost_weight": 0.55},
            "best": {
                "exponent": 0.15,
                "rerank_top_n": 100,
                "metrics": metrics(0.66),
            },
        }
    }
    selected = select_variant(single, ensemble)
    assert selected["ranking"] == "model_ensemble"
    assert selected["qrmse_weight"] == 0.65
    assert selected["external_yeti_weight"] == 0.35


def test_full_command_carries_candidate_and_ranking_contract() -> None:
    args = Namespace(
        python=Path("/venv/python"),
        experiment="experiment",
        full_run="20260809_0715_full",
        runs=Path("/shared/runs"),
        cache=Path("/shared/cache"),
        immutable_artifacts=Path("/shared/artifacts"),
        artifact_override=Path("/shared/artifacts/variant"),
        full_reuse_run=Path("/shared/runs/donor"),
        full_external_yeti_model=Path("/shared/runs/yeti/models/catboost.cbm"),
    )
    choice = {
        "ranking": "model_ensemble",
        "external_yeti_weight": 0.5,
        "catboost_weight": 0.55,
        "exponent": 0.2,
        "rerank_top_n": 75,
    }
    command = full_command(args, iterations=185, choice=choice)
    assert "paths.two_tower_v2_walk_forward_artifact=/shared/artifacts/variant" in command
    assert "candidates.reuse_run=/shared/runs/donor" in command
    assert "submission.ranking=model_ensemble" in command
    assert "submission.model_ensemble.model_a_weight=0.5" in command
    assert "ranker.iterations=185" in command
