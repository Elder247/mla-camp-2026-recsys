from __future__ import annotations

from mla_recsys.config import compose_config
from scripts.continue_to_full import select_ranking


def test_promotion_gate_is_stricter_than_iteration_zero_metrics() -> None:
    cfg = compose_config(
        "i1_more_cg_features_sc",
        run_id="20260808_1400_promotion",
        mode="full",
        scope="full",
    )
    assert float(cfg.promotion_gate.candidate_sourcecost_recall_at_500) > 0.70
    assert float(cfg.promotion_gate.ranker_sourcecost_recall_at_50) > 0.62


def test_promotion_selects_best_temporal_ranking_and_prefers_rrf_on_tie() -> None:
    metrics = {
        "rrf": {"50": {"sourcecost_recall": 0.63}},
        "catboost": {"50": {"sourcecost_recall": 0.62}},
    }
    assert select_ranking(metrics, ["rrf", "catboost"]) == ("rrf", 0.63)
    metrics["catboost"]["50"]["sourcecost_recall"] = 0.63
    assert select_ranking(metrics, ["rrf", "catboost"])[0] == "rrf"
