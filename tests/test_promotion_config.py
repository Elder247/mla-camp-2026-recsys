from __future__ import annotations

from mla_recsys.config import compose_config
import json

from scripts.continue_to_full import load_blend_probe, select_ranking


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


def test_promotion_loads_only_the_configured_blend(tmp_path) -> None:
    cfg = compose_config(
        "i2_walk_forward_10m_fast_quality",
        run_id="20260808_2030_blend",
        mode="full",
        scope="full",
    )
    metrics = tmp_path / "metrics"
    metrics.mkdir()
    expected = {"50": {"sourcecost_recall": 0.6175}}
    (metrics / "rank_blend_fine.json").write_text(
        json.dumps(
            {
                "results": [
                    {"method": "rank_linear", "alpha": 0.7, "metrics": {}},
                    {"method": "rank_linear", "alpha": 0.75, "metrics": expected},
                ]
            }
        ),
        encoding="utf-8",
    )

    loaded = load_blend_probe(tmp_path, cfg)

    assert loaded is not None
    assert loaded[0] == expected
    assert loaded[1].endswith("rank_blend_fine.json")
