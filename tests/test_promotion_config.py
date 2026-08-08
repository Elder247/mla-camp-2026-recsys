from __future__ import annotations

from mla_recsys.config import compose_config


def test_promotion_gate_is_stricter_than_iteration_zero_metrics() -> None:
    cfg = compose_config(
        "i1_more_cg_features_sc",
        run_id="20260808_1400_promotion",
        mode="full",
        scope="full",
    )
    assert float(cfg.promotion_gate.candidate_sourcecost_recall_at_500) > 0.70
    assert float(cfg.promotion_gate.ranker_sourcecost_recall_at_50) > 0.62
