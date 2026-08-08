from __future__ import annotations

from omegaconf import OmegaConf

from scripts.train_ranker import label_spec


def test_raw_sourcecost_label_is_not_log_surrogate() -> None:
    raw = OmegaConf.create(
        {"ranker": {"kind": "ranker_raw_sc_label", "raw_sc_scale": 1_000_000.0}}
    )
    log = OmegaConf.create(
        {"ranker": {"kind": "ranker_logsc", "raw_sc_scale": 1_000_000.0}}
    )
    assert label_spec(raw) == ("label_raw_sc", 1_000_000.0)
    assert label_spec(log) == ("label_logsc", 1.0)
