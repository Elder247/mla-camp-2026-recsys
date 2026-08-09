from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from two_tower_v2.data import pack_bags, yt_read_options  # noqa: E402
from two_tower_v2.model import TwoTowerV2, embedding_dimension  # noqa: E402
from scripts.train_two_tower_v2 import load_config  # noqa: E402


def test_embedding_dimension_uses_formula_rounding_and_caps() -> None:
    kwargs = {"multiplier": 6.0, "min_dim": 8, "max_dim": 96, "round_to": 8}
    assert embedding_dimension(1, **kwargs) == 8
    assert embedding_dimension(16, **kwargs) == 16
    assert embedding_dimension(65_536, **kwargs) == 96
    assert embedding_dimension(10**9, **kwargs) == 96


def test_four_cross_three_deep_towers_produce_normalized_vectors() -> None:
    cardinalities = {"query_word_ids": 32, "region_ids": 16}
    banner_cardinalities = {
        "banner_id_ids": 32,
        "ad_group_id_ids": 16,
        "title_word_ids": 32,
        "text_word_ids": 32,
    }
    model = TwoTowerV2(
        query_cardinalities=cardinalities,
        banner_cardinalities=banner_cardinalities,
        embedding_policy={"multiplier": 6.0, "min_dim": 8, "max_dim": 16, "round_to": 8},
        hidden_dim=16,
        output_dim=8,
        cross_layers=4,
        deep_layers=3,
        dropout=0.0,
    ).eval()
    rows = [
        {
            "query_word_ids": [1, 2],
            "region_ids": [3],
            "banner_id_ids": [4],
            "ad_group_id_ids": [5],
            "title_word_ids": [6, 7],
            "text_word_ids": [],
        },
        {
            "query_word_ids": [8],
            "region_ids": [],
            "banner_id_ids": [9],
            "ad_group_id_ids": [10],
            "title_word_ids": [11],
            "text_word_ids": [12, 13],
        },
    ]
    bags = pack_bags(
        rows,
        cardinalities={**cardinalities, **banner_cardinalities},
        device=torch.device("cpu"),
    )
    with torch.inference_mode():
        query = model.encode_query(bags)
        banner = model.encode_banner(bags)
    assert query.shape == (2, 8)
    assert banner.shape == (2, 8)
    assert torch.allclose(query.norm(dim=1), torch.ones(2), atol=1e-5)
    assert torch.allclose(banner.norm(dim=1), torch.ones(2), atol=1e-5)
    assert len(model.query_tower.cross) == 4
    assert len(model.query_tower.deep) == 3


def test_full_training_config_keeps_all_rows_and_requested_architecture() -> None:
    cfg = load_config(ROOT / "configs" / "two_tower" / "v2_dcn4_mlp3_full.yaml")
    assert cfg.training.max_examples == 0
    assert cfg.training.max_steps == 0
    assert cfg.model.cross_layers == 4
    assert cfg.model.deep_layers == 3
    assert cfg.model.embedding_policy.multiplier == 6.0
    assert cfg.model.embedding_policy.max_dim == 96
    assert cfg.export.max_index_rows == 0
    assert not cfg.training.strict_chronological


def test_chronological_config_preserves_sorted_yt_stream() -> None:
    cfg = load_config(
        ROOT / "configs" / "two_tower" / "v2_dcn4_mlp3_chrono_10m.yaml"
    )
    assert cfg.training.strict_chronological
    assert cfg.training.shuffle_buffer == 1
    assert cfg.paths.train_table.endswith("train_clicks_10m_v1")
    assert yt_read_options(ordered=True) == {
        "unordered": False,
        "enable_read_parallel": False,
    }
    assert yt_read_options(ordered=False) == {
        "unordered": True,
        "enable_read_parallel": True,
    }


def test_10m_order_probe_changes_only_the_stream_policy() -> None:
    chronological = load_config(
        ROOT / "configs" / "two_tower" / "v2_dcn4_mlp3_chrono_10m.yaml"
    )
    shuffled = load_config(
        ROOT / "configs" / "two_tower" / "v2_dcn4_mlp3_shuffled_10m.yaml"
    )

    assert chronological.paths.train_table == shuffled.paths.train_table
    assert chronological.training.seed == shuffled.training.seed
    assert chronological.model == shuffled.model
    assert chronological.training.strict_chronological
    assert chronological.training.shuffle_buffer == 1
    assert not shuffled.training.strict_chronological
    assert shuffled.training.shuffle_buffer == 20000
