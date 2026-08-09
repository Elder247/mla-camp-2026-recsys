from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from two_tower_v2.data import (  # noqa: E402
    enrich_rows,
    pack_bags,
    prefetch_batches,
    source_fields,
    wide_feature_bucket,
    yt_read_options,
)
from two_tower_v2.model import TwoTowerV2, embedding_dimension  # noqa: E402
from two_tower_v2.training import (  # noqa: E402
    positive_mask,
    retrieval_objective,
    sourcecost_example_weights,
)
from scripts.train_two_tower_v2 import load_config  # noqa: E402
from scripts.finetune_two_tower_validation import select_rows  # noqa: E402


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


def test_bpe_and_query_region_features_are_derived_in_batch() -> None:
    class Encoding:
        def __init__(self, ids):
            self.ids = ids

    class Tokenizer:
        def encode_batch(self, texts):
            return [Encoding([len(text), 17, 19]) for text in texts]

    cardinalities = {
        "query_word_ids": 32,
        "region_ids": 32,
        "query_bpe_ids": 16,
        "query_region_ids": 64,
    }
    rows = enrich_rows(
        [
            {
                "query_word_ids": [2, 3],
                "region_ids": [5],
                "query_text": "hello",
            }
        ],
        cardinalities=cardinalities,
        tokenizer=Tokenizer(),
        bpe_limits={"query_bpe_ids": 2},
    )
    assert rows[0]["query_bpe_ids"] == [5, 1]
    assert rows[0]["query_region_ids"] == [35, 60]
    assert source_fields(cardinalities)[-3:] == (
        "query_text",
        "title_text",
        "text_text",
    )


def test_source_cost_bucket_is_config_gated_and_bounded() -> None:
    cardinalities = {"banner_id_ids": 32, "source_cost_bucket_ids": 64}
    assert source_fields(cardinalities)[-1] == "source_cost"
    rows = enrich_rows(
        [
            {"banner_id_ids": [1], "source_cost": 0.0},
            {"banner_id_ids": [2], "source_cost": 1_000_000.0},
        ],
        cardinalities=cardinalities,
        tokenizer=None,
        source_cost_log1p_scale=8.0,
    )
    assert rows[0]["source_cost_bucket_ids"] == [0]
    assert rows[1]["source_cost_bucket_ids"] == [63]


def test_sourcecost_weights_are_bounded_mean_one_and_value_aware() -> None:
    weights = sourcecost_example_weights(
        [{"source_cost": 10.0}, {"source_cost": 1_000_000.0}],
        power=0.5,
        minimum=0.25,
        maximum=4.0,
        device=torch.device("cpu"),
    )
    assert torch.isclose(weights.mean(), torch.tensor(1.0))
    assert weights[1] > weights[0]


def test_multi_positive_loss_does_not_treat_duplicate_clicks_as_negatives() -> None:
    rows = [
        {"query_word_ids": [1], "region_ids": [4], "banner_id_ids": [8]},
        {"query_word_ids": [1], "region_ids": [4], "banner_id_ids": [9]},
        {"query_word_ids": [2], "region_ids": [4], "banner_id_ids": [8]},
    ]
    logits = torch.tensor(
        [[5.0, 4.0, 3.0], [4.0, 5.0, 0.0], [3.0, 0.0, 5.0]],
        requires_grad=True,
    )
    mask = positive_mask(rows, device=torch.device("cpu"))
    assert mask.tolist() == [
        [True, True, True],
        [True, True, False],
        [True, False, True],
    ]
    loss, returned = retrieval_objective(
        logits,
        rows,
        objective="multi_positive",
        symmetric_weight=1.0,
    )
    assert torch.equal(mask, returned)
    assert torch.isfinite(loss)
    loss.backward()


def test_v3_config_adds_bpe_capacity_without_changing_old_config() -> None:
    old = load_config(ROOT / "configs" / "two_tower" / "v2_dcn4_mlp3_full.yaml")
    new = load_config(
        ROOT / "configs" / "two_tower" / "v3_bpe_multipos_chrono_10m.yaml"
    )
    assert "query_bpe_ids" not in old.model.query_cardinalities
    assert new.model.query_cardinalities.query_bpe_ids == 16384
    assert new.model.query_cardinalities.query_region_ids == 65536
    assert new.model.banner_cardinalities.title_bpe_ids == 16384
    assert new.training.objective == "multi_positive"
    assert new.training.symmetric_weight == 1.0
    assert new.training.batch_size == 1024
    assert new.model.deep_residual


def test_v4_config_aligns_loss_and_banner_feature_with_sourcecost() -> None:
    cfg = load_config(
        ROOT / "configs" / "two_tower" / "v4_scweighted_chrono_10m.yaml"
    )
    assert cfg.model.banner_cardinalities.source_cost_bucket_ids == 256
    assert cfg.numeric_features.source_cost_log1p_scale == 8.0
    assert cfg.training.sourcecost_weight_power == 0.5
    assert cfg.training.sourcecost_weight_min == 0.25
    assert cfg.training.sourcecost_weight_max == 4.0


def test_v6_config_adds_existing_context_and_ad_metadata() -> None:
    cfg = load_config(
        ROOT / "configs" / "two_tower" / "v6_context_metadata_chrono_10m.yaml"
    )
    assert cfg.model.query_cardinalities.device_ids == 1024
    assert cfg.model.query_cardinalities.age_bucket_ids == 128
    assert cfg.model.query_cardinalities.gender_ids == 8
    assert cfg.model.banner_cardinalities.client_id_ids == 65536
    assert cfg.model.banner_cardinalities.banner_id_hash2_ids == 262144
    assert cfg.model.banner_cardinalities.caesar_sku_id_ids == 65536
    assert cfg.model.banner_cardinalities.product_price_bucket_ids == 256
    assert cfg.model.banner_cardinalities.url_domain_ids == 65536
    required = set(source_fields({
        **dict(cfg.model.query_cardinalities),
        **dict(cfg.model.banner_cardinalities),
    }))
    assert {
        "device_ids",
        "age_bucket_ids",
        "gender_ids",
        "client_id_ids",
        "order_id_ids",
        "caesar_model_id_ids",
        "caesar_sku_id_ids",
        "product_price",
        "banner_url",
        "banner_id",
    } <= required


def test_product_price_and_url_domain_features_are_bounded() -> None:
    rows = enrich_rows(
        [
            {
                "product_price": 1_000_000.0,
                "banner_url": "https://Example.COM:443/path?a=1",
            }
        ],
        cardinalities={
            "product_price_bucket_ids": 64,
            "url_domain_ids": 128,
        },
        tokenizer=None,
        product_price_log1p_scale=8.0,
    )
    assert rows[0]["product_price_bucket_ids"] == [63]
    assert 0 <= rows[0]["url_domain_ids"][0] < 128


def test_second_banner_hash_is_independent_and_config_gated() -> None:
    rows = enrich_rows(
        [{"banner_id": 123456}],
        cardinalities={"banner_id_hash2_ids": 262144},
        tokenizer=None,
    )
    expected = wide_feature_bucket("banner2:123456") % 262144
    assert rows[0]["banner_id_hash2_ids"] == [expected]


def test_v7_uses_more_in_batch_negatives() -> None:
    cfg = load_config(
        ROOT / "configs" / "two_tower" / "v7_large_batch_chrono_10m.yaml"
    )
    assert cfg.training.batch_size == 4096
    full = load_config(
        ROOT / "configs" / "two_tower" / "v7_large_batch_chrono_100m.yaml"
    )
    assert full.training.batch_size == 4096
    assert full.paths.train_table.endswith("train_clicks_100m_metadata_v1")
    assert full.paths.artifact_dir.endswith("v7_large_batch_chrono_100m_model")
    walk_forward = load_config(
        ROOT / "configs" / "two_tower" / "v7_large_batch_walk_forward_100m_s10.yaml"
    )
    assert walk_forward.training.prefetch_batches == 2


def test_prefetch_preserves_order_and_propagates_reader_errors() -> None:
    values = [[{"value": 1}], [{"value": 2}], [{"value": 3}]]
    assert list(prefetch_batches(iter(values), 2)) == values

    def failing():
        yield [{"value": 1}]
        raise RuntimeError("remote read failed")

    iterator = prefetch_batches(failing(), 1)
    assert next(iterator) == [{"value": 1}]
    with pytest.raises(RuntimeError, match="remote read failed"):
        next(iterator)


def test_validation_source_can_neutralize_new_optional_fields(monkeypatch) -> None:
    from two_tower_v2.data import YtTableSource

    class Client:
        def exists(self, path):
            return True

        def get(self, path):
            if path.endswith("/@row_count"):
                return 1
            return [{"name": "banner_id_ids"}]

    monkeypatch.setattr("common.yt_data.make_client", lambda: Client())
    source = YtTableSource(
        "//validation",
        "proxy",
        fields=("banner_id_ids", "banner_id", "banner_url"),
        allow_missing_fields=True,
    )
    assert source.read_fields == ("banner_id_ids",)


def test_source_rows_drop_null_elements_inside_categorical_lists() -> None:
    from two_tower_v2.data import _source_value

    assert _source_value({"device_ids": [None, 17]}, "device_ids") == [17]


def test_validation_finetune_split_is_strictly_temporal() -> None:
    rows = [
        {"show_time": 10, "hit_log_id": 3},
        {"show_time": 7, "hit_log_id": 2},
        {"show_time": 20, "hit_log_id": 1},
    ]
    temporal = select_rows(rows, scope="temporal_fit", boundary=15)
    assert [row["hit_log_id"] for row in temporal] == [2, 3]
    full = select_rows(rows, scope="full", boundary=15)
    assert [row["hit_log_id"] for row in full] == [2, 3, 1]
