from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mla_recsys.config import compose_config, config_fingerprint, parse_cli_dotlist  # noqa: E402
from mla_recsys.candidate_cache import feature_name  # noqa: E402
from mla_recsys.feature_cache import configured_feature_names  # noqa: E402
from mla_recsys.merge import merged_schema  # noqa: E402


class ConfigTest(unittest.TestCase):
    def test_inheritance_paths_and_override_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = compose_config(
                "i0_reproduce",
                run_id="20260807_2200_test",
                mode="smoke",
                overrides=[
                    f"paths.root={root}",
                    f"paths.runs={root / 'runs'}",
                    f"paths.cache={root / 'cache'}",
                    f"paths.python={sys.executable}",
                ],
            )
            self.assertEqual(cfg.experiment.name, "i0_reproduce")
            self.assertEqual(cfg.candidates.generators.tfidf_v1.quota, 1000)
            self.assertEqual(Path(str(cfg.paths.runs)), root / "runs")
            changed = compose_config(
                "i0_reproduce",
                run_id="20260807_2201_test",
                mode="smoke",
                overrides=[
                    f"paths.root={root}",
                    f"paths.runs={root / 'runs'}",
                    f"paths.cache={root / 'cache'}",
                    f"paths.python={sys.executable}",
                    "candidates.ranker_pool=1000",
                ],
            )
            self.assertNotEqual(config_fingerprint(cfg), config_fingerprint(changed))

    def test_temporal_boundary_and_cli_parser(self) -> None:
        cfg = compose_config("i0_reproduce", mode="offline")
        self.assertEqual(cfg.split.fit.end_exclusive, cfg.split.holdout.start_inclusive)
        self.assertEqual(cfg.ranker.importance.type, "PredictionValuesChange")
        self.assertNotIn("shap_sample_rows", cfg.ranker.importance)
        self.assertNotIn("permutation_sample_rows", cfg.ranker.importance)
        runtime, overrides = parse_cli_dotlist(
            ["experiment=i0_reproduce", "run_id=20260807_2200_x", "ranker.depth=7"]
        )
        self.assertEqual(runtime["experiment"], "i0_reproduce")
        self.assertEqual(overrides, ["ranker.depth=7"])

    def test_every_configured_stage_script_exists(self) -> None:
        cfg = compose_config("i0_reproduce", mode="offline")
        missing = [
            str(stage.script)
            for stage in cfg.pipeline.stages
            if not (ROOT / "scripts" / str(stage.script)).is_file()
        ]
        self.assertEqual(missing, [])

    def test_cache_parity_runs_in_every_pipeline_mode(self) -> None:
        cfg = compose_config("i0_reproduce", mode="offline")
        parity = next(
            stage for stage in cfg.pipeline.stages if stage.name == "validate_cache_parity"
        )
        self.assertEqual(list(parity.modes), ["smoke", "offline", "full"])

    def test_every_i1_generator_has_an_implementation(self) -> None:
        cfg = compose_config("i1_more_cg_features_sc", mode="offline")
        for name, item in cfg.candidates.generators.items():
            if not bool(item.get("enabled", False)):
                continue
            if str(item.get("kind") or "") == "temporal_history":
                self.assertIn(
                    name,
                    {"history_user_v1", "region_pop_sc_v1", "global_pop_sc_v1"},
                )
            else:
                self.assertTrue(item.get("code_path_key"), name)
                self.assertTrue(item.get("artifact_path_key"), name)

    def test_every_i1_generator_is_present_in_merged_schema(self) -> None:
        cfg = compose_config("i1_more_cg_features_sc", mode="offline")
        names = set(merged_schema(cfg).names)
        for source, item in cfg.candidates.generators.items():
            if not bool(item.get("enabled", False)):
                continue
            alias = feature_name(cfg, str(source))
            self.assertTrue(
                {f"{alias}__present", f"{alias}__rank", f"{alias}__score"} <= names,
                source,
            )

    def test_fast_value_contract_is_compact_and_budgeted(self) -> None:
        cfg = compose_config("i1_fast_value", mode="offline")
        enabled = [
            str(name)
            for name, item in cfg.candidates.generators.items()
            if bool(item.enabled)
        ]
        self.assertEqual(
            enabled,
            ["tfidf_v1", "two_tower_fps_v1", "history_legacy_v1"],
        )
        self.assertEqual(cfg.candidates.ranker_pool, 500)
        self.assertEqual(cfg.candidates.union_max_candidates, 1000)
        self.assertEqual(cfg.pipeline.max_wall_seconds, 10800)
        self.assertEqual(cfg.ranker.kind, "ranker_logsc")
        self.assertEqual(len(configured_feature_names(cfg)), 151)
        self.assertEqual(
            list(cfg.features.counter_families),
            ["region", "domain", "group", "query"],
        )

    def test_fast_quality_restores_depth_and_only_valuable_history_sources(self) -> None:
        cfg = compose_config("i1_fast_quality", mode="offline")
        enabled = [
            str(name)
            for name, item in cfg.candidates.generators.items()
            if bool(item.enabled)
        ]
        self.assertEqual(
            enabled,
            [
                "tfidf_v1",
                "two_tower_fps_v1",
                "history_legacy_v1",
                "history_query_click_v1",
                "history_query_sc_v1",
                "history_query_region_v1",
            ],
        )
        self.assertEqual(cfg.candidates.generators.tfidf_v1.top_k, 1000)
        self.assertEqual(cfg.candidates.generators.two_tower_fps_v1.top_k, 1000)
        self.assertEqual(cfg.candidates.ranker_pool, 500)
        self.assertEqual(cfg.candidates.union_max_candidates, 2200)
        self.assertEqual(cfg.candidates.rrf_constant, 40.0)
        self.assertEqual(cfg.ranker.kind, "ranker_raw_sc_label")
        self.assertEqual(cfg.ranker.iterations, 900)
        self.assertEqual(cfg.pipeline.max_wall_seconds, 10800)
        self.assertEqual(
            list(cfg.features.counter_families),
            ["region", "domain", "group", "query"],
        )

    def test_two_tower_v2_probe_has_only_one_batched_source(self) -> None:
        cfg = compose_config("i2_two_tower_v2_probe", mode="offline")
        enabled = [
            str(name)
            for name, item in cfg.candidates.generators.items()
            if bool(item.enabled)
        ]
        self.assertEqual(enabled, ["two_tower_v2"])
        source = cfg.candidates.generators.two_tower_v2
        self.assertEqual(source.batch_size, 256)
        self.assertEqual(source.top_k, 1000)
        self.assertEqual(cfg.pipeline.max_wall_seconds, 3600)


if __name__ == "__main__":
    unittest.main()
