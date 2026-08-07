from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mla_recsys.config import compose_config, config_fingerprint, parse_cli_dotlist  # noqa: E402


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
        runtime, overrides = parse_cli_dotlist(
            ["experiment=i0_reproduce", "run_id=20260807_2200_x", "ranker.depth=7"]
        )
        self.assertEqual(runtime["experiment"], "i0_reproduce")
        self.assertEqual(overrides, ["ranker.depth=7"])


if __name__ == "__main__":
    unittest.main()

