from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mla_recsys.artifacts import RunStore  # noqa: E402
from mla_recsys.config import compose_config  # noqa: E402
from mla_recsys.stage_runner import StageRunner  # noqa: E402


class StageRunnerTest(unittest.TestCase):
    def test_subprocess_log_timing_and_status(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = compose_config(
                "i0_reproduce",
                run_id="20260807_2200_stage",
                mode="smoke",
                overrides=[
                    f"paths.root={root}",
                    f"paths.runs={root / 'runs'}",
                    f"paths.cache={root / 'cache'}",
                    f"paths.python={sys.executable}",
                ],
            )
            store = RunStore.initialize(cfg, repo_root=ROOT)
            value = StageRunner(store, echo=False).run(
                "unit_stage",
                [sys.executable, "-c", "print('stage-ok')"],
                cwd=ROOT,
            )
            self.assertEqual(value["status"], "completed")
            self.assertIn("stage-ok", (store.path / "logs" / "unit_stage.log").read_text())
            saved = json.loads((store.path / "stages" / "unit_stage.json").read_text())
            self.assertEqual(saved["return_code"], 0)
            self.assertGreater(saved["peak_rss_bytes"], 0)
            self.assertIn(
                saved["peak_rss_measurement"],
                {"linux_proc_stage", "gnu_time_stage", "children_upper_bound"},
            )
            self.assertTrue((store.path / "reports" / "timing.csv").is_file())


if __name__ == "__main__":
    unittest.main()
