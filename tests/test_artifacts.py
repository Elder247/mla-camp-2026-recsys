from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mla_recsys.artifacts import (  # noqa: E402
    RunStore,
    make_cache_key,
    write_output_manifest,
    validate_output_cache,
)
from mla_recsys.config import compose_config, config_fingerprint  # noqa: E402


class ArtifactTest(unittest.TestCase):
    def test_cache_invalidation_and_schema_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "part.parquet"
            output.write_bytes(b"stable-output")
            inputs = [{"path": "input", "sha256": "abc"}]
            key = make_cache_key(
                stage="candidates",
                artifact_version="v1",
                config_sha256="cfg-a",
                inputs=inputs,
            )
            write_output_manifest(
                output,
                stage="candidates",
                artifact_version="v1",
                config_sha256="cfg-a",
                inputs=inputs,
                rows=2,
                schema={"banner_id": "uint64"},
            )
            self.assertEqual(
                validate_output_cache(
                    output,
                    expected_cache_key=key,
                    expected_rows=2,
                    expected_schema={"banner_id": "uint64"},
                ),
                (True, "valid"),
            )
            other_key = make_cache_key(
                stage="candidates",
                artifact_version="v1",
                config_sha256="cfg-b",
                inputs=inputs,
            )
            self.assertEqual(
                validate_output_cache(output, expected_cache_key=other_key)[1],
                "cache_key_mismatch",
            )

    def test_run_cannot_resume_with_different_config(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            common = [
                f"paths.root={root}",
                f"paths.runs={root / 'runs'}",
                f"paths.cache={root / 'cache'}",
                f"paths.python={sys.executable}",
            ]
            cfg = compose_config(
                "i0_reproduce",
                run_id="20260807_2200_contract",
                mode="smoke",
                overrides=common,
            )
            store = RunStore.initialize(cfg, repo_root=ROOT)
            self.assertTrue((store.path / "config.yaml").is_file())
            changed = compose_config(
                "i0_reproduce",
                run_id="20260807_2200_contract",
                mode="smoke",
                overrides=[*common, "ranker.depth=7"],
            )
            self.assertNotEqual(config_fingerprint(cfg), config_fingerprint(changed))
            with self.assertRaises(RuntimeError):
                RunStore.initialize(changed, repo_root=ROOT)

    def test_resume_records_changed_git_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = compose_config(
                "i0_reproduce",
                run_id="20260807_2201_resume",
                mode="smoke",
                overrides=[
                    f"paths.root={root}",
                    f"paths.runs={root / 'runs'}",
                    f"paths.cache={root / 'cache'}",
                    f"paths.python={sys.executable}",
                ],
            )
            store = RunStore.initialize(cfg, repo_root=ROOT)
            manifest_path = store.path / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["git"] = {"sha": "stale", "branch": "main", "dirty": False}
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            RunStore.initialize(cfg, repo_root=ROOT)
            resumed = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(len(resumed["resume_events"]), 1)
            self.assertNotEqual(resumed["resume_events"][0]["git"]["sha"], "stale")

    def test_successful_finalize_removes_stale_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cfg = compose_config(
                "i0_reproduce",
                run_id="20260807_2202_finalize",
                mode="smoke",
                overrides=[
                    f"paths.root={root}",
                    f"paths.runs={root / 'runs'}",
                    f"paths.cache={root / 'cache'}",
                    f"paths.python={sys.executable}",
                ],
            )
            store = RunStore.initialize(cfg, repo_root=ROOT)
            store.finalize("failed", error="PreviousFailure")
            result = store.finalize("completed")
            self.assertEqual(result["status"], "completed")
            self.assertNotIn("error", result)


if __name__ == "__main__":
    unittest.main()
