#!/usr/bin/env python3
"""Create an immutable walk-forward manifest with a different final model."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mla_recsys.artifacts import atomic_write_json, fingerprint_file, utc_now  # noqa: E402


FINAL_FILES = (
    "model.pt",
    "candidate_embeddings.npy",
    "candidate_metadata.parquet",
    "manifest.json",
)


def materialize_variant(
    *, source_artifact: Path, target_artifact: Path, final_artifact: Path
) -> dict:
    source_artifact = source_artifact.resolve()
    target_artifact = target_artifact.resolve()
    final_artifact = final_artifact.resolve()
    if source_artifact == target_artifact:
        raise ValueError("Target must not overwrite the source walk-forward artifact")

    source_manifest_path = source_artifact / "manifest.json"
    if not source_manifest_path.is_file():
        raise FileNotFoundError(f"Source manifest is missing: {source_manifest_path}")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("status") != "completed":
        raise RuntimeError("Source walk-forward artifact is not completed")
    if not source_manifest.get("snapshots"):
        raise RuntimeError("Source walk-forward artifact has no OOF snapshots")

    final_files = [final_artifact / name for name in FINAL_FILES]
    missing = [str(path) for path in final_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Final TwoTower artifact is incomplete: {missing}")

    manifest = dict(source_manifest)
    manifest.update(
        final_artifact=str(final_artifact),
        final_artifact_source="immutable_variant_override",
        final_lifecycle={
            "predict_state": "configured_full_quality_override",
            "trained_on": "full_prevalidation_raw_train",
            "target_week_seen": False,
            "purpose": "validation_and_test",
        },
        variant={
            "created_at": utc_now(),
            "source_artifact": str(source_artifact),
            "source_manifest": fingerprint_file(source_manifest_path),
            "original_final_artifact": str(source_manifest.get("final_artifact") or ""),
            "selected_final_artifact": str(final_artifact),
            "selected_final_inputs": [fingerprint_file(path) for path in final_files],
        },
    )

    target_manifest = target_artifact / "manifest.json"
    if target_artifact.exists():
        if not target_manifest.is_file():
            raise FileExistsError(f"Target exists without a manifest: {target_artifact}")
        existing = json.loads(target_manifest.read_text(encoding="utf-8"))
        existing_variant = dict(existing.get("variant") or {})
        if (
            existing_variant.get("source_artifact") != str(source_artifact)
            or existing_variant.get("selected_final_artifact") != str(final_artifact)
        ):
            raise FileExistsError(f"Target belongs to another variant: {target_artifact}")
        return {
            "status": "reused",
            "target_artifact": str(target_artifact),
            "final_artifact": str(final_artifact),
        }

    target_artifact.mkdir(parents=True)
    atomic_write_json(target_manifest, manifest)
    atomic_write_json(target_artifact / "metrics.json", manifest)
    report = {
        "status": "completed",
        "source_artifact": str(source_artifact),
        "target_artifact": str(target_artifact),
        "final_artifact": str(final_artifact),
    }
    atomic_write_json(target_artifact / "variant.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a non-mutating walk-forward final-model variant"
    )
    parser.add_argument("--source-artifact", type=Path, required=True)
    parser.add_argument("--target-artifact", type=Path, required=True)
    parser.add_argument("--final-artifact", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            materialize_variant(
                source_artifact=args.source_artifact,
                target_artifact=args.target_artifact,
                final_artifact=args.final_artifact,
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
