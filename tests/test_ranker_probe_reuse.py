from __future__ import annotations

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from scripts.materialize_ranker_probe import (
    materialize_tree,
    ranker_probe_semantics,
    reusable_metric,
    reusable_stage,
    validate_donor,
)


def config(*, loss: str, feature_version: str = "f1", reuse: str | None = None):
    return OmegaConf.create(
        {
            "runtime": {"run_id": "run", "resume": True, "scope": "offline"},
            "ranker": {"loss_function": loss},
            "submission": {"ranking": "catboost"},
            "candidates": {"reuse_run": reuse, "rrf_constant": 40},
            "features": {"reuse_run": reuse, "version": feature_version},
            "data": {"partition_count": 2},
        }
    )


def test_ranker_loss_and_reuse_paths_do_not_change_upstream_semantics() -> None:
    donor = config(loss="YetiRankPairwise")
    probe = config(loss="QueryRMSE", reuse="/donor")

    assert ranker_probe_semantics(donor) == ranker_probe_semantics(probe)


def test_submission_ranking_does_not_change_upstream_semantics() -> None:
    donor = config(loss="YetiRankPairwise")
    probe = config(loss="QueryRMSE")
    probe.submission.ranking = "value_geometry"

    assert ranker_probe_semantics(donor) == ranker_probe_semantics(probe)


def test_feature_change_rejects_upstream_reuse() -> None:
    donor = config(loss="YetiRankPairwise")
    probe = config(loss="QueryRMSE", feature_version="f2")

    assert ranker_probe_semantics(donor) != ranker_probe_semantics(probe)


def test_disabled_candidate_declarations_do_not_change_upstream_semantics() -> None:
    donor = config(loss="YetiRankPairwise")
    probe = config(loss="QueryRMSE")
    donor.candidates.generators = {
        "active": {"enabled": True, "quota": 100},
    }
    probe.candidates.generators = {
        "active": {"enabled": True, "quota": 100},
        "future_probe": {"enabled": False, "quota": 50},
    }

    assert ranker_probe_semantics(donor) == ranker_probe_semantics(probe)


def test_enabled_candidate_change_still_rejects_upstream_reuse() -> None:
    donor = config(loss="YetiRankPairwise")
    probe = config(loss="QueryRMSE")
    donor.candidates.generators = {
        "active": {"enabled": True, "quota": 100},
    }
    probe.candidates.generators = {
        "active": {"enabled": True, "quota": 100},
        "new_active": {"enabled": True, "quota": 50},
    }

    assert ranker_probe_semantics(donor) != ranker_probe_semantics(probe)


def test_materialize_tree_preserves_files_without_overwrite(tmp_path: Path) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    source.mkdir()
    (source / "part.parquet").write_bytes(b"immutable")

    files, logical_bytes = materialize_tree(source, target)

    assert files == 1
    assert logical_bytes == len(b"immutable")
    assert (target / "part.parquet").read_bytes() == b"immutable"
    with pytest.raises(FileExistsError):
        materialize_tree(source, target)


def test_history_feature_profile_excludes_merged_candidates(tmp_path: Path) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    (source / "train" / "history").mkdir(parents=True)
    (source / "train" / "merged").mkdir(parents=True)
    (source / "train" / "history" / "part.parquet").write_bytes(b"source")
    (source / "train" / "merged" / "part.parquet").write_bytes(b"merged")

    files, _ = materialize_tree(
        source, target, excluded_directory_names=frozenset({"merged"})
    )

    assert files == 1
    assert (target / "train" / "history" / "part.parquet").is_file()
    assert not (target / "train" / "merged").exists()


def test_history_feature_profile_reuses_only_true_upstream_contracts() -> None:
    assert reusable_stage("prepare_data", "history_features")
    assert reusable_stage("generate_test_tfidf_v1", "history_features")
    assert not reusable_stage("merge_candidates_test", "history_features")
    assert not reusable_stage("build_features_test", "history_features")
    assert reusable_metric("data.json", "history_features")
    assert reusable_metric("generate_test_tfidf_v1.json", "history_features")
    assert not reusable_metric("merge_test.json", "history_features")


def test_ranker_profile_recomputes_submission_and_validation() -> None:
    assert not reusable_stage("train_ranker", "ranker")
    assert not reusable_stage("make_submission", "ranker")
    assert not reusable_stage("validate_submission", "ranker")
    assert reusable_stage("build_features_test", "ranker")


def test_validate_donor_requires_successful_parity(tmp_path: Path) -> None:
    donor_cfg = config(loss="YetiRankPairwise")
    (tmp_path / "config.yaml").write_text(OmegaConf.to_yaml(donor_cfg))
    (tmp_path / "result.json").write_text(json.dumps({"status": "completed"}))
    for relative in ("data", "counters", "candidates", "features", "metrics"):
        (tmp_path / relative).mkdir()
    (tmp_path / "metrics" / "cache_parity.json").write_text(
        json.dumps({"ok": False})
    )

    with pytest.raises(ValueError, match="parity"):
        validate_donor(tmp_path, config(loss="QueryRMSE"))
