from __future__ import annotations

from omegaconf import OmegaConf

from scripts.generate_candidates import _generator_semantics, _same_fingerprint_inputs


def fingerprint(sha256: str, *, path: str) -> dict:
    return {
        "path": path,
        "exists": True,
        "size_bytes": 10,
        "sha256": sha256,
    }


def test_reuse_compares_content_and_ignores_paths() -> None:
    current = [fingerprint("same", path="/new/run/input.parquet")]
    previous = [fingerprint("same", path="/old/run/input.parquet")]

    assert _same_fingerprint_inputs(current, previous)


def test_reuse_rejects_changed_code_or_artifact() -> None:
    current = [fingerprint("new-code", path="temporal_candidates.py")]
    previous = [fingerprint("old-code", path="temporal_candidates.py")]

    assert not _same_fingerprint_inputs(current, previous)


def test_execution_parallelism_is_not_generator_semantics() -> None:
    donor = OmegaConf.create({"top_k": 1000, "quota": 1000, "weight": 0.25})
    current = OmegaConf.create(
        {
            "top_k": 1000,
            "quota": 1000,
            "weight": 0.25,
            "request_workers": 8,
            "parallel_batch_size": 8,
            "parallel_priority": 100,
        }
    )

    assert _generator_semantics(donor) == _generator_semantics(current)


def test_quality_parameters_remain_generator_semantics() -> None:
    donor = OmegaConf.create({"top_k": 500, "quota": 500, "weight": 0.25})
    current = OmegaConf.create({"top_k": 1000, "quota": 500, "weight": 0.25})

    assert _generator_semantics(donor) != _generator_semantics(current)
