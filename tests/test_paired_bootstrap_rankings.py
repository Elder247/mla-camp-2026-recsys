from __future__ import annotations

import numpy as np

from scripts.paired_bootstrap_rankings import bootstrap_ratio_delta


def test_bootstrap_ratio_delta_is_paired_and_reproducible() -> None:
    control = np.asarray([0.0, 1.0, 0.0])
    candidate = np.asarray([1.0, 1.0, 0.0])
    denominator = np.asarray([1.0, 1.0, 1.0])
    first = bootstrap_ratio_delta(
        control, candidate, denominator, samples=2000, seed=7
    )
    second = bootstrap_ratio_delta(
        control, candidate, denominator, samples=2000, seed=7
    )
    assert first == second
    assert first["mean"] > 0.0
    assert first["p_delta_ge_0"] == 1.0
