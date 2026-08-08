from __future__ import annotations

import numpy as np

from mla_recsys.importance import first_complete_groups, topk_value_capture


def test_topk_value_capture_uses_group_ranking() -> None:
    scores = np.array([0.1, 0.9, 0.8, 0.2])
    labels = np.array([10.0, 0.0, 20.0, 0.0])
    groups = np.array([1, 1, 2, 2])
    assert topk_value_capture(scores, labels, groups, top_k=1) == 20.0 / 30.0


def test_sample_never_cuts_a_group() -> None:
    groups = np.array([1, 1, 2, 2, 2, 3])
    indices = first_complete_groups(groups, 3)
    assert indices.tolist() == [0, 1, 2, 3, 4]
