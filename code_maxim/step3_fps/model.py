from __future__ import annotations

import torch
from torch.nn import functional as F

from code_maxim.step2_ce.model import TwoTowerModel


def sampled_softmax_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """One positive and batch_size - 1 sampled negatives per query.

    There is deliberately no logQ correction in this course step.
    """
    return F.cross_entropy(logits, labels)


__all__ = ["TwoTowerModel", "sampled_softmax_loss"]
