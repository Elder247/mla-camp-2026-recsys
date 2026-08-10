from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


def stable_hash(value: object, buckets: int) -> int:
    if buckets <= 1:
        raise ValueError("buckets must be greater than one")
    payload = str(value if value is not None else "").encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little") % buckets


@dataclass(frozen=True)
class SampledGroups:
    indices: np.ndarray
    targets: np.ndarray
    weights: np.ndarray


def sample_listwise_groups(
    group_ids: np.ndarray,
    pre_ranks: np.ndarray,
    source_costs: np.ndarray,
    *,
    candidates_per_group: int,
    hard_fraction: float,
    seed: int,
) -> SampledGroups:
    """Build deterministic positive + hard/tail-negative listwise groups."""
    if candidates_per_group < 2:
        raise ValueError("candidates_per_group must be at least two")
    if not 0.0 <= hard_fraction <= 1.0:
        raise ValueError("hard_fraction must be in [0, 1]")
    if not (len(group_ids) == len(pre_ranks) == len(source_costs)):
        raise ValueError("group arrays have different lengths")
    if len(group_ids) == 0:
        raise ValueError("cannot sample empty groups")
    starts = np.r_[0, np.flatnonzero(group_ids[1:] != group_ids[:-1]) + 1]
    ends = np.r_[starts[1:], len(group_ids)]
    keys = group_ids[starts]
    if np.unique(keys).size != keys.size:
        raise ValueError("ranking groups must be contiguous")

    rng = np.random.default_rng(seed)
    sampled: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    raw_weights: list[float] = []
    for start, end in zip(starts, ends):
        local = np.arange(start, end, dtype=np.int64)
        positive = local[source_costs[start:end] > 0.0]
        if positive.size == 0:
            continue
        negative = local[source_costs[start:end] <= 0.0]
        negative = negative[np.argsort(pre_ranks[negative], kind="stable")]
        slots = candidates_per_group - min(len(positive), candidates_per_group - 1)
        hard_count = min(len(negative), int(round(slots * hard_fraction)))
        chosen_negative = list(negative[:hard_count])
        remaining = negative[hard_count:]
        tail_count = slots - len(chosen_negative)
        if tail_count > 0 and len(remaining) > 0:
            picked = rng.choice(
                remaining,
                size=tail_count,
                replace=len(remaining) < tail_count,
            )
            chosen_negative.extend(int(value) for value in picked)
        while len(chosen_negative) < slots:
            fallback = int(negative[0]) if len(negative) else int(positive[0])
            chosen_negative.append(fallback)
        chosen_positive = positive[: candidates_per_group - slots]
        row = np.concatenate(
            [chosen_positive, np.asarray(chosen_negative, dtype=np.int64)]
        )
        order = rng.permutation(len(row))
        row = row[order]
        target = np.sqrt(np.maximum(source_costs[row], 0.0)).astype(np.float32)
        target_sum = float(target.sum())
        if target_sum <= 0.0:
            raise AssertionError("sampled group lost every positive")
        sampled.append(row)
        targets.append(target / target_sum)
        raw_weights.append(float(np.sqrt(source_costs[positive].sum())))

    if not sampled:
        raise ValueError("no positive ranking groups were found")
    weight_values = np.asarray(raw_weights, dtype=np.float64)
    cap = float(np.quantile(weight_values, 0.99))
    weight_values = np.minimum(weight_values, cap)
    weight_values /= float(weight_values.mean())
    return SampledGroups(
        indices=np.stack(sampled),
        targets=np.stack(targets),
        weights=weight_values.astype(np.float32),
    )


class CrossNetV2(nn.Module):
    def __init__(self, dimension: int, layers: int) -> None:
        super().__init__()
        self.kernels = nn.ModuleList(
            [nn.Linear(dimension, dimension) for _ in range(layers)]
        )
        for kernel in self.kernels:
            nn.init.xavier_uniform_(kernel.weight, gain=0.1)
            nn.init.zeros_(kernel.bias)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        base = values
        crossed = values
        for kernel in self.kernels:
            crossed = base * kernel(crossed) + crossed
        return crossed


class DCNv2Ranker(nn.Module):
    """Compact DCNv2 residual ranker for a natural candidate pool."""

    def __init__(
        self,
        continuous_features: int,
        categorical_buckets: list[int],
        embedding_dims: list[int],
        *,
        cross_layers: int = 3,
        deep_dims: tuple[int, ...] = (256, 128),
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if len(categorical_buckets) != len(embedding_dims):
            raise ValueError("categorical bucket and embedding dimensions differ")
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(int(buckets), int(dimension))
                for buckets, dimension in zip(categorical_buckets, embedding_dims)
            ]
        )
        for embedding in self.embeddings:
            nn.init.zeros_(embedding.weight)
        input_dimension = continuous_features + sum(embedding_dims)
        self.cross = CrossNetV2(input_dimension, cross_layers)
        deep: list[nn.Module] = []
        previous = input_dimension
        for dimension in deep_dims:
            deep.extend(
                [
                    nn.Linear(previous, dimension),
                    nn.LayerNorm(dimension),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                ]
            )
            previous = dimension
        self.deep = nn.Sequential(*deep)
        self.output = nn.Linear(input_dimension + previous, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        continuous: torch.Tensor,
        categorical: torch.Tensor,
        base_score: torch.Tensor,
    ) -> torch.Tensor:
        embedded = [
            embedding(categorical[:, index])
            for index, embedding in enumerate(self.embeddings)
        ]
        values = torch.cat([continuous, *embedded], dim=1)
        correction = self.output(torch.cat([self.cross(values), self.deep(values)], dim=1))
        return base_score + correction.squeeze(1)
