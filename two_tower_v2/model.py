from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F


def embedding_dimension(
    cardinality: int,
    *,
    multiplier: float,
    min_dim: int,
    max_dim: int,
    round_to: int,
) -> int:
    """Return a capped hardware-friendly approximation to multiplier*n**0.25."""

    if cardinality <= 0:
        raise ValueError("cardinality must be positive")
    if multiplier <= 0 or min_dim <= 0 or max_dim < min_dim or round_to <= 0:
        raise ValueError("invalid embedding dimension policy")
    raw = multiplier * cardinality**0.25
    rounded = int(math.ceil(raw / round_to) * round_to)
    return max(min_dim, min(max_dim, rounded))


class FieldAwareEncoder(nn.Module):
    def __init__(
        self,
        *,
        cardinalities: Mapping[str, int],
        multiplier: float,
        min_dim: int,
        max_dim: int,
        round_to: int,
    ) -> None:
        super().__init__()
        if not cardinalities:
            raise ValueError("at least one field is required")
        self.field_names = tuple(cardinalities)
        self.dimensions = {
            name: embedding_dimension(
                int(cardinality),
                multiplier=multiplier,
                min_dim=min_dim,
                max_dim=max_dim,
                round_to=round_to,
            )
            for name, cardinality in cardinalities.items()
        }
        self.embeddings = nn.ModuleDict(
            {
                name: nn.EmbeddingBag(
                    int(cardinalities[name]),
                    self.dimensions[name],
                    mode="mean",
                    include_last_offset=False,
                )
                for name in self.field_names
            }
        )
        for embedding in self.embeddings.values():
            nn.init.normal_(embedding.weight, std=0.02)

    @property
    def output_dim(self) -> int:
        return sum(self.dimensions.values())

    def forward(
        self,
        bags: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        return torch.cat(
            [self.embeddings[name](*bags[name]) for name in self.field_names],
            dim=1,
        )


class CrossLayer(nn.Module):
    """Full-matrix DCNv2 cross layer: x <- x0 * (W*x + b) + x."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x0: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return x0 * self.linear(x) + x


class DeepLayer(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float, *, residual: bool) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.residual = bool(residual)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        transformed = self.dropout(F.gelu(self.norm(self.linear(x))))
        return x + transformed if self.residual else transformed


class DcnTower(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        cross_layers: int,
        deep_layers: int,
        dropout: float,
        deep_residual: bool = False,
    ) -> None:
        super().__init__()
        if cross_layers <= 0 or deep_layers <= 0:
            raise ValueError("cross_layers and deep_layers must be positive")
        self.input = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.cross = nn.ModuleList(
            [CrossLayer(hidden_dim) for _ in range(cross_layers)]
        )
        self.deep = nn.ModuleList(
            [
                DeepLayer(hidden_dim, dropout, residual=deep_residual)
                for _ in range(deep_layers)
            ]
        )
        self.output = nn.Linear(2 * hidden_dim, output_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x0 = self.input(features)
        cross = x0
        for layer in self.cross:
            cross = layer(x0, cross)
        deep = x0
        for layer in self.deep:
            deep = layer(deep)
        return F.normalize(self.output(torch.cat([cross, deep], dim=1)), dim=1)


class TwoTowerV2(nn.Module):
    def __init__(
        self,
        *,
        query_cardinalities: Mapping[str, int],
        banner_cardinalities: Mapping[str, int],
        embedding_policy: Mapping[str, float | int],
        hidden_dim: int,
        output_dim: int,
        cross_layers: int,
        deep_layers: int,
        dropout: float,
        deep_residual: bool = False,
    ) -> None:
        super().__init__()
        policy = {
            "multiplier": float(embedding_policy["multiplier"]),
            "min_dim": int(embedding_policy["min_dim"]),
            "max_dim": int(embedding_policy["max_dim"]),
            "round_to": int(embedding_policy["round_to"]),
        }
        self.query_fields = FieldAwareEncoder(
            cardinalities=query_cardinalities,
            **policy,
        )
        self.banner_fields = FieldAwareEncoder(
            cardinalities=banner_cardinalities,
            **policy,
        )
        self.query_tower = DcnTower(
            input_dim=self.query_fields.output_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            cross_layers=cross_layers,
            deep_layers=deep_layers,
            dropout=dropout,
            deep_residual=deep_residual,
        )
        self.banner_tower = DcnTower(
            input_dim=self.banner_fields.output_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            cross_layers=cross_layers,
            deep_layers=deep_layers,
            dropout=dropout,
            deep_residual=deep_residual,
        )

    def encode_query(
        self,
        bags: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        selected = {name: bags[name] for name in self.query_fields.field_names}
        return self.query_tower(self.query_fields(selected))

    def encode_banner(
        self,
        bags: Mapping[str, tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        selected = {name: bags[name] for name in self.banner_fields.field_names}
        return self.banner_tower(self.banner_fields(selected))
