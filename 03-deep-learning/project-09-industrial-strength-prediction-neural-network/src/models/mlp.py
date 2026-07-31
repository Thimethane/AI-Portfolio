"""Multi-Layer Perceptron for concrete compressive strength prediction."""

from __future__ import annotations

import torch
from torch import nn


class ConcreteStrengthMLP(nn.Module):
    """MLP regression network: Linear -> ReLU hidden blocks -> Linear output."""

    def __init__(
        self,
        input_dim: int = 8,
        hidden_layers: tuple[int, ...] = (64, 32),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous_dim = input_dim

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            previous_dim = hidden_dim

        layers.append(nn.Linear(previous_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.network(X)


def build_model(config: dict[str, object] | None = None) -> ConcreteStrengthMLP:
    config = config or {}
    hidden_layers = tuple(config.get("hidden_layers", (64, 32)))
    return ConcreteStrengthMLP(
        input_dim=int(config.get("input_dim", 8)),
        hidden_layers=hidden_layers,
        dropout=float(config.get("dropout", 0.0)),
    )

