from __future__ import annotations

import torch
from torch import nn

from .kan import TemporalKAN


class MicrostructureForecaster(nn.Module):
    """Predict next-event midpoint return from a rolling LOB window."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        depth: int = 3,
        grid_size: int = 8,
    ) -> None:
        super().__init__()
        self.encoder = TemporalKAN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=1,
            depth=depth,
            kernel_size=3,
            grid_size=grid_size,
            spline_order=3,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def regularization_loss(self) -> torch.Tensor:
        return self.encoder.regularization_loss()
