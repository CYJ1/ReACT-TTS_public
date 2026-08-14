from __future__ import annotations

import torch
import torch.nn as nn


class ResponseStyleAdapter(nn.Module):
    """Maps frozen Stage-A response representation to TTS style space."""

    def __init__(
        self,
        input_dim: int = 256,
        output_dim: int = 256,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(
        self,
        response_embedding: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(response_embedding)
