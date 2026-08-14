"""Shared sinusoidal positional encoding, used by the text, reaction, and TTS encoders."""
from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, hidden_dim: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, hidden_dim, 2, dtype=torch.float32) * (-math.log(10000.0) / hidden_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, L, H] -> x + positional encoding for the first L positions."""
        return x + self.pe[: x.size(1)].unsqueeze(0).to(dtype=x.dtype, device=x.device)
