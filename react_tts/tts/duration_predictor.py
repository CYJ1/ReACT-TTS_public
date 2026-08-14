"""Per-phoneme duration predictor, conditioned on the global speaker+style
vector so that response style (e.g. slower/hesitant vs. brisk) shapes timing,
not only pitch/energy."""
from __future__ import annotations

import torch
from torch import nn


class DurationPredictor(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 256,
        kernel_size: int = 3,
        num_layers: int = 2,
        dropout: float = 0.5,
        cond_dim: int | None = None,
    ):
        super().__init__()
        self.cond_proj = nn.Linear(cond_dim, hidden_dim) if cond_dim else None
        layers = []
        for _ in range(num_layers):
            layers.append(
                nn.Sequential(
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
            )
        self.convs = nn.ModuleList(layers)
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.proj = nn.Linear(hidden_dim, 1)

    def forward(self, h_ph: torch.Tensor, phoneme_mask: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        """h_ph: [B, L, H], phoneme_mask: [B, L] bool -> log_duration [B, L]."""
        x = h_ph
        if self.cond_proj is not None and cond is not None:
            x = x + self.cond_proj(cond).unsqueeze(1)
        mask = phoneme_mask.unsqueeze(-1).to(x.dtype)
        x = x * mask
        for conv_block, norm in zip(self.convs, self.norms):
            h = x.transpose(1, 2)
            h = conv_block(h)
            h = h.transpose(1, 2)
            h = norm(h)
            x = h * mask
        log_dur = self.proj(x).squeeze(-1)
        return log_dur * phoneme_mask.to(log_dur.dtype)
