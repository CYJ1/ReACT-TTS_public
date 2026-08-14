from __future__ import annotations

import torch


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """x: [..., L, ...H], mask: same shape as x minus last dim, True where valid."""
    mask_f = mask.unsqueeze(-1).to(x.dtype)
    summed = (x * mask_f).sum(dim=dim)
    counts = mask_f.sum(dim=dim).clamp(min=1.0)
    return summed / counts
