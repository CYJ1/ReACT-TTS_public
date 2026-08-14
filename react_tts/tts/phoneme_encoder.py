"""Phoneme/text encoder for the TTS backbone (Grad-TTS lineage: a Transformer
encoder over phonemes, whose output is later projected to a mel-shaped prior
`mu` that the diffusion decoder refines)."""
from __future__ import annotations

import torch
from torch import nn

from react_tts.modules.positional import SinusoidalPositionalEncoding


class PhonemeEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
        pad_id: int = 0,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_id)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=ffn_dim,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, phoneme_ids: torch.Tensor, phoneme_mask: torch.Tensor) -> torch.Tensor:
        """phoneme_ids: [B, L], phoneme_mask: [B, L] bool True=valid -> [B, L, H]."""
        x = self.embedding(phoneme_ids)
        x = self.pos_encoding(x)
        x = self.encoder(x, src_key_padding_mask=~phoneme_mask)
        return self.norm(x)
