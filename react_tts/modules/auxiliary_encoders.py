"""Auxiliary mel encoders used only by the counterfactual / style-consistency
losses (E_ASR content embedding, E_spk speaker embedding, E_style style
embedding in the design doc's §12-13).

For real experiments these should be **frozen, pretrained** networks (an
ASR encoder for content, a speaker-verification encoder for identity, an
emotion/prosody encoder for style) -- exactly the "frozen pretrained
representation" pattern used elsewhere in this codebase (see
`listener_reaction_encoder.py`, `text_context_encoder.py`'s tokenizer note).

`PlaceholderMelEncoder` is a small trainable conv encoder that exists so the
loss functions and the counterfactual pipeline are shape-correct and
testable without downloading checkpoints. Swap it for a real frozen encoder
before running real experiments (e.g. wav2vec2/Whisper encoder for content,
ECAPA-TDNN/ResNet speaker embedding for identity, emotion2vec for style).
"""
from __future__ import annotations

import torch
from torch import nn

from react_tts.utils.masking import masked_mean


class PlaceholderMelEncoder(nn.Module):
    def __init__(self, n_mels: int = 80, hidden_dim: int = 128, out_dim: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_mels, hidden_dim, 5, padding=2), nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, 5, padding=2), nn.GELU(),
        )
        self.proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, mel: torch.Tensor, mel_mask: torch.Tensor) -> torch.Tensor:
        """mel: [B, n_mels, T], mel_mask: [B, T] bool -> [B, out_dim]."""
        h = self.conv(mel).transpose(1, 2)  # [B, T, hidden]
        pooled = masked_mean(h, mel_mask, dim=1)
        return self.proj(pooled)


class AuxiliaryEncoders(nn.Module):
    """Bundles the three encoders needed by L_style / the counterfactual losses."""

    def __init__(self, n_mels: int = 80, content_dim: int = 256, speaker_dim: int = 256, style_dim: int = 256):
        super().__init__()
        self.content_encoder = PlaceholderMelEncoder(n_mels, out_dim=content_dim)
        self.speaker_encoder = PlaceholderMelEncoder(n_mels, out_dim=speaker_dim)
        self.style_encoder = PlaceholderMelEncoder(n_mels, out_dim=style_dim)

    def content_embedding(self, mel, mel_mask):
        return self.content_encoder(mel, mel_mask)

    def speaker_embedding(self, mel, mel_mask):
        return self.speaker_encoder(mel, mel_mask)

    def style_embedding(self, mel, mel_mask):
        return self.style_encoder(mel, mel_mask)
