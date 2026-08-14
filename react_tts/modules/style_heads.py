"""Prediction heads on top of the fused context representation.

Interpretable style representation (kept as separate heads rather than one
opaque latent, per the design doc):

    z_style = [z_emotion (categorical + VAD), z_prosody (F0/energy/rate)]

`style_embedding` is the continuous vector handed to the TTS backbone for
AdaLN conditioning -- it is the fused representation itself (projected),
*not* reconstructed from the discrete emotion/prosody predictions, so
gradients from the TTS side can still shape it during Stage C joint
fine-tuning.
"""
from __future__ import annotations

import torch
from torch import nn


class StyleHeads(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 256,
        emotion_hidden: int = 128,
        num_emotion_classes: int = 7,
        vad_dim: int = 3,
        prosody_dim: int = 4,
        style_emb_dim: int | None = None,
    ):
        super().__init__()
        style_emb_dim = style_emb_dim or hidden_dim
        self.style_proj = nn.Sequential(
            nn.Linear(hidden_dim, style_emb_dim), nn.GELU(), nn.Linear(style_emb_dim, style_emb_dim)
        )
        self.emotion_head = nn.Sequential(
            nn.Linear(hidden_dim, emotion_hidden), nn.GELU(), nn.Linear(emotion_hidden, num_emotion_classes)
        )
        self.vad_head = nn.Sequential(
            nn.Linear(hidden_dim, emotion_hidden), nn.GELU(), nn.Linear(emotion_hidden, vad_dim), nn.Tanh()
        )
        self.prosody_head = nn.Sequential(
            nn.Linear(hidden_dim, emotion_hidden), nn.GELU(), nn.Linear(emotion_hidden, prosody_dim)
        )

    def forward(self, h_fused: torch.Tensor):
        return {
            "style_embedding": self.style_proj(h_fused),
            "emotion_logits": self.emotion_head(h_fused),
            "vad": self.vad_head(h_fused),          # in [-1, 1]^3 (valence, arousal, dominance)
            "prosody": self.prosody_head(h_fused),   # [f0_mean, f0_std, energy, speaking_rate], speaker-normalized
        }
