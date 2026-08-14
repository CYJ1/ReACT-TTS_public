"""A trivial ground-truth style embedding table, used only to pretrain/
reproduce the Stage B expressive-TTS backbone before Stage A's continuous
planner is connected (the "FEIM-TTS reproduction" step: emotion label ->
expressive speech, no dialogue context involved).

Once Stage C connects the real planner, `ResponseStylePredictor`'s
continuous `style_embedding` output replaces this lookup table entirely.
"""
from __future__ import annotations

import torch
from torch import nn


class GroundTruthStyleEmbedding(nn.Module):
    def __init__(self, num_emotion_classes: int, style_emb_dim: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(num_emotion_classes, style_emb_dim)

    def forward(self, emotion_id: torch.Tensor) -> torch.Tensor:
        return self.embedding(emotion_id)
