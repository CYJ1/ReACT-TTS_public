"""Target-text-conditioned multimodal fusion.

Two mechanisms, both described in the design doc, used together:

1. Target-conditioned cross-attention -- the target sentence's embedding is
   the *query*; dialogue turns and the listener-reaction embedding are
   *keys/values*. This lets the model pick which past cue matters for the
   sentence about to be spoken (e.g. attend to the negative listener
   reaction and the recent conflict turn when the target sentence is an
   apology, and rely less on the face when the sentence is purely
   informative).

2. A modality gate `g_m = sigmoid(W[h_target; h_m])` per modality. Logging
   `g_react` at eval time is the qualitative analysis the paper wants:
   does the gate open wider on ambiguous transcripts / negative reactions?
"""
from __future__ import annotations

import torch
from torch import nn

from react_tts.utils.masking import masked_mean


class ContextFusion(nn.Module):
    def __init__(self, hidden_dim: int = 256, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.gate_text = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())
        self.gate_react = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.Sigmoid())
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        h_target: torch.Tensor,   # [B, H]
        h_turns: torch.Tensor,    # [B, T, H]
        turn_mask: torch.Tensor,  # [B, T] bool, True = valid
        h_reaction: torch.Tensor,  # [B, H]
    ):
        B = h_target.size(0)
        kv = torch.cat([h_turns, h_reaction.unsqueeze(1)], dim=1)  # [B, T+1, H]
        kv_mask_valid = torch.cat(
            [turn_mask, torch.ones(B, 1, dtype=torch.bool, device=turn_mask.device)], dim=1
        )
        key_padding_mask = ~kv_mask_valid  # True = ignore

        query = h_target.unsqueeze(1)  # [B, 1, H]
        attn_out, attn_weights = self.cross_attn(
            query, kv, kv, key_padding_mask=key_padding_mask, need_weights=True
        )
        h_ctx = attn_out.squeeze(1)  # [B, H]

        h_text_pooled = masked_mean(h_turns, turn_mask, dim=1)  # [B, H]
        g_text = self.gate_text(torch.cat([h_target, h_text_pooled], dim=-1))
        g_react = self.gate_react(torch.cat([h_target, h_reaction], dim=-1))

        fused = h_target + h_ctx + g_text * h_text_pooled + g_react * h_reaction
        fused = self.out_norm(fused)
        fused = self.final_norm(fused + self.out_mlp(fused))

        return {
            "h_fused": fused,
            "gate_text": g_text.mean(dim=-1),   # [B] scalar summary per sample, for logging/analysis
            "gate_react": g_react.mean(dim=-1),  # [B]
            "attn_weights": attn_weights,        # [B, 1, T+1]
        }
