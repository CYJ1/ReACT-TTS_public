"""Dialogue text context encoder.

Encodes the last K dialogue turns plus the target sentence. Following the
paper design, turns are NOT simply concatenated: each turn carries explicit
metadata embeddings (current speaker vs. interlocutor, turn distance from
the target, and an is-target flag) so the model can tell *who* said *what*
*how long ago*, and *which* turn is the sentence about to be spoken.

    [SPK_A] I changed several parts of your presentation.
    [SPK_B] Oh... okay.
    [TARGET_SPK_A] I'm sorry if that made you uncomfortable.

Two-level (hierarchical) encoder:
  1. a shared token-level Transformer encodes each turn independently and
     mean-pools it into a turn embedding;
  2. metadata embeddings are added to each turn embedding, and a small
     turn-level Transformer contextualizes the turn sequence (this is the
     "dialogue history" the fusion module attends over).

The target turn's contextualized embedding is also returned separately as
the query used downstream (Stage A fusion, and RQ3's target-text-conditioned
style planning: the same listener reaction must map to different styles
depending on what the target sentence says).
"""
from __future__ import annotations

import torch
from torch import nn

from react_tts.modules.positional import SinusoidalPositionalEncoding

ROLE_TARGET_SPEAKER = 0
ROLE_INTERLOCUTOR = 1
ROLE_PAD = 2


class TextContextEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 4,
        ffn_dim: int = 1024,
        max_turns: int = 8,
        dropout: float = 0.1,
        pad_id: int = 0,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.hidden_dim = hidden_dim

        self.token_embedding = nn.Embedding(vocab_size, hidden_dim, padding_idx=pad_id)
        self.token_pos_encoding = SinusoidalPositionalEncoding(hidden_dim)
        token_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=ffn_dim,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.token_encoder = nn.TransformerEncoder(token_layer, num_layers=max(1, num_layers // 2), enable_nested_tensor=False,)

        # turn-level metadata embeddings
        self.role_embedding = nn.Embedding(3, hidden_dim)  # target-speaker / interlocutor / pad
        self.turn_distance_embedding = nn.Embedding(max_turns, hidden_dim)  # 0 = target turn
        self.is_target_embedding = nn.Embedding(2, hidden_dim)

        turn_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=ffn_dim,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.turn_encoder = nn.TransformerEncoder(turn_layer, num_layers=max(1, num_layers // 2), enable_nested_tensor=False,)
        self.dropout = nn.Dropout(dropout)

    def _encode_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        """token_ids: [B*T, L] -> turn embeddings [B*T, H] (mean pool over valid tokens)."""
        pad_mask = token_ids.eq(self.pad_id)  # [B*T, L], True where padding
        x = self.token_embedding(token_ids)
        x = self.token_pos_encoding(x)
        # transformer needs at least one non-pad token per row; guard fully-padded rows
        fully_padded = pad_mask.all(dim=1)
        safe_pad_mask = pad_mask & (~fully_padded.unsqueeze(1))
        x = self.token_encoder(x, src_key_padding_mask=safe_pad_mask)
        valid = (~pad_mask).unsqueeze(-1).to(x.dtype)
        summed = (x * valid).sum(dim=1)
        counts = valid.sum(dim=1).clamp(min=1.0)
        pooled = summed / counts
        return pooled

    def forward(
        self,
        token_ids: torch.Tensor,      # [B, T, L]
        role_ids: torch.Tensor,       # [B, T] in {ROLE_TARGET_SPEAKER, ROLE_INTERLOCUTOR, ROLE_PAD}
        turn_distance: torch.Tensor,  # [B, T] int, 0 = target turn, increasing into the past
        is_target: torch.Tensor,      # [B, T] in {0, 1}, exactly one 1 per sample
        turn_mask: torch.Tensor,      # [B, T] bool, True where the turn is valid (not padding)
    ):
        B, T, L = token_ids.shape
        turn_emb = self._encode_tokens(token_ids.reshape(B * T, L)).reshape(B, T, self.hidden_dim)

        turn_emb = (
            turn_emb
            + self.role_embedding(role_ids)
            + self.turn_distance_embedding(turn_distance.clamp(max=self.turn_distance_embedding.num_embeddings - 1))
            + self.is_target_embedding(is_target)
        )
        turn_emb = self.dropout(turn_emb)

        turn_pad_mask = ~turn_mask  # True where padding, for attention masking
        h_turns = self.turn_encoder(turn_emb, src_key_padding_mask=turn_pad_mask)

        # pull out the target turn's contextualized embedding as the query
        target_idx = is_target.float().argmax(dim=1)  # [B]
        h_target = h_turns[torch.arange(B, device=h_turns.device), target_idx]

        return {"h_turns": h_turns, "turn_mask": turn_mask, "h_target": h_target}
