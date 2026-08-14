"""Temporal listener facial-reaction encoder (RQ2: does the *change* in
expression matter more than a single static frame?).

Expects precomputed per-frame face embeddings -- by default 52-dim MediaPipe
FaceLandmarker blendshape scores (see `preprocessing/pretrained_face_embedder.py`),
a frozen pretrained facial-expression descriptor, following the same "use a
frozen pretrained representation" pattern as the speaker/audio encoders. Any
other fixed-dim per-frame face embedding can be swapped in ahead of this
module without changing anything downstream, since this module only
consumes `face_features: [B, N, face_feat_dim]`.

Produces:
  - `h_temporal` [B, N, H]: contextualized per-frame representation.
  - `h_reaction` [B, H]:    the fused reaction representation
        MLP([h_pre; h_post; h_post - h_pre; mean(h_temporal)])
    i.e. current state + explicit pre->post delta + overall temporal summary.

Two architectural ablations are built in (see configs/stage_a.yaml
`model.reaction`), so each row of the ablation table is one flag flip:
  - `temporal_face=False`  -> collapses to a single (first) frame, no
    temporal modeling: the "static listener face" baseline.
  - `use_reaction_delta=False` -> zeroes the explicit delta term fed to the
    fusion MLP (the MLP still runs over the same input width, so capacity is
    matched): the "no reaction-delta" baseline.
"""
from __future__ import annotations

import torch
from torch import nn

from react_tts.modules.positional import SinusoidalPositionalEncoding
from react_tts.utils.masking import masked_mean


class ListenerReactionEncoder(nn.Module):
    def __init__(
        self,
        face_feat_dim: int = 52,  # MediaPipe blendshape score vector, see preprocessing/pretrained_face_embedder.py
        hidden_dim: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
        temporal_face: bool = True,
        use_reaction_delta: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.temporal_face = temporal_face
        self.use_reaction_delta = use_reaction_delta

        self.input_proj = nn.Linear(face_feat_dim, hidden_dim)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=ffn_dim,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False, )
        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, face_features: torch.Tensor, frame_mask: torch.Tensor):
        """
        face_features: [B, N, face_feat_dim]
        frame_mask:    [B, N] bool, True where the frame is valid (not padding).
                       Frame 0 is assumed the earliest ("pre") and the last
                       valid frame per-sample is the latest ("post") in the
                       listener-reaction window.
        """
        B, N, _ = face_features.shape
        x = self.input_proj(face_features)  # [B, N, H]

        if not self.temporal_face:
            # static-listener-face baseline: only the first frame, no temporal model
            static_feat = x[:, 0, :]
            h_temporal = static_feat.unsqueeze(1)
            h_pre = static_feat
            h_post = static_feat
            pooled = static_feat
        else:
            x = self.pos_encoding(x)
            pad_mask = ~frame_mask
            # guard rows that are entirely padding (shouldn't happen in practice)
            fully_padded = pad_mask.all(dim=1)
            safe_pad_mask = pad_mask & (~fully_padded.unsqueeze(1))
            h_temporal = self.transformer(x, src_key_padding_mask=safe_pad_mask)

            lengths = frame_mask.sum(dim=1).clamp(min=1)
            last_idx = (lengths - 1).long()
            batch_idx = torch.arange(B, device=x.device)
            h_pre = h_temporal[:, 0, :]
            h_post = h_temporal[batch_idx, last_idx]
            pooled = masked_mean(h_temporal, frame_mask, dim=1)

        if self.use_reaction_delta:
            delta = h_post - h_pre
        else:
            delta = torch.zeros_like(h_pre)

        h_reaction = self.fusion_mlp(torch.cat([h_pre, h_post, delta, pooled], dim=-1))
        return {"h_reaction": h_reaction, "h_temporal": h_temporal, "frame_mask": frame_mask}


def shuffle_listener_reaction(face_features: torch.Tensor, frame_mask: torch.Tensor, generator=None):
    """Control experiment (modality-neglect probe): permute listener reactions
    across the batch so each sample sees a *mismatched* listener. If model
    performance is roughly unchanged under this permutation, the model is
    ignoring the visual modality. See RQ1 / "random listener" ablation.
    A cyclic shift is used instead of a random permutation so that
    self-matches are impossible whenever batch size > 1.
    """
    B = face_features.size(0)

    if B <= 1:
        return face_features, frame_mask

    idx = torch.roll(
        torch.arange(B, device=face_features.device),
        shifts=-1,
    )

    return face_features[idx], frame_mask[idx]