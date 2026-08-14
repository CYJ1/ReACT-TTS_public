"""Stage A: dialogue context + listener reaction (+ target text) -> response style.

    z_style = P(e_listener, h_dialogue, x_target, r_speaker)

Note this is deliberately NOT `e_listener -> e_speaker` (emotion mirroring):
the target-text query participates in the fusion cross-attention, so the
same listener reaction can route to different predicted styles depending on
what the target sentence says (apologetic vs. defensive vs. confrontational
for the same "listener looks upset" reaction). RQ3 is tested by comparing
this planner against a mirroring baseline (see `scripts/train_stage_a.py
--mode mirror`).

Every ablation in the paper's Table (text-only / static-face / temporal-face
/ reaction-delta / random-listener) is a config flag here, not a separate
model, so results are directly comparable (same parameter budget).
"""
from __future__ import annotations

import torch
from torch import nn

from react_tts.config import Config
from react_tts.modules.fusion import ContextFusion
from react_tts.modules.listener_reaction_encoder import ListenerReactionEncoder, shuffle_listener_reaction
from react_tts.modules.style_heads import StyleHeads
from react_tts.modules.text_context_encoder import TextContextEncoder


class ResponseStylePredictor(nn.Module):
    def __init__(
        self,
        text_cfg: Config,
        reaction_cfg: Config,
        fusion_cfg: Config,
        style_cfg: Config,
        num_emotion_classes: int,
        pad_id: int = 0,
    ):
        super().__init__()
        self.hidden_dim = fusion_cfg.hidden_dim
        self.use_listener_face = reaction_cfg.use_listener_face
        self.random_listener = reaction_cfg.random_listener

        self.text_encoder = TextContextEncoder(
            vocab_size=text_cfg.vocab_size,
            hidden_dim=text_cfg.hidden_dim,
            num_layers=text_cfg.num_layers,
            num_heads=text_cfg.num_heads,
            ffn_dim=text_cfg.ffn_dim,
            max_turns=text_cfg.max_turns,
            dropout=text_cfg.dropout,
            pad_id=pad_id,
        )
        self.reaction_encoder = ListenerReactionEncoder(
            face_feat_dim=reaction_cfg.face_feat_dim,
            hidden_dim=reaction_cfg.hidden_dim,
            num_layers=reaction_cfg.num_layers,
            num_heads=reaction_cfg.num_heads,
            ffn_dim=reaction_cfg.ffn_dim,
            dropout=reaction_cfg.dropout,
            temporal_face=reaction_cfg.temporal_face,
            use_reaction_delta=reaction_cfg.use_reaction_delta,
        )
        # keep modality hidden dims aligned to the fusion width; project if configs differ
        self.text_proj = (
            nn.Identity() if text_cfg.hidden_dim == fusion_cfg.hidden_dim
            else nn.Linear(text_cfg.hidden_dim, fusion_cfg.hidden_dim)
        )
        self.react_proj = (
            nn.Identity() if reaction_cfg.hidden_dim == fusion_cfg.hidden_dim
            else nn.Linear(reaction_cfg.hidden_dim, fusion_cfg.hidden_dim)
        )
        self.fusion = ContextFusion(
            hidden_dim=fusion_cfg.hidden_dim, num_heads=fusion_cfg.num_heads, dropout=fusion_cfg.dropout
        )
        self.style_heads = StyleHeads(
            hidden_dim=fusion_cfg.hidden_dim,
            emotion_hidden=style_cfg.emotion_hidden,
            num_emotion_classes=num_emotion_classes,
            vad_dim=style_cfg.vad_dim,
            prosody_dim=style_cfg.prosody_dim,
            style_emb_dim=fusion_cfg.hidden_dim,
        )

    def forward(self, batch: dict, random_listener_override: bool | None = None) -> dict:
        text_out = self.text_encoder(
            token_ids=batch["token_ids"],
            role_ids=batch["role_ids"],
            turn_distance=batch["turn_distance"],
            is_target=batch["is_target"],
            turn_mask=batch["turn_mask"],
        )
        h_turns = self.text_proj(text_out["h_turns"])
        h_target = self.text_proj(text_out["h_target"])

        use_random = self.random_listener if random_listener_override is None else random_listener_override

        if self.use_listener_face:
            face_features, frame_mask = batch["face_features"], batch["frame_mask"]
            if use_random:
                face_features, frame_mask = shuffle_listener_reaction(face_features, frame_mask)
            reaction_out = self.reaction_encoder(face_features, frame_mask)
            h_reaction = self.react_proj(reaction_out["h_reaction"])
        else:
            h_reaction = torch.zeros(h_target.size(0), self.hidden_dim, device=h_target.device, dtype=h_target.dtype)

        fusion_out = self.fusion(h_target, h_turns, text_out["turn_mask"], h_reaction)
        style_out = self.style_heads(fusion_out["h_fused"])

        return {
            **style_out,
            "h_fused": fusion_out["h_fused"],
            "gate_text": fusion_out["gate_text"],
            "gate_react": fusion_out["gate_react"],
            "attn_weights": fusion_out["attn_weights"],
        }


def build_response_style_predictor(cfg: Config) -> ResponseStylePredictor:
    m = cfg.model
    return ResponseStylePredictor(
        text_cfg=m.text,
        reaction_cfg=m.reaction,
        fusion_cfg=m.fusion,
        style_cfg=m.style,
        num_emotion_classes=cfg.data.num_emotion_classes,
    )
