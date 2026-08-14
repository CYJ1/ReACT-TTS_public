"""ReACT-TTS: Stage A (response style planner) + Stage B (expressive TTS
backbone) wired together, plus the counterfactual-listener generation used
for RQ3 / controllability evaluation.

    a_t_hat = G(x_t, s_t, H_t, v_{t-1}^L)

`style_source`:
  - "predicted"    -> style comes from `ResponseStylePredictor` (the real
    end-to-end pipeline; used for Stage C fine-tuning and inference).
  - "ground_truth" -> style comes straight from `batch["target_style_embedding"]`
    (only used to sanity-check the TTS backbone in isolation, e.g. during
    Stage B pretraining/debugging).
"""
from __future__ import annotations

import torch
from torch import nn

from react_tts.config import Config
from react_tts.models.response_style_predictor import ResponseStylePredictor, build_response_style_predictor
from react_tts.tts.backbone import TTSBackbone


class ReactTTS(nn.Module):
    def __init__(self, style_predictor: ResponseStylePredictor, tts_backbone: TTSBackbone, freeze_style_predictor: bool = False):
        super().__init__()
        self.style_predictor = style_predictor
        self.tts_backbone = tts_backbone
        self.freeze_style_predictor = freeze_style_predictor
        if freeze_style_predictor:
            for p in self.style_predictor.parameters():
                p.requires_grad_(False)

    def _resolve_style(self, batch: dict, style_source: str, random_listener_override: bool | None = None):
        if style_source == "ground_truth":
            return batch["target_style_embedding"], None
        ctx = torch.no_grad() if self.freeze_style_predictor else _nullcontext()
        with ctx:
            style_out = self.style_predictor(batch, random_listener_override=random_listener_override)
        return style_out["style_embedding"], style_out

    def forward_train(self, batch: dict, style_source: str = "predicted") -> dict:
        style_emb, style_out = self._resolve_style(batch, style_source)
        tts_out = self.tts_backbone.forward_train(
            phoneme_ids=batch["phoneme_ids"],
            phoneme_mask=batch["phoneme_mask"],
            durations=batch["durations"],
            target_mel=batch["target_mel"],
            mel_mask=batch["mel_mask"],
            speaker_emb=batch["speaker_embedding"],
            style_emb=style_emb,
        )
        return {**tts_out, "style_out": style_out, "style_embedding": style_emb}

    @torch.no_grad()
    def infer(self, batch: dict, style_source: str = "predicted", num_steps: int | None = None,
              random_listener_override: bool | None = None) -> dict:
        style_emb, style_out = self._resolve_style(batch, style_source, random_listener_override)
        tts_out = self.tts_backbone.infer(
            phoneme_ids=batch["phoneme_ids"],
            phoneme_mask=batch["phoneme_mask"],
            speaker_emb=batch["speaker_embedding"],
            style_emb=style_emb,
            num_steps=num_steps,
        )
        return {**tts_out, "style_out": style_out, "style_embedding": style_emb}

    @torch.no_grad()
    def counterfactual(self, batch: dict, negative_reaction: dict, positive_reaction: dict,
                        num_steps: int | None = None) -> dict:
        """Hold target text + target speaker fixed, swap only the listener
        reaction clip, and generate both. `negative_reaction` /
        `positive_reaction` are dicts overriding `batch["face_features"]` /
        `batch["frame_mask"]` (e.g. {"face_features": ..., "frame_mask": ...}).
        """
        batch_neg = {**batch, **negative_reaction}
        batch_pos = {**batch, **positive_reaction}
        out_neg = self.infer(batch_neg, style_source="predicted", num_steps=num_steps)
        out_pos = self.infer(batch_pos, style_source="predicted", num_steps=num_steps)
        return {"negative": out_neg, "positive": out_pos}


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def build_react_tts(stage_a_cfg: Config, stage_b_cfg: Config, freeze_style_predictor: bool = False) -> ReactTTS:
    style_predictor = build_response_style_predictor(stage_a_cfg)
    m = stage_b_cfg.model
    tts_backbone = TTSBackbone(
        phoneme_vocab_size=m.phoneme_vocab_size,
        n_mels=m.n_mels,
        speaker_emb_dim=m.speaker_emb_dim,
        style_emb_dim=m.style_emb_dim,
        encoder_cfg=m.encoder,
        duration_cfg=m.duration_predictor,
        decoder_cfg=m.decoder,
    )
    return ReactTTS(style_predictor, tts_backbone, freeze_style_predictor=freeze_style_predictor)
