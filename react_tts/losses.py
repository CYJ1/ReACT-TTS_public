"""All loss terms from the design doc, §12-13:

    L = L_TTS + lambda1 L_style + lambda2 L_prosody + lambda3 L_emotion
        + lambda4 L_reaction + lambda5 L_counterfactual

L_TTS (diffusion + duration) lives in `react_tts.tts.backbone.TTSBackbone`.
Everything else is here. `L_reaction` is not a separate term in this MVP --
RQ2 (does the reaction delta help) is measured via ablation, not an
auxiliary loss; the hook is left as `reaction_regularization` in case a
future version wants to directly supervise the delta representation.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def emotion_loss(logits: torch.Tensor, target: torch.Tensor, class_weights: torch.Tensor | None = None) -> torch.Tensor:
    return F.cross_entropy(logits, target, weight=class_weights)


def _ccc(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Concordance correlation coefficient, per-dimension, averaged over the batch dim."""
    pred_mean, target_mean = pred.mean(dim=0), target.mean(dim=0)
    pred_var, target_var = pred.var(dim=0, unbiased=False), target.var(dim=0, unbiased=False)
    covariance = ((pred - pred_mean) * (target - target_mean)).mean(dim=0)
    ccc = (2 * covariance) / (pred_var + target_var + (pred_mean - target_mean) ** 2 + 1e-8)
    return ccc  # [D]


def vad_loss(pred_vad: torch.Tensor, target_vad: torch.Tensor) -> torch.Tensor:
    """1 - mean CCC across valence/arousal/dominance -> 0 is perfect agreement."""
    ccc = _ccc(pred_vad, target_vad)
    return (1.0 - ccc).mean()


def prosody_loss(pred_prosody: torch.Tensor, target_prosody: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(pred_prosody, target_prosody)


def style_consistency_loss(generated_style_emb: torch.Tensor, target_style_emb: torch.Tensor) -> torch.Tensor:
    """L_style = 1 - cos(E_audio(a_hat), z_response)."""
    cos = F.cosine_similarity(generated_style_emb, target_style_emb, dim=-1)
    return (1.0 - cos).mean()


def counterfactual_losses(
    content_neg: torch.Tensor, content_pos: torch.Tensor,
    speaker_neg: torch.Tensor, speaker_pos: torch.Tensor,
    style_neg: torch.Tensor, style_pos: torch.Tensor,
    margin: float = 0.2,
) -> dict:
    """Same target text/speaker, listener reaction swapped between a
    "negative" and a "positive" clip (§13). Content and speaker identity
    should stay put; style should move.

    Returns a dict of scalar losses (all "smaller is better", including
    `separation` which is already a hinge loss).
    """
    content_dist = 1.0 - F.cosine_similarity(content_neg, content_pos, dim=-1)
    speaker_dist = 1.0 - F.cosine_similarity(speaker_neg, speaker_pos, dim=-1)
    style_dist = 1.0 - F.cosine_similarity(style_neg, style_pos, dim=-1)

    L_content = content_dist.mean()
    L_spk = speaker_dist.mean()
    L_sep = F.relu(margin - style_dist).mean()  # hinge: penalize when styles are NOT separated enough

    return {
        "counterfactual_content": L_content,
        "counterfactual_speaker": L_spk,
        "counterfactual_separation": L_sep,
        "style_distance": style_dist.mean().detach(),  # for logging, not backprop'd twice
    }


def weighted_sum(losses: dict, weights: dict) -> torch.Tensor:
    total = None
    for name, value in losses.items():
        w = weights.get(name, 0.0)
        if w == 0.0:
            continue
        term = w * value
        total = term if total is None else total + term
    if total is None:
        total = torch.tensor(0.0)
    return total
