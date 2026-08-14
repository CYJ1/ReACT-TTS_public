"""Stage C: connect the Stage A response-style planner to the Stage B TTS
backbone and jointly fine-tune, including the counterfactual-listener loss
(design doc §13) that checks the model actually *uses* the listener face
rather than ignoring it.

    python scripts/train_stage_c.py --config configs/stage_c.yaml

The main TTS loss (diffusion + duration) is computed every step. The
style-consistency and counterfactual losses require sampling audio from the
diffusion decoder, which is expensive at full step count -- they're
computed with a low step count (`train.counterfactual.aux_sample_steps`)
and only on a `train.counterfactual.prob` fraction of steps, both
configurable.
"""
from __future__ import annotations

import argparse
import os
import random

import torch
from torch.utils.data import DataLoader

from react_tts.config import load_config
from react_tts.data.tts_dataset import TTSDialogueDataset
from react_tts.losses import counterfactual_losses, emotion_loss, prosody_loss, style_consistency_loss, weighted_sum
from react_tts.modules.auxiliary_encoders import AuxiliaryEncoders
from react_tts.models.react_tts import build_react_tts


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/stage_c.yaml")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def build_dataset(cfg, stage_a_cfg, stage_b_cfg, manifest_path, synthetic):
    dialogue_kwargs = dict(
        tokenizer_spec=stage_a_cfg.data.tokenizer,
        vocab_size=stage_a_cfg.model.text.vocab_size,
        num_context_turns=stage_a_cfg.data.num_context_turns,
        max_text_len=stage_a_cfg.data.max_text_len,
        reaction_num_frames=stage_a_cfg.data.reaction_num_frames,
        face_feat_dim=stage_a_cfg.model.reaction.face_feat_dim,
    )
    tts_kwargs = dict(
        n_mels=stage_b_cfg.model.n_mels,
        max_text_len=stage_b_cfg.data.max_text_len,
        max_mel_len=stage_b_cfg.data.max_mel_len,
        speaker_emb_dim=stage_b_cfg.model.speaker_emb_dim,
        phoneme_vocab_size=stage_b_cfg.model.phoneme_vocab_size,
    )
    return TTSDialogueDataset(
        manifest_path, dialogue_kwargs=dialogue_kwargs, tts_kwargs=tts_kwargs, synthetic=synthetic, synthetic_size=64
    )


def main():
    args = parse_args()
    cfg = load_config(args.config)
    stage_a_cfg = load_config(cfg.stage_a_config)
    stage_b_cfg = load_config(cfg.stage_b_config)
    if args.epochs:
        cfg.train.epochs = args.epochs
    torch.manual_seed(cfg.get("seed", 42))
    device = args.device

    train_ds = build_dataset(cfg, stage_a_cfg, stage_b_cfg, cfg.data.get("manifest_train"), cfg.data.synthetic)
    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True)

    model = build_react_tts(stage_a_cfg, stage_b_cfg, freeze_style_predictor=cfg.train.freeze_style_predictor).to(device)

    if os.path.exists(cfg.get("stage_a_ckpt", "")):
        model.style_predictor.load_state_dict(torch.load(cfg.stage_a_ckpt, map_location=device))
        print(f"Loaded Stage A checkpoint from {cfg.stage_a_ckpt}")
    if os.path.exists(cfg.get("stage_b_ckpt", "")):
        ckpt = torch.load(cfg.stage_b_ckpt, map_location=device)
        model.tts_backbone.load_state_dict(ckpt.get("backbone", ckpt))
        print(f"Loaded Stage B checkpoint from {cfg.stage_b_ckpt}")

    aux_encoders = AuxiliaryEncoders(n_mels=stage_b_cfg.model.n_mels, style_dim=stage_b_cfg.model.style_emb_dim).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad] + list(aux_encoders.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    os.makedirs(cfg.train.ckpt_dir, exist_ok=True)

    weights = cfg.train.loss_weights
    cf_cfg = cfg.train.counterfactual
    step = 0
    for epoch in range(cfg.train.epochs):
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            out = model.forward_train(batch, style_source=cfg.train.style_source)
            tts_loss = out["diffusion_loss"] + stage_b_cfg.train.duration_loss_weight * out["duration_loss"]

            losses = {"tts": tts_loss}
            if out["style_out"] is not None:
                losses["emotion"] = emotion_loss(out["style_out"]["emotion_logits"], batch["emotion_id"])
                losses["prosody"] = prosody_loss(out["style_out"]["prosody"], batch["prosody"])

            do_counterfactual = (
                cf_cfg.enabled
                and "cf_negative_face_features" in batch
                and random.random() < cf_cfg.prob
            )
            if do_counterfactual:
                neg_batch = {**batch, "face_features": batch["cf_negative_face_features"], "frame_mask": batch["cf_negative_frame_mask"]}
                pos_batch = {**batch, "face_features": batch["cf_positive_face_features"], "frame_mask": batch["cf_positive_frame_mask"]}
                out_neg = model.infer(neg_batch, style_source="predicted", num_steps=cf_cfg.aux_sample_steps)
                out_pos = model.infer(pos_batch, style_source="predicted", num_steps=cf_cfg.aux_sample_steps)

                content_neg = aux_encoders.content_embedding(out_neg["mel"], out_neg["mel_mask"])
                content_pos = aux_encoders.content_embedding(out_pos["mel"], out_pos["mel_mask"])
                speaker_neg = aux_encoders.speaker_embedding(out_neg["mel"], out_neg["mel_mask"])
                speaker_pos = aux_encoders.speaker_embedding(out_pos["mel"], out_pos["mel_mask"])
                style_neg = aux_encoders.style_embedding(out_neg["mel"], out_neg["mel_mask"])
                style_pos = aux_encoders.style_embedding(out_pos["mel"], out_pos["mel_mask"])

                cf = counterfactual_losses(
                    content_neg, content_pos, speaker_neg, speaker_pos, style_neg, style_pos, margin=cf_cfg.margin
                )
                losses["counterfactual_content"] = cf["counterfactual_content"]
                losses["counterfactual_speaker"] = cf["counterfactual_speaker"]
                losses["counterfactual_separation"] = cf["counterfactual_separation"]

                # style consistency: the (primary-listener) generated style should match the predicted style
                gen_style = aux_encoders.style_embedding(out_neg["mel"], out_neg["mel_mask"])
                losses["style_consistency"] = style_consistency_loss(gen_style, out["style_embedding"])

            loss = weighted_sum(losses, weights)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, cfg.train.grad_clip)
            optimizer.step()

            if step % cfg.train.log_every == 0:
                readable = {k: round(v.item(), 4) for k, v in losses.items()}
                print(f"epoch {epoch} step {step} total {loss.item():.4f} {readable}")
            step += 1

        ckpt = {"style_predictor": model.style_predictor.state_dict(), "tts_backbone": model.tts_backbone.state_dict()}
        torch.save(ckpt, os.path.join(cfg.train.ckpt_dir, "last.pt"))

    print("Stage C training complete.")


if __name__ == "__main__":
    main()
