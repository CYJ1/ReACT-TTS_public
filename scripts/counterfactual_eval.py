"""RQ3 / controllability evaluation (design doc §13, §15-16):

1. Counterfactual listener swap: same target text + speaker, different
   listener reaction clip (negative vs. positive). Reports content /
   speaker / style distances between the two generations -- content and
   speaker identity should stay close, style should separate.

2. Random-listener control (the "does the model actually use the face"
   probe): compares Stage A style predictions using the correct listener
   reaction vs. a within-batch-shuffled ("random") one. If emotion/prosody
   metrics barely change, the model is ignoring the visual modality.

    python scripts/counterfactual_eval.py --stage_a_config configs/stage_a.yaml \
        --stage_b_config configs/stage_b.yaml --stage_a_ckpt checkpoints/stage_a/best.pt \
        --stage_b_ckpt checkpoints/stage_b/best.pt
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from eval.metrics import emotion_accuracy_f1, prosody_metrics, vad_ccc
from react_tts.config import load_config
from react_tts.data.dialogue_dataset import DialogueStyleDataset
from react_tts.data.tts_dataset import TTSDialogueDataset
from react_tts.losses import counterfactual_losses
from react_tts.modules.auxiliary_encoders import AuxiliaryEncoders
from react_tts.models.react_tts import build_react_tts


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage_a_config", default="configs/stage_a.yaml")
    p.add_argument("--stage_b_config", default="configs/stage_b.yaml")
    p.add_argument("--stage_c_config", default="configs/stage_c.yaml")
    p.add_argument("--stage_a_ckpt", default=None)
    p.add_argument("--stage_b_ckpt", default=None)
    p.add_argument("--num_steps", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--stage_a_only", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def random_listener_control(
    stage_a_cfg,
    stage_a_ckpt,
    device,
    batch_size,
):
    from react_tts.models.response_style_predictor import (
        build_response_style_predictor,
    )

    d = stage_a_cfg.data

    val_ds = DialogueStyleDataset(
        d.get("manifest_val"),
        tokenizer_spec=d.tokenizer,
        vocab_size=stage_a_cfg.model.text.vocab_size,
        num_context_turns=d.num_context_turns,
        max_text_len=d.max_text_len,
        reaction_num_frames=d.reaction_num_frames,
        face_feat_dim=stage_a_cfg.model.reaction.face_feat_dim,
        synthetic=d.synthetic,
    )

    loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
    )

    model = build_response_style_predictor(stage_a_cfg).to(device)

    if stage_a_ckpt is None:
        raise ValueError(
            "--stage_a_ckpt is required for random-listener control"
        )

    ckpt = torch.load(stage_a_ckpt, map_location=device)

    # Support either raw state_dict or wrapped checkpoint.
    if isinstance(ckpt, dict) and "model" in ckpt:
        ckpt = ckpt["model"]

    model.load_state_dict(ckpt)
    model.eval()

    print(f"[loaded Stage A checkpoint] {stage_a_ckpt}")

    results = {}

    for tag, override in [
        ("correct_listener", False),
        ("random_listener", True),
    ]:
        preds, targets = [], []

        with torch.no_grad():
            for batch in loader:
                batch = {
                    k: v.to(device)
                    for k, v in batch.items()
                }

                out = model(
                    batch,
                    random_listener_override=override,
                )

                preds.append(
                    out["emotion_logits"]
                    .argmax(dim=-1)
                    .cpu()
                    .numpy()
                )

                targets.append(
                    batch["emotion_id"].cpu().numpy()
                )

        results[tag] = emotion_accuracy_f1(
            np.concatenate(preds),
            np.concatenate(targets),
            stage_a_cfg.data.num_emotion_classes,
        )

    return results


def counterfactual_generation_eval(stage_a_cfg, stage_b_cfg, stage_a_ckpt, stage_b_ckpt, device, num_steps, batch_size):
    model = build_react_tts(stage_a_cfg, stage_b_cfg).to(device).eval()
    if stage_a_ckpt:
        model.style_predictor.load_state_dict(torch.load(stage_a_ckpt, map_location=device))
    if stage_b_ckpt:
        ckpt = torch.load(stage_b_ckpt, map_location=device)
        model.tts_backbone.load_state_dict(ckpt.get("backbone", ckpt))

    aux_encoders = AuxiliaryEncoders(n_mels=stage_b_cfg.model.n_mels, style_dim=stage_b_cfg.model.style_emb_dim).to(device).eval()

    val_ds = TTSDialogueDataset(
        None,
        dialogue_kwargs=dict(
            tokenizer_spec=stage_a_cfg.data.tokenizer, vocab_size=stage_a_cfg.model.text.vocab_size,
            num_context_turns=stage_a_cfg.data.num_context_turns,
            max_text_len=stage_a_cfg.data.max_text_len, reaction_num_frames=stage_a_cfg.data.reaction_num_frames,
            face_feat_dim=stage_a_cfg.model.reaction.face_feat_dim,
        ),
        tts_kwargs=dict(
            n_mels=stage_b_cfg.model.n_mels, max_text_len=stage_b_cfg.data.max_text_len,
            max_mel_len=stage_b_cfg.data.max_mel_len, speaker_emb_dim=stage_b_cfg.model.speaker_emb_dim,
            phoneme_vocab_size=stage_b_cfg.model.phoneme_vocab_size,
        ),
        synthetic=True, synthetic_size=32,
    )
    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    all_losses = {"counterfactual_content": [], "counterfactual_speaker": [], "counterfactual_separation": [], "style_distance": []}
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            neg = {**batch, "face_features": batch["cf_negative_face_features"], "frame_mask": batch["cf_negative_frame_mask"]}
            pos = {**batch, "face_features": batch["cf_positive_face_features"], "frame_mask": batch["cf_positive_frame_mask"]}
            out_neg = model.infer(neg, num_steps=num_steps)
            out_pos = model.infer(pos, num_steps=num_steps)

            content_neg = aux_encoders.content_embedding(out_neg["mel"], out_neg["mel_mask"])
            content_pos = aux_encoders.content_embedding(out_pos["mel"], out_pos["mel_mask"])
            speaker_neg = aux_encoders.speaker_embedding(out_neg["mel"], out_neg["mel_mask"])
            speaker_pos = aux_encoders.speaker_embedding(out_pos["mel"], out_pos["mel_mask"])
            style_neg = aux_encoders.style_embedding(out_neg["mel"], out_neg["mel_mask"])
            style_pos = aux_encoders.style_embedding(out_pos["mel"], out_pos["mel_mask"])

            cf = counterfactual_losses(content_neg, content_pos, speaker_neg, speaker_pos, style_neg, style_pos)
            for k in all_losses:
                all_losses[k].append(cf[k].item())

    return {k: float(np.mean(v)) for k, v in all_losses.items()}


def main():
    args = parse_args()
    stage_a_cfg = load_config(args.stage_a_config)
    stage_b_cfg = load_config(args.stage_b_config)

    print("=== Random-listener control (RQ1, modality-neglect probe) ===")
    rl = random_listener_control(stage_a_cfg, args.stage_a_ckpt, args.device, args.batch_size,)
    for tag, metrics in rl.items():
        print(f"  {tag}: {metrics}")

    if args.stage_a_only:
        return

    print("\n=== Counterfactual listener swap (RQ3, controllability probe) ===")
    print("  NOTE: uses PlaceholderMelEncoder for content/speaker/style unless swapped for pretrained encoders.")
    cf = counterfactual_generation_eval(
        stage_a_cfg, stage_b_cfg, args.stage_a_ckpt, args.stage_b_ckpt, args.device, args.num_steps, args.batch_size
    )
    print(f"  {cf}")
    print("  (want: content/speaker distance LOW, style distance/separation-satisfying HIGH)")


if __name__ == "__main__":
    main()
