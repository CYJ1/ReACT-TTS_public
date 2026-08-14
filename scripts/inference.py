"""Single-sample inference: dialogue context + listener reaction + target
text + target speaker -> generated mel-spectrogram.

    python scripts/inference.py \
        --stage_a_config configs/stage_a.yaml --stage_b_config configs/stage_b.yaml \
        --stage_a_ckpt checkpoints/stage_a/best.pt --stage_b_ckpt checkpoints/stage_b/best.pt \
        --input_json examples/apology_example.json \
        --out_mel out/apology.npy

`--input_json` schema: same as one line of the Stage A/C manifest, i.e.
`{"context_turns": [...], "target_text": ..., "target_speaker_id": ...,
"listener_reaction_path": "...", "speaker_embedding_path": "..."}`.

This script produces a mel-spectrogram only. Converting mel -> waveform
needs a vocoder (e.g. HiFi-GAN, matching the Face-TTS lineage this backbone
follows) -- out of scope for this MVP; see README "Setup".
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from react_tts.config import load_config
from react_tts.data.dialogue_dataset import DialogueStyleDataset
from react_tts.models.react_tts import build_react_tts
from react_tts.tts.text_frontend import TextFrontend


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage_a_config", default="configs/stage_a.yaml")
    p.add_argument("--stage_b_config", default="configs/stage_b.yaml")
    p.add_argument("--stage_a_ckpt", default=None)
    p.add_argument("--stage_b_ckpt", default=None)
    p.add_argument("--input_json", required=True)
    p.add_argument("--out_mel", default="out/generated.npy")
    p.add_argument("--num_steps", type=int, default=None)
    p.add_argument("--style_source", choices=["predicted", "ground_truth"], default="predicted")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def build_single_batch(sample: dict, stage_a_cfg, stage_b_cfg, device):
    ds = DialogueStyleDataset(
        None,
        tokenizer_spec=stage_a_cfg.data.tokenizer,
        vocab_size=stage_a_cfg.model.text.vocab_size,
        num_context_turns=stage_a_cfg.data.num_context_turns,
        max_text_len=stage_a_cfg.data.max_text_len,
        reaction_num_frames=stage_a_cfg.data.reaction_num_frames,
        face_feat_dim=stage_a_cfg.model.reaction.face_feat_dim,
        synthetic=True,  # only used to reach the tokenizer; we build the sample ourselves below
    )
    token_ids, role_ids, turn_distance, is_target, turn_mask = ds.build_turns(
        sample["context_turns"], sample["target_text"], sample["target_speaker_id"]
    )
    face_features, frame_mask = ds.load_reaction(sample["listener_reaction_path"])

    frontend = TextFrontend()
    phoneme_ids = frontend.text_to_phoneme_ids(sample["target_text"])
    max_text_len = stage_b_cfg.data.max_text_len
    phoneme_mask = [True] * min(len(phoneme_ids), max_text_len)
    phoneme_ids = (phoneme_ids[:max_text_len] + [0] * max_text_len)[:max_text_len]
    phoneme_mask = (phoneme_mask + [False] * max_text_len)[:max_text_len]

    speaker_emb = np.load(sample["speaker_embedding_path"]).astype(np.float32)

    def t(x, dtype):
        return torch.tensor(x, dtype=dtype, device=device).unsqueeze(0)

    batch = {
        "token_ids": t(token_ids, torch.long),
        "role_ids": t(role_ids, torch.long),
        "turn_distance": t(turn_distance, torch.long),
        "is_target": t(is_target, torch.long),
        "turn_mask": t(turn_mask, torch.bool),
        "face_features": t(face_features, torch.float32),
        "frame_mask": t(frame_mask, torch.bool),
        "phoneme_ids": t(phoneme_ids, torch.long),
        "phoneme_mask": t(phoneme_mask, torch.bool),
        "speaker_embedding": t(speaker_emb, torch.float32),
    }
    if "target_style_embedding_path" in sample:
        style_emb = np.load(sample["target_style_embedding_path"]).astype(np.float32)
        batch["target_style_embedding"] = t(style_emb, torch.float32)
    return batch


def main():
    args = parse_args()
    stage_a_cfg = load_config(args.stage_a_config)
    stage_b_cfg = load_config(args.stage_b_config)
    device = args.device

    model = build_react_tts(stage_a_cfg, stage_b_cfg).to(device)
    model.eval()

    if args.stage_a_ckpt:
        model.style_predictor.load_state_dict(torch.load(args.stage_a_ckpt, map_location=device))
    if args.stage_b_ckpt:
        ckpt = torch.load(args.stage_b_ckpt, map_location=device)
        model.tts_backbone.load_state_dict(ckpt.get("backbone", ckpt.get("tts_backbone", ckpt)))

    with open(args.input_json) as f:
        sample = json.load(f)
    batch = build_single_batch(sample, stage_a_cfg, stage_b_cfg, device)

    out = model.infer(batch, style_source=args.style_source, num_steps=args.num_steps)
    mel = out["mel"][0].cpu().numpy()

    import os

    os.makedirs(os.path.dirname(args.out_mel) or ".", exist_ok=True)
    np.save(args.out_mel, mel)
    print(f"Saved generated mel {mel.shape} to {args.out_mel}")
    if out["style_out"] is not None:
        emo_id = out["style_out"]["emotion_logits"].argmax(dim=-1).item()
        print(f"Predicted response emotion id: {emo_id}, VAD: {out['style_out']['vad'][0].tolist()}")
        print(f"Gate(listener reaction) = {out['style_out']['gate_react'].item():.3f}")


if __name__ == "__main__":
    main()
