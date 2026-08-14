from __future__ import annotations

import argparse
import json
import os

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader

from react_tts.config import load_config
from react_tts.models.response_style_predictor import build_response_style_predictor
from react_tts.data.meld_gradtts_dataset import (
    MELDGradTTSDialogueDataset,
    GradTTSDialogueBatchCollate,
)
from react_tts.tts.react_grad_tts import ReactGradTTS
from react_tts.tts.style_adapter import ResponseStyleAdapter


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--stage_a_config",
        default="configs/stage_a_no_delta_final.yaml",
    )
    p.add_argument(
        "--stage_a_ckpt",
        default="checkpoints/stage_a_no_delta_final/best.pt",
    )
    p.add_argument(
        "--stage_c_ckpt",
        required=True,
    )
    p.add_argument(
        "--manifest",
        default="../data/meld_gradtts/stage_c_cached/test_seen.jsonl",
    )
    p.add_argument(
        "--speaker_table",
        default="../data/meld_gradtts/speaker_embeddings_train/speaker_table.json",
    )

    p.add_argument("--sample_idx", type=int, default=0)
    p.add_argument("--timesteps", type=int, default=50)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--length_scale", type=float, default=1.0)

    p.add_argument(
        "--output_dir",
        default="outputs/grad_stage_c",
    )
    p.add_argument("--device", default="cuda:0")

    return p.parse_args()


def load_stage_a(planner, path, device):
    ckpt = torch.load(path, map_location=device)

    if isinstance(ckpt, dict) and "model" in ckpt:
        ckpt = ckpt["model"]

    planner.load_state_dict(ckpt)


def make_dataset(manifest, cfg, speaker_table):
    dialogue_kwargs = dict(
        tokenizer_spec=cfg.data.tokenizer,
        vocab_size=cfg.model.text.vocab_size,
        num_context_turns=cfg.data.num_context_turns,
        max_text_len=cfg.data.max_text_len,
        reaction_num_frames=cfg.data.reaction_num_frames,
        face_feat_dim=cfg.model.reaction.face_feat_dim,
    )

    acoustic_kwargs = dict(
        sample_rate=16000,
        n_fft=1024,
        n_mels=128,
        hop_length=160,
        win_length=1024,
        f_min=0.0,
        f_max=8000.0,
        add_blank=True,
        speaker_table_path=speaker_table,
    )

    return MELDGradTTSDialogueDataset(
        manifest,
        dialogue_kwargs=dialogue_kwargs,
        acoustic_kwargs=acoustic_kwargs,
    )


def move_dialogue(batch, device):
    return {
        "token_ids": batch["token_ids"].to(device),
        "role_ids": batch["role_ids"].to(device),
        "turn_distance": batch["turn_distance"].to(device),
        "is_target": batch["is_target"].to(device),
        "turn_mask": batch["turn_mask"].to(device),
        "face_features": batch["face_features"].to(device),
        "frame_mask": batch["frame_mask"].to(device),
    }


def main():
    args = parse_args()
    device = torch.device(args.device)

    os.makedirs(args.output_dir, exist_ok=True)

    cfg = load_config(args.stage_a_config)

    # ---------------------------------------------------------
    # Stage A planner
    # ---------------------------------------------------------
    planner = build_response_style_predictor(cfg).to(device)
    load_stage_a(planner, args.stage_a_ckpt, device)

    planner.eval()
    for p in planner.parameters():
        p.requires_grad = False

    # ---------------------------------------------------------
    # Stage C acoustic + adapter
    # ---------------------------------------------------------
    acoustic = ReactGradTTS(
        n_vocab=149,
        n_feats=128,
        speaker_emb_dim=256,
        style_emb_dim=256,
    ).to(device)

    adapter = ResponseStyleAdapter(
        input_dim=256,
        output_dim=256,
        hidden_dim=256,
        dropout=0.1,
    ).to(device)

    ckpt = torch.load(
        args.stage_c_ckpt,
        map_location=device,
    )

    acoustic.load_state_dict(ckpt["acoustic"])
    adapter.load_state_dict(ckpt["adapter"])

    acoustic.eval()
    adapter.eval()

    # ---------------------------------------------------------
    # Dataset/sample
    # ---------------------------------------------------------
    ds = make_dataset(
        args.manifest,
        cfg,
        args.speaker_table,
    )

    if args.sample_idx < 0 or args.sample_idx >= len(ds):
        raise IndexError(
            f"sample_idx={args.sample_idx}, dataset size={len(ds)}"
        )

    sample = ds[args.sample_idx]

    collate = GradTTSDialogueBatchCollate()
    batch = collate([sample])

    dialogue = move_dialogue(batch, device)

    x = batch["x"].to(device)
    x_lengths = batch["x_lengths"].to(device)
    speaker = batch["speaker_embedding"].to(device)

    # ---------------------------------------------------------
    # Response planning
    # ---------------------------------------------------------
    with torch.no_grad():
        planner_out = planner(dialogue)
        response_style = planner_out["style_embedding"]

        acoustic_style = adapter(response_style)

        # Grad-TTS inference.
        out = acoustic(
            x=x,
            x_lengths=x_lengths,
            speaker_embedding=speaker,
            style_embedding=acoustic_style,
            n_timesteps=args.timesteps,
            temperature=args.temperature,
            length_scale=args.length_scale,
        )

    # Handle either tensor output or dictionary output defensively.
    if isinstance(out, dict):
        if "mel" in out:
            mel = out["mel"]
        elif "y_dec" in out:
            mel = out["y_dec"]
        else:
            raise KeyError(
                f"Cannot find generated mel in output keys: {list(out.keys())}"
            )
    else:
        mel = out

    if isinstance(mel, (list, tuple)):
        mel = mel[-1]

    if mel.dim() == 3:
        mel = mel[0]

    print("Generated mel:", tuple(mel.shape))

    # ---------------------------------------------------------
    # HiFi-GAN 16k
    # ---------------------------------------------------------
    vocoder = torch.hub.load(
        "bshall/hifigan:main",
        "hifigan",
        trust_repo=True,
    ).eval().to(device)

    with torch.no_grad():
        wav = (
            vocoder.forward(mel)
            .detach()
            .cpu()
            .squeeze()
            .clamp(-1.0, 1.0)
            .numpy()
        )

    # ---------------------------------------------------------
    # Metadata
    # ---------------------------------------------------------
    with open(args.manifest, "r") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    meta = rows[args.sample_idx]

    stem = (
        f"idx{args.sample_idx:04d}_"
        f"dia{meta.get('dialogue_id', 'x')}_"
        f"utt{meta.get('target_utterance_id', meta.get('utterance_id', 'x'))}"
    )

    wav_path = os.path.join(
        args.output_dir,
        stem + ".wav",
    )
    mel_path = os.path.join(
        args.output_dir,
        stem + ".npy",
    )
    meta_path = os.path.join(
        args.output_dir,
        stem + ".json",
    )

    sf.write(
        wav_path,
        wav,
        16000,
    )

    np.save(
        mel_path,
        mel.detach().cpu().numpy(),
    )

    with open(meta_path, "w") as f:
        json.dump(
            {
                **meta,
                "sample_idx": args.sample_idx,
                "stage_c_ckpt": args.stage_c_ckpt,
                "timesteps": args.timesteps,
                "temperature": args.temperature,
                "length_scale": args.length_scale,
                "generated_wav": wav_path,
                "generated_mel": mel_path,
            },
            f,
            indent=2,
        )

    print("WAV:", wav_path)
    print("MEL:", mel_path)
    print("META:", meta_path)
    print("Text:", meta.get("target_text"))
    print("Speaker:", meta.get("target_speaker_id"))
    print("Emotion:", meta.get("target_emotion"))


if __name__ == "__main__":
    main()
