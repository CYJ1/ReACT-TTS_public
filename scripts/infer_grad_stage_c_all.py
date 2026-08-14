from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

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
        default="checkpoints/grad_stage_c_seed43/best.pt",
    )
    p.add_argument(
        "--manifest",
        default="../data/meld_gradtts/stage_c_cached/test_seen.jsonl",
    )
    p.add_argument(
        "--speaker_table",
        default="../data/meld_gradtts/speaker_embeddings_train/speaker_table.json",
    )

    p.add_argument(
        "--output_dir",
        default="outputs/grad_stage_c_test252",
    )

    p.add_argument("--timesteps", type=int, default=50)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--length_scale", type=float, default=1.0)

    p.add_argument("--device", default="cuda:0")
    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument("--end_idx", type=int, default=None)

    return p.parse_args()


def load_stage_a(model, path, device):
    ckpt = torch.load(path, map_location=device)

    if isinstance(ckpt, dict) and "model" in ckpt:
        ckpt = ckpt["model"]

    model.load_state_dict(ckpt)


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


def get_generated_mel(out):
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

    return mel


def main():
    args = parse_args()
    device = torch.device(args.device)

    output_dir = Path(args.output_dir)
    wav_dir = output_dir / "wav"
    mel_dir = output_dir / "mel"
    meta_dir = output_dir / "meta"

    wav_dir.mkdir(parents=True, exist_ok=True)
    mel_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

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

    print(
        "Loaded Stage C:",
        args.stage_c_ckpt,
        "epoch=",
        ckpt.get("epoch"),
        "val=",
        ckpt.get("val_loss"),
    )

    # ---------------------------------------------------------
    # Vocoder
    # ---------------------------------------------------------
    print("Loading HiFi-GAN...")
    vocoder = torch.hub.load(
        "bshall/hifigan:main",
        "hifigan",
        trust_repo=True,
    ).eval().to(device)
    print("HiFi-GAN loaded.")

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------
    ds = make_dataset(
        args.manifest,
        cfg,
        args.speaker_table,
    )

    with open(args.manifest, "r") as f:
        rows = [
            json.loads(line)
            for line in f
            if line.strip()
        ]

    if len(ds) != len(rows):
        raise RuntimeError(
            f"Manifest mismatch: dataset={len(ds)} rows={len(rows)}"
        )

    start = max(0, args.start_idx)
    end = len(ds) if args.end_idx is None else min(args.end_idx, len(ds))

    print(f"Dataset size: {len(ds)}")
    print(f"Generating indices [{start}, {end})")
    print(f"Total requested: {end-start}")

    collate = GradTTSDialogueBatchCollate()

    success = 0
    skipped = 0
    failures = []

    # ---------------------------------------------------------
    # Inference loop
    # ---------------------------------------------------------
    for idx in range(start, end):

        meta = rows[idx]

        dia = meta.get("dialogue_id", "x")
        utt = meta.get(
            "target_utterance_id",
            meta.get("utterance_id", "x"),
        )

        stem = f"idx{idx:04d}_dia{dia}_utt{utt}"

        wav_path = wav_dir / f"{stem}.wav"
        mel_path = mel_dir / f"{stem}.npy"
        meta_path = meta_dir / f"{stem}.json"

        # Resume-friendly:
        if wav_path.exists() and mel_path.exists() and meta_path.exists():
            skipped += 1
            print(
                f"[{idx+1}/{end}] SKIP {stem}"
            )
            continue

        try:
            sample = ds[idx]
            batch = collate([sample])

            dialogue = move_dialogue(
                batch,
                device,
            )

            x = batch["x"].to(device)
            x_lengths = batch["x_lengths"].to(device)
            speaker = batch["speaker_embedding"].to(device)

            with torch.no_grad():
                planner_out = planner(dialogue)

                response_style = planner_out[
                    "style_embedding"
                ]

                acoustic_style = adapter(
                    response_style
                )

                out = acoustic(
                    x=x,
                    x_lengths=x_lengths,
                    speaker_embedding=speaker,
                    style_embedding=acoustic_style,
                    n_timesteps=args.timesteps,
                    temperature=args.temperature,
                    length_scale=args.length_scale,
                )

                mel = get_generated_mel(out)

                wav = (
                    vocoder.forward(mel)
                    .detach()
                    .cpu()
                    .squeeze()
                    .clamp(-1.0, 1.0)
                    .numpy()
                )

            np.save(
                mel_path,
                mel.detach().cpu().numpy(),
            )

            sf.write(
                wav_path,
                wav,
                16000,
            )

            generated_meta = {
                **meta,
                "sample_idx": idx,
                "stage_a_ckpt": args.stage_a_ckpt,
                "stage_c_ckpt": args.stage_c_ckpt,
                "timesteps": args.timesteps,
                "temperature": args.temperature,
                "length_scale": args.length_scale,
                "generated_wav": str(wav_path),
                "generated_mel": str(mel_path),
                "num_samples": int(len(wav)),
                "duration_sec": float(len(wav) / 16000.0),
            }

            with open(meta_path, "w") as f:
                json.dump(
                    generated_meta,
                    f,
                    indent=2,
                )

            success += 1

            print(
                f"[{idx+1}/{end}] OK "
                f"{stem} "
                f"{len(wav)/16000:.2f}s "
                f"| {meta.get('target_speaker_id')} "
                f"| {meta.get('target_emotion')} "
                f"| {meta.get('target_text')}"
            )

        except Exception as e:
            failures.append(
                {
                    "idx": idx,
                    "stem": stem,
                    "error": repr(e),
                }
            )

            print(
                f"[{idx+1}/{end}] FAIL "
                f"{stem}: {repr(e)}"
            )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------
    failure_path = output_dir / "failures.json"

    with open(failure_path, "w") as f:
        json.dump(
            failures,
            f,
            indent=2,
        )

    summary = {
        "manifest": args.manifest,
        "stage_a_ckpt": args.stage_a_ckpt,
        "stage_c_ckpt": args.stage_c_ckpt,
        "start_idx": start,
        "end_idx": end,
        "requested": end - start,
        "success": success,
        "skipped": skipped,
        "failures": len(failures),
        "timesteps": args.timesteps,
        "temperature": args.temperature,
        "length_scale": args.length_scale,
    }

    summary_path = output_dir / "summary.json"

    with open(summary_path, "w") as f:
        json.dump(
            summary,
            f,
            indent=2,
        )

    print()
    print("========== DONE ==========")
    print("Requested:", end - start)
    print("Generated:", success)
    print("Skipped:", skipped)
    print("Failures:", len(failures))
    print("Output:", output_dir)
    print("Summary:", summary_path)
    print("==========================")


if __name__ == "__main__":
    main()
