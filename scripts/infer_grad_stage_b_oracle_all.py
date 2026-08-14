from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn

from react_tts.config import load_config
from react_tts.data.meld_gradtts_dataset import (
    MELDGradTTSDialogueDataset,
    GradTTSDialogueBatchCollate,
)
from react_tts.tts.react_grad_tts import ReactGradTTS


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--stage_a_config",
        default="configs/stage_a_no_delta_final.yaml",
        help="Used only for dataset/dialogue preprocessing compatibility.",
    )
    p.add_argument(
        "--stage_b_ckpt",
        default="checkpoints/grad_stage_b_seed42/best.pt",
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
        default="outputs/grad_stage_b_oracle_test252",
    )

    p.add_argument("--timesteps", type=int, default=50)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--length_scale", type=float, default=1.0)

    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument("--end_idx", type=int, default=None)
    p.add_argument("--device", default="cuda:0")

    return p.parse_args()


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

    out_root = Path(args.output_dir)
    wav_dir = out_root / "wav"
    mel_dir = out_root / "mel"
    meta_dir = out_root / "meta"

    wav_dir.mkdir(parents=True, exist_ok=True)
    mel_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(args.stage_a_config)

    # ---------------------------------------------------------
    # Stage-B acoustic model
    # ---------------------------------------------------------
    acoustic = ReactGradTTS(
        n_vocab=149,
        n_feats=128,
        speaker_emb_dim=256,
        style_emb_dim=256,
    ).to(device)

    emotion_embedding = nn.Embedding(
        7,
        256,
    ).to(device)

    ckpt = torch.load(
        args.stage_b_ckpt,
        map_location=device,
    )

    acoustic.load_state_dict(
        ckpt["model"]
    )

    emotion_embedding.load_state_dict(
        ckpt["emotion_embedding"]
    )

    acoustic.eval()
    emotion_embedding.eval()

    print(
        "Loaded Stage B:",
        args.stage_b_ckpt,
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

    with open(args.manifest) as f:
        rows = [
            json.loads(line)
            for line in f
            if line.strip()
        ]

    if len(ds) != len(rows):
        raise RuntimeError(
            f"Dataset/manifest mismatch: "
            f"{len(ds)} vs {len(rows)}"
        )

    start = max(0, args.start_idx)
    end = (
        len(ds)
        if args.end_idx is None
        else min(args.end_idx, len(ds))
    )

    collate = GradTTSDialogueBatchCollate()

    print("Dataset size:", len(ds))
    print(f"Generating [{start}, {end})")

    success = 0
    skipped = 0
    failures = []

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

        if (
            wav_path.exists()
            and mel_path.exists()
            and meta_path.exists()
        ):
            skipped += 1
            print(f"[{idx+1}/{end}] SKIP {stem}")
            continue

        try:
            sample = ds[idx]
            batch = collate([sample])

            x = batch["x"].to(device)
            x_lengths = batch["x_lengths"].to(device)
            speaker = batch["speaker_embedding"].to(device)
            emotion_id = batch["emotion_id"].to(device)

            with torch.no_grad():
                style = emotion_embedding(
                    emotion_id
                )

                out = acoustic(
                    x=x,
                    x_lengths=x_lengths,
                    speaker_embedding=speaker,
                    style_embedding=style,
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

            out_meta = {
                **meta,
                "sample_idx": idx,
                "stage_b_ckpt": args.stage_b_ckpt,
                "conditioning": "ground_truth_emotion",
                "timesteps": args.timesteps,
                "temperature": args.temperature,
                "length_scale": args.length_scale,
                "generated_wav": str(wav_path),
                "generated_mel": str(mel_path),
                "duration_sec": float(len(wav) / 16000.0),
            }

            with open(meta_path, "w") as f:
                json.dump(
                    out_meta,
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

    print()
    print("========== DONE ==========")
    print("Success:", success)
    print("Skipped:", skipped)
    print("Failures:", len(failures))

    with open(
        out_root / f"failures_{start}_{end}.json",
        "w",
    ) as f:
        json.dump(
            failures,
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
