from __future__ import annotations

import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from react_tts.data.meld_gradtts_dataset import (
    MELDGradTTSDataset,
    GradTTSBatchCollate,
)
from react_tts.tts.react_grad_tts import ReactGradTTS


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--train_manifest",
        default="../data/meld_gradtts/stage_b/train.jsonl",
    )
    p.add_argument(
        "--val_manifest",
        default="../data/meld_gradtts/stage_b/dev_seen.jsonl",
    )
    p.add_argument(
        "--speaker_table",
        default="../data/meld_gradtts/speaker_embeddings_train/speaker_table.json",
    )

    p.add_argument("--ckpt_dir", default="checkpoints/grad_stage_b")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--out_size", type=int, default=400)
    p.add_argument("--num_workers", type=int, default=2)

    # Smoke-test option.
    p.add_argument("--max_steps", type=int, default=None)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")

    return p.parse_args()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_dataset(manifest, speaker_table):
    return MELDGradTTSDataset(
        manifest,
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


def main():
    args = parse_args()
    seed_all(args.seed)

    device = torch.device(args.device)

    train_ds = make_dataset(
        args.train_manifest,
        args.speaker_table,
    )
    val_ds = make_dataset(
        args.val_manifest,
        args.speaker_table,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=GradTTSBatchCollate(),
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=GradTTSBatchCollate(),
        pin_memory=True,
    )

    model = ReactGradTTS(
        n_vocab=149,
        n_feats=128,
        speaker_emb_dim=256,
        style_emb_dim=256,
    ).to(device)

    # Stage B: GT emotion -> learned acoustic style embedding.
    emotion_embedding = nn.Embedding(
        7,
        256,
    ).to(device)

    params = (
        list(model.parameters())
        + list(emotion_embedding.parameters())
    )

    optimizer = torch.optim.AdamW(
        params,
        lr=args.lr,
    )

    os.makedirs(args.ckpt_dir, exist_ok=True)

    global_step = 0
    best_val = float("inf")

    print("Train samples:", len(train_ds))
    print("Val samples:", len(val_ds))
    print("Device:", device)

    for epoch in range(args.epochs):
        model.train()
        emotion_embedding.train()

        running = []

        for batch_idx, batch in enumerate(train_loader):
            x = batch["x"].to(device, non_blocking=True)
            x_lengths = batch["x_lengths"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            y_lengths = batch["y_lengths"].to(device, non_blocking=True)
            spk = batch["speaker_embedding"].to(device, non_blocking=True)
            emotion_id = batch["emotion_id"].to(device, non_blocking=True)

            style = emotion_embedding(emotion_id)

            out = model.compute_loss(
                x=x,
                x_lengths=x_lengths,
                y=y,
                y_lengths=y_lengths,
                speaker_embedding=spk,
                style_embedding=style,
                out_size=args.out_size,
            )

            loss = out["loss"]

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                params,
                1.0,
            )

            optimizer.step()

            running.append(loss.item())

            if global_step % 20 == 0:
                print(
                    f"epoch={epoch} "
                    f"step={global_step} "
                    f"loss={loss.item():.4f} "
                    f"dur={out['duration_loss'].item():.4f} "
                    f"prior={out['prior_loss'].item():.4f} "
                    f"diff={out['diffusion_loss'].item():.4f}"
                )

            global_step += 1

            if (
                args.max_steps is not None
                and global_step >= args.max_steps
            ):
                break

        # -------------------------
        # Validation
        # -------------------------
        model.eval()
        emotion_embedding.eval()

        val_losses = []

        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device)
                x_lengths = batch["x_lengths"].to(device)
                y = batch["y"].to(device)
                y_lengths = batch["y_lengths"].to(device)
                spk = batch["speaker_embedding"].to(device)
                emotion_id = batch["emotion_id"].to(device)

                style = emotion_embedding(emotion_id)

                out = model.compute_loss(
                    x=x,
                    x_lengths=x_lengths,
                    y=y,
                    y_lengths=y_lengths,
                    speaker_embedding=spk,
                    style_embedding=style,
                    out_size=args.out_size,
                )

                val_losses.append(
                    out["loss"].item()
                )

        val_loss = float(
            np.mean(val_losses)
        )

        train_loss = float(
            np.mean(running)
        )

        print(
            f"[epoch {epoch}] "
            f"train={train_loss:.4f} "
            f"val={val_loss:.4f}"
        )

        ckpt = {
            "model": model.state_dict(),
            "emotion_embedding": emotion_embedding.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "val_loss": val_loss,
        }

        torch.save(
            ckpt,
            os.path.join(
                args.ckpt_dir,
                "last.pt",
            ),
        )

        if val_loss < best_val:
            best_val = val_loss

            torch.save(
                ckpt,
                os.path.join(
                    args.ckpt_dir,
                    "best.pt",
                ),
            )

            print(
                f"New best val loss: {best_val:.4f}"
            )

        if (
            args.max_steps is not None
            and global_step >= args.max_steps
        ):
            print("Reached max_steps.")
            break


if __name__ == "__main__":
    main()
