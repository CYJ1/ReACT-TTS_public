from __future__ import annotations

import argparse
import os
import random
import warnings

warnings.filterwarnings(
    "ignore",
    message="Converting mask without torch.bool dtype to bool.*",
    category=UserWarning,
)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from react_tts.config import load_config
from react_tts.models.response_style_predictor import (
    build_response_style_predictor,
)
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
        "--stage_b_ckpt",
        default="checkpoints/grad_stage_b_seed42/best.pt",
    )

    p.add_argument(
        "--train_manifest",
        default="../data/meld_gradtts/stage_c_cached/train.jsonl",
    )
    p.add_argument(
        "--val_manifest",
        default="../data/meld_gradtts/stage_c_cached/dev_seen.jsonl",
    )
    p.add_argument(
        "--speaker_table",
        default="../data/meld_gradtts/speaker_embeddings_train/speaker_table.json",
    )

    p.add_argument(
        "--ckpt_dir",
        default="checkpoints/grad_stage_c",
    )

    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=2)

    p.add_argument(
        "--lr",
        type=float,
        default=2e-5,
    )
    p.add_argument(
        "--adapter_lr",
        type=float,
        default=1e-4,
    )

    p.add_argument(
        "--style_align_weight",
        type=float,
        default=0.5,
    )

    p.add_argument(
        "--out_size",
        type=int,
        default=400,
    )

    p.add_argument(
        "--max_steps",
        type=int,
        default=None,
    )

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")

    return p.parse_args()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_state(module, path, device):
    state = torch.load(
        path,
        map_location=device,
    )

    # Stage-A checkpoints may be raw state_dict or wrapped.
    if isinstance(state, dict) and "model" in state:
        state = state["model"]

    module.load_state_dict(state)


def make_dataset(
    manifest,
    stage_a_cfg,
    speaker_table,
):
    dialogue_kwargs = dict(
        tokenizer_spec=stage_a_cfg.data.tokenizer,
        vocab_size=stage_a_cfg.model.text.vocab_size,
        num_context_turns=stage_a_cfg.data.num_context_turns,
        max_text_len=stage_a_cfg.data.max_text_len,
        reaction_num_frames=stage_a_cfg.data.reaction_num_frames,
        face_feat_dim=stage_a_cfg.model.reaction.face_feat_dim,
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
    seed_all(args.seed)

    device = torch.device(args.device)

    # ---------------------------------------------------------
    # Stage A: frozen response planner
    # ---------------------------------------------------------
    stage_a_cfg = load_config(
        args.stage_a_config
    )

    planner = build_response_style_predictor(
        stage_a_cfg
    ).to(device)

    load_state(
        planner,
        args.stage_a_ckpt,
        device,
    )

    planner.eval()

    for p in planner.parameters():
        p.requires_grad = False

    print(
        "Loaded frozen Stage A:",
        args.stage_a_ckpt,
    )

    # ---------------------------------------------------------
    # Stage B acoustic model + learned GT emotion style table
    # ---------------------------------------------------------
    acoustic = ReactGradTTS(
        n_vocab=149,
        n_feats=128,
        speaker_emb_dim=256,
        style_emb_dim=256,
    ).to(device)

    teacher_style = nn.Embedding(
        7,
        256,
    ).to(device)

    b_ckpt = torch.load(
        args.stage_b_ckpt,
        map_location=device,
    )

    acoustic.load_state_dict(
        b_ckpt["model"]
    )

    teacher_style.load_state_dict(
        b_ckpt["emotion_embedding"]
    )

    # Teacher emotion space is fixed.
    teacher_style.eval()

    for p in teacher_style.parameters():
        p.requires_grad = False

    print(
        "Loaded Stage B:",
        args.stage_b_ckpt,
        "epoch=",
        b_ckpt.get("epoch"),
        "val_loss=",
        b_ckpt.get("val_loss"),
    )

    # ---------------------------------------------------------
    # Stage A -> acoustic style adapter
    # ---------------------------------------------------------
    adapter = ResponseStyleAdapter(
        input_dim=256,
        output_dim=256,
        hidden_dim=256,
        dropout=0.1,
    ).to(device)

    # ---------------------------------------------------------
    # Data
    # ---------------------------------------------------------
    train_ds = make_dataset(
        args.train_manifest,
        stage_a_cfg,
        args.speaker_table,
    )

    val_ds = make_dataset(
        args.val_manifest,
        stage_a_cfg,
        args.speaker_table,
    )

    collate = GradTTSDialogueBatchCollate()

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=True,
    )

    # Acoustic model gets small LR; adapter learns faster.
    optimizer = torch.optim.AdamW(
        [
            {
                "params": acoustic.parameters(),
                "lr": args.lr,
            },
            {
                "params": adapter.parameters(),
                "lr": args.adapter_lr,
            },
        ],
        weight_decay=0.0,
    )

    os.makedirs(
        args.ckpt_dir,
        exist_ok=True,
    )

    best_val = float("inf")
    global_step = 0

    print("Stage C Train:", len(train_ds))
    print("Stage C Val:", len(val_ds))
    print("Device:", device)

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------
    for epoch in range(args.epochs):

        acoustic.train()
        adapter.train()
        planner.eval()

        train_losses = []

        for batch in train_loader:

            dialogue = move_dialogue(
                batch,
                device,
            )

            # Frozen Stage A.
            with torch.no_grad():
                style_out = planner(dialogue)

                response_emb = (
                    style_out["style_embedding"]
                )

                teacher = teacher_style(
                    batch["emotion_id"].to(device)
                )

            predicted_style = adapter(
                response_emb
            )

            # Align response representation with the acoustic
            # style space learned during Stage B.
            mse_align = F.mse_loss(
                predicted_style,
                teacher,
            )

            cosine_align = (
                1.0
                - F.cosine_similarity(
                    predicted_style,
                    teacher,
                    dim=-1,
                ).mean()
            )

            style_align = (
                0.5 * mse_align
                + 0.5 * cosine_align
            )

            out = acoustic.compute_loss(
                x=batch["x"].to(device),
                x_lengths=batch["x_lengths"].to(device),
                y=batch["y"].to(device),
                y_lengths=batch["y_lengths"].to(device),
                speaker_embedding=batch[
                    "speaker_embedding"
                ].to(device),
                style_embedding=predicted_style,
                out_size=args.out_size,
            )

            acoustic_loss = out["loss"]

            total = (
                acoustic_loss
                + args.style_align_weight
                * style_align
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            total.backward()

            torch.nn.utils.clip_grad_norm_(
                acoustic.parameters(),
                1.0,
            )

            torch.nn.utils.clip_grad_norm_(
                adapter.parameters(),
                1.0,
            )

            optimizer.step()

            train_losses.append(
                total.item()
            )

            if global_step % 20 == 0:
                print(
                    f"epoch={epoch} "
                    f"step={global_step} "
                    f"total={total.item():.4f} "
                    f"tts={acoustic_loss.item():.4f} "
                    f"align={style_align.item():.4f} "
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

        # -----------------------------------------------------
        # Validation
        # -----------------------------------------------------
        acoustic.eval()
        adapter.eval()
        planner.eval()

        val_losses = []

        with torch.no_grad():
            for batch in val_loader:

                dialogue = move_dialogue(
                    batch,
                    device,
                )

                style_out = planner(
                    dialogue
                )

                response_emb = (
                    style_out["style_embedding"]
                )

                predicted_style = adapter(
                    response_emb
                )

                teacher = teacher_style(
                    batch["emotion_id"].to(device)
                )

                mse_align = F.mse_loss(
                    predicted_style,
                    teacher,
                )

                cosine_align = (
                    1.0
                    - F.cosine_similarity(
                        predicted_style,
                        teacher,
                        dim=-1,
                    ).mean()
                )

                style_align = (
                    0.5 * mse_align
                    + 0.5 * cosine_align
                )

                out = acoustic.compute_loss(
                    x=batch["x"].to(device),
                    x_lengths=batch["x_lengths"].to(device),
                    y=batch["y"].to(device),
                    y_lengths=batch["y_lengths"].to(device),
                    speaker_embedding=batch[
                        "speaker_embedding"
                    ].to(device),
                    style_embedding=predicted_style,
                    out_size=args.out_size,
                )

                total = (
                    out["loss"]
                    + args.style_align_weight
                    * style_align
                )

                val_losses.append(
                    total.item()
                )

        train_loss = float(
            np.mean(train_losses)
        )

        val_loss = float(
            np.mean(val_losses)
        )

        print(
            f"[epoch {epoch}] "
            f"train={train_loss:.4f} "
            f"val={val_loss:.4f}"
        )

        ckpt = {
            "acoustic": acoustic.state_dict(),
            "adapter": adapter.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "val_loss": val_loss,
            "stage_a_ckpt": args.stage_a_ckpt,
            "stage_b_ckpt": args.stage_b_ckpt,
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
                f"New best Stage C val: "
                f"{best_val:.4f}"
            )

        if (
            args.max_steps is not None
            and global_step >= args.max_steps
        ):
            print(
                "Reached max_steps."
            )
            break


if __name__ == "__main__":
    main()
