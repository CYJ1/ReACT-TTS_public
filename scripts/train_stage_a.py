"""Train (or evaluate) the Stage A response style predictor.

Every ablation in the paper's Table is a CLI flag here, flipping the same
config that `ResponseStylePredictor` reads -- so all rows share the same
architecture/parameter budget:

    python scripts/train_stage_a.py --config configs/stage_a.yaml                       # full model (RQ1/RQ2)
    python scripts/train_stage_a.py --config configs/stage_a.yaml --no_listener_face     # text-only baseline
    python scripts/train_stage_a.py --config configs/stage_a.yaml --static_face          # static-listener-face baseline
    python scripts/train_stage_a.py --config configs/stage_a.yaml --no_reaction_delta    # RQ2 ablation
    python scripts/train_stage_a.py --config configs/stage_a.yaml --random_listener      # modality-neglect control
    python scripts/train_stage_a.py --config configs/stage_a.yaml --mode mirror          # RQ3: naive emotion-mirroring baseline (no training)
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from eval.metrics import emotion_accuracy_f1, prosody_metrics, vad_ccc
from react_tts.config import load_config
from react_tts.data.dialogue_dataset import DialogueStyleDataset
from react_tts.losses import emotion_loss, prosody_loss, vad_loss
from react_tts.models.response_style_predictor import build_response_style_predictor


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/stage_a.yaml")
    p.add_argument("--mode", choices=["plan", "mirror"], default="plan")
    p.add_argument("--no_listener_face", action="store_true")
    p.add_argument("--static_face", action="store_true")
    p.add_argument("--no_reaction_delta", action="store_true")
    p.add_argument("--random_listener", action="store_true")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def build_datasets(cfg):
    d = cfg.data
    common = dict(
        tokenizer_spec=d.tokenizer,
        vocab_size=cfg.model.text.vocab_size,
        num_context_turns=d.num_context_turns,
        max_text_len=d.max_text_len,
        reaction_num_frames=d.reaction_num_frames,
        face_feat_dim=cfg.model.reaction.face_feat_dim,
        synthetic=d.synthetic,
    )
    train_ds = DialogueStyleDataset(d.get("manifest_train"), **common)
    val_ds = DialogueStyleDataset(d.get("manifest_val"), **common)
    return train_ds, val_ds


def evaluate(model, loader, device, num_emotion_classes, mirror_mode: bool = False):
    model.eval()
    all_emo_pred, all_emo_true = [], []
    all_vad_pred, all_vad_true = [], []
    all_prosody_pred, all_prosody_true = [], []
    gate_react_values = []

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            if mirror_mode:
                mirrored = batch["listener_last_emotion_id"]
                valid = mirrored >= 0
                if valid.sum() == 0:
                    continue
                all_emo_pred.append(mirrored[valid].cpu().numpy())
                all_emo_true.append(batch["emotion_id"][valid].cpu().numpy())
                continue

            out = model(batch)
            all_emo_pred.append(out["emotion_logits"].argmax(dim=-1).cpu().numpy())
            all_emo_true.append(batch["emotion_id"].cpu().numpy())
            all_vad_pred.append(out["vad"].cpu().numpy())
            all_vad_true.append(batch["vad"].cpu().numpy())
            all_prosody_pred.append(out["prosody"].cpu().numpy())
            all_prosody_true.append(batch["prosody"].cpu().numpy())
            gate_react_values.append(out["gate_react"].cpu().numpy())

    metrics = emotion_accuracy_f1(np.concatenate(all_emo_pred), np.concatenate(all_emo_true), num_emotion_classes)
    if not mirror_mode:
        metrics.update(vad_ccc(np.concatenate(all_vad_pred), np.concatenate(all_vad_true)))
        metrics.update(prosody_metrics(np.concatenate(all_prosody_pred), np.concatenate(all_prosody_true)))
        metrics["mean_gate_react"] = float(np.concatenate(gate_react_values).mean())
    return metrics


def main():
    args = parse_args()
    cfg = load_config(args.config)

    if args.no_listener_face:
        cfg.model.reaction.use_listener_face = False
    if args.static_face:
        cfg.model.reaction.temporal_face = False
    if args.no_reaction_delta:
        cfg.model.reaction.use_reaction_delta = False
    if args.random_listener:
        cfg.model.reaction.random_listener = True
    if args.epochs:
        cfg.train.epochs = args.epochs

    torch.manual_seed(cfg.get("seed", 42))
    train_ds, val_ds = build_datasets(cfg)

    num_classes = cfg.data.num_emotion_classes

    train_emotion_ids = torch.tensor(
        [
            int(train_ds[i]["emotion_id"])
            for i in range(len(train_ds))
        ],
        dtype=torch.long,
    )

    class_counts = torch.bincount(
        train_emotion_ids,
        minlength=num_classes,
    ).float()

    if torch.any(class_counts == 0):
        raise RuntimeError(
            f"Missing emotion class in training data: {class_counts.tolist()}"
        )

    total = class_counts.sum()

    class_weights = torch.sqrt(
        total / (num_classes * class_counts)
    )

    # Normalize to mean 1 so the overall CE scale remains
    # comparable with the previous training setup.
    class_weights = (
        class_weights / class_weights.mean()
    ).to(args.device)

    print(
        "[emotion class counts]",
        class_counts.tolist(),
        flush=True,
    )
    print(
        "[emotion class weights]",
        class_weights.detach().cpu().tolist(),
        flush=True,
    )

    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False)

    device = args.device
    model = build_response_style_predictor(cfg).to(device)

    if args.mode == "mirror":
        metrics = evaluate(model, val_loader, device, cfg.data.num_emotion_classes, mirror_mode=True)
        print("[emotion-mirroring baseline, RQ3]", metrics)
        return

    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    os.makedirs(cfg.train.ckpt_dir, exist_ok=True)
    best_f1 = -1.0
    step = 0
    for epoch in range(cfg.train.epochs):
        model.train()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch)

            l_emo = emotion_loss(out["emotion_logits"], batch["emotion_id"], class_weights=class_weights,)
            l_vad = vad_loss(out["vad"], batch["vad"])
            l_prosody = prosody_loss(out["prosody"], batch["prosody"])
            loss = (
                cfg.train.emotion_loss_weight * l_emo
                + cfg.train.vad_loss_weight * l_vad
                + cfg.train.prosody_loss_weight * l_prosody
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            optimizer.step()

            if step % cfg.train.log_every == 0:
                print(f"epoch {epoch} step {step} loss {loss.item():.4f} "
                      f"(emo {l_emo.item():.4f} vad {l_vad.item():.4f} prosody {l_prosody.item():.4f})")
            step += 1

        metrics = evaluate(model, val_loader, device, cfg.data.num_emotion_classes)
        print(f"[epoch {epoch}] val:", metrics)

        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            torch.save(model.state_dict(), os.path.join(cfg.train.ckpt_dir, "best.pt"))
        torch.save(model.state_dict(), os.path.join(cfg.train.ckpt_dir, "last.pt"))


if __name__ == "__main__":
    main()
