"""Stage B: pretrain the expressive TTS backbone with GROUND-TRUTH style
embeddings (a ground-truth emotion-label lookup table stands in for the
Stage A planner, which isn't connected yet -- this reproduces the
FEIM-TTS-style "emotion -> expressive speech" setting on e.g. CREMA-D).

    python scripts/train_stage_b.py --config configs/stage_b.yaml
"""
from __future__ import annotations

import argparse
import os

import torch
from torch.utils.data import DataLoader

from react_tts.config import load_config
from react_tts.data.emotion_labels import MELD_EMOTIONS
from react_tts.data.tts_dataset import TTSDataset
from react_tts.tts.backbone import TTSBackbone
from react_tts.tts.style_lookup import GroundTruthStyleEmbedding


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/stage_b.yaml")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    if args.epochs:
        cfg.train.epochs = args.epochs
    torch.manual_seed(cfg.get("seed", 42))
    device = args.device

    d = cfg.data
    train_ds = TTSDataset(
        d.get("manifest_train"), n_mels=d.n_mels, max_text_len=d.max_text_len, max_mel_len=d.max_mel_len,
        speaker_emb_dim=cfg.model.speaker_emb_dim, synthetic=d.synthetic, phoneme_vocab_size=cfg.model.phoneme_vocab_size,
    )
    val_ds = TTSDataset(
        d.get("manifest_val"), n_mels=d.n_mels, max_text_len=d.max_text_len, max_mel_len=d.max_mel_len,
        speaker_emb_dim=cfg.model.speaker_emb_dim, synthetic=d.synthetic, phoneme_vocab_size=cfg.model.phoneme_vocab_size,
    )
    train_loader = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.train.batch_size, shuffle=False)

    m = cfg.model
    backbone = TTSBackbone(
        phoneme_vocab_size=m.phoneme_vocab_size, n_mels=m.n_mels, speaker_emb_dim=m.speaker_emb_dim,
        style_emb_dim=m.style_emb_dim, encoder_cfg=m.encoder, duration_cfg=m.duration_predictor, decoder_cfg=m.decoder,
    ).to(device)
    style_lookup = GroundTruthStyleEmbedding(len(MELD_EMOTIONS), m.style_emb_dim).to(device)

    if cfg.train.get("pretrained_ckpt"):
        state = torch.load(cfg.train.pretrained_ckpt, map_location=device)
        backbone.load_state_dict(state, strict=False)
        print(f"Warm-started backbone from {cfg.train.pretrained_ckpt}")

    params = list(backbone.parameters()) + list(style_lookup.parameters())
    optimizer = torch.optim.AdamW(params, lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    os.makedirs(cfg.train.ckpt_dir, exist_ok=True)

    step = 0
    best_val = float("inf")
    for epoch in range(cfg.train.epochs):
        backbone.train()
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            style_emb = style_lookup(batch["emotion_id"])
            out = backbone.forward_train(
                phoneme_ids=batch["phoneme_ids"], phoneme_mask=batch["phoneme_mask"], durations=batch["durations"],
                target_mel=batch["target_mel"], mel_mask=batch["mel_mask"], speaker_emb=batch["speaker_embedding"],
                style_emb=style_emb,
            )
            loss = out["diffusion_loss"] + cfg.train.duration_loss_weight * out["duration_loss"]

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, cfg.train.grad_clip)
            optimizer.step()

            if step % cfg.train.log_every == 0:
                print(f"epoch {epoch} step {step} loss {loss.item():.4f} "
                      f"(diff {out['diffusion_loss'].item():.4f} dur {out['duration_loss'].item():.4f})")
            step += 1

        backbone.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                style_emb = style_lookup(batch["emotion_id"])
                out = backbone.forward_train(
                    phoneme_ids=batch["phoneme_ids"], phoneme_mask=batch["phoneme_mask"], durations=batch["durations"],
                    target_mel=batch["target_mel"], mel_mask=batch["mel_mask"], speaker_emb=batch["speaker_embedding"],
                    style_emb=style_emb,
                )
                val_losses.append((out["diffusion_loss"] + cfg.train.duration_loss_weight * out["duration_loss"]).item())
        val_loss = sum(val_losses) / max(len(val_losses), 1)
        print(f"[epoch {epoch}] val loss {val_loss:.4f}")

        ckpt = {"backbone": backbone.state_dict(), "style_lookup": style_lookup.state_dict()}
        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, os.path.join(cfg.train.ckpt_dir, "best.pt"))
        torch.save(ckpt, os.path.join(cfg.train.ckpt_dir, "last.pt"))


if __name__ == "__main__":
    main()
