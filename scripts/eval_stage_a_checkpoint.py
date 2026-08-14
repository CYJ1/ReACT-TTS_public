from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from react_tts.config import load_config
from react_tts.models.response_style_predictor import build_response_style_predictor
from scripts.train_stage_a import build_datasets, evaluate


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    # Same architecture flags used during training.
    p.add_argument("--no_listener_face", action="store_true")
    p.add_argument("--static_face", action="store_true")
    p.add_argument("--no_reaction_delta", action="store_true")

    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    if args.no_listener_face:
        cfg.model.reaction.use_listener_face = False

    if args.static_face:
        cfg.model.reaction.temporal_face = False

    if args.no_reaction_delta:
        cfg.model.reaction.use_reaction_delta = False

    _, val_ds = build_datasets(cfg)

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
    )

    model = build_response_style_predictor(cfg).to(args.device)

    state = torch.load(args.ckpt, map_location=args.device)
    model.load_state_dict(state)

    metrics = evaluate(
        model,
        val_loader,
        args.device,
        cfg.data.num_emotion_classes,
    )

    print("[checkpoint]", args.ckpt)
    print("[num_val]", len(val_ds))
    print("[metrics]", metrics)


if __name__ == "__main__":
    main()