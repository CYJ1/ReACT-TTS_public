from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from react_tts.config import load_config
from react_tts.data.dialogue_dataset import DialogueStyleDataset
from react_tts.models.response_style_predictor import build_response_style_predictor


EMOTIONS = [
    "anger",
    "disgust",
    "fear",
    "joy",
    "neutral",
    "sadness",
    "surprise",
]


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--face_config", required=True)
    p.add_argument("--face_ckpt", required=True)

    p.add_argument("--text_config", required=True)
    p.add_argument("--text_ckpt", required=True)

    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    p.add_argument("--out_csv", required=True)

    return p.parse_args()


def build_val_dataset(cfg):
    d = cfg.data

    return DialogueStyleDataset(
        d.get("manifest_val"),
        tokenizer_spec=d.tokenizer,
        vocab_size=cfg.model.text.vocab_size,
        num_context_turns=d.num_context_turns,
        max_text_len=d.max_text_len,
        reaction_num_frames=d.reaction_num_frames,
        face_feat_dim=cfg.model.reaction.face_feat_dim,
        synthetic=d.synthetic,
    )


@torch.no_grad()
def predict(model, loader, device):
    model.eval()

    preds = []
    probs = []
    gates = []
    truths = []

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}

        out = model(batch)

        p = torch.softmax(out["emotion_logits"], dim=-1)

        preds.append(p.argmax(dim=-1).cpu().numpy())
        probs.append(p.cpu().numpy())
        truths.append(batch["emotion_id"].cpu().numpy())

        if "gate_react" in out:
            g = out["gate_react"].detach().cpu().numpy()

            # collapse all non-batch dims to one scalar per sample
            if g.ndim > 1:
                g = g.reshape(g.shape[0], -1).mean(axis=1)

            gates.append(g)

    preds = np.concatenate(preds)
    probs = np.concatenate(probs)
    truths = np.concatenate(truths)

    if gates:
        gates = np.concatenate(gates)
    else:
        gates = np.full(len(preds), np.nan)

    return {
        "pred": preds,
        "prob": probs,
        "truth": truths,
        "gate": gates,
    }


def load_model(cfg, ckpt, device):
    model = build_response_style_predictor(cfg).to(device)

    state = torch.load(ckpt, map_location=device)

    # current checkpoints are raw state_dict, but keep compatibility
    if isinstance(state, dict) and "model" in state:
        state = state["model"]

    model.load_state_dict(state)
    model.eval()

    return model


def safe_metadata(sample, idx):
    context = sample.get("context_turns", [])

    return {
        "sample_index": idx,
        "dialogue_id": sample.get("dialogue_id", ""),
        "utterance_id": sample.get("utterance_id", ""),
        "target_speaker_id": sample.get("target_speaker_id", ""),
        "target_text": sample.get("target_text", ""),
        "context_text": " || ".join(
            str(x.get("text", "")) for x in context
        ),
        "context_speakers": " || ".join(
            str(x.get("speaker_id", "")) for x in context
        ),
        "listener_reaction_path": sample.get("listener_reaction_path", ""),
    }


def main():
    args = parse_args()

    device = args.device

    # ---------------------------------------------------------
    # FACE / TEMPORAL NO-DELTA
    # ---------------------------------------------------------
    face_cfg = load_config(args.face_config)
    face_cfg.model.reaction.use_reaction_delta = False

    face_ds = build_val_dataset(face_cfg)

    face_loader = DataLoader(
        face_ds,
        batch_size=args.batch_size,
        shuffle=False,
    )

    face_model = load_model(
        face_cfg,
        args.face_ckpt,
        device,
    )

    face = predict(
        face_model,
        face_loader,
        device,
    )

    # ---------------------------------------------------------
    # TEXT ONLY
    # ---------------------------------------------------------
    text_cfg = load_config(args.text_config)
    text_cfg.model.reaction.use_listener_face = False

    text_ds = build_val_dataset(text_cfg)

    text_loader = DataLoader(
        text_ds,
        batch_size=args.batch_size,
        shuffle=False,
    )

    text_model = load_model(
        text_cfg,
        args.text_ckpt,
        device,
    )

    text = predict(
        text_model,
        text_loader,
        device,
    )

    # ---------------------------------------------------------
    # SANITY CHECKS
    # ---------------------------------------------------------
    assert len(face["truth"]) == len(text["truth"])
    assert len(face["truth"]) == len(face_ds)
    assert np.array_equal(face["truth"], text["truth"])

    n = len(face["truth"])

    rows = []

    counts = {
        "face_rescue": 0,
        "face_hurt": 0,
        "both_correct": 0,
        "both_wrong": 0,
    }

    print("\n========== SAMPLE-LEVEL COMPLEMENTARITY ==========")

    for i in range(n):
        y = int(face["truth"][i])

        pf = int(face["pred"][i])
        pt = int(text["pred"][i])

        face_correct = (pf == y)
        text_correct = (pt == y)

        if face_correct and not text_correct:
            category = "face_rescue"
        elif text_correct and not face_correct:
            category = "face_hurt"
        elif face_correct and text_correct:
            category = "both_correct"
        else:
            category = "both_wrong"

        counts[category] += 1

        sample = face_ds.samples[i]
        meta = safe_metadata(sample, i)

        row = {
            **meta,
            "true_id": y,
            "true_emotion": EMOTIONS[y],

            "face_pred_id": pf,
            "face_pred_emotion": EMOTIONS[pf],

            "text_pred_id": pt,
            "text_pred_emotion": EMOTIONS[pt],

            "face_correct": int(face_correct),
            "text_correct": int(text_correct),

            "category": category,

            "face_true_prob": float(face["prob"][i, y]),
            "text_true_prob": float(text["prob"][i, y]),

            "true_prob_delta_face_minus_text":
                float(face["prob"][i, y] - text["prob"][i, y]),

            "face_gate_react": float(face["gate"][i]),
        }

        rows.append(row)

    # ---------------------------------------------------------
    # BASIC SUMMARY
    # ---------------------------------------------------------
    print(f"N = {n}")

    for k in [
        "face_rescue",
        "face_hurt",
        "both_correct",
        "both_wrong",
    ]:
        print(
            f"{k:15s}: "
            f"{counts[k]:3d} "
            f"({100.0 * counts[k] / n:.1f}%)"
        )

    net = counts["face_rescue"] - counts["face_hurt"]

    print("\nNet rescue:")
    print(
        f"face_rescue - face_hurt = "
        f"{counts['face_rescue']} - "
        f"{counts['face_hurt']} = "
        f"{net:+d}"
    )

    # ---------------------------------------------------------
    # PER EMOTION
    # ---------------------------------------------------------
    print("\n========== PER-EMOTION RESCUE / HURT ==========")

    for eid, emo in enumerate(EMOTIONS):
        subset = [
            r for r in rows
            if r["true_id"] == eid
        ]

        rescue = sum(
            r["category"] == "face_rescue"
            for r in subset
        )

        hurt = sum(
            r["category"] == "face_hurt"
            for r in subset
        )

        print(
            f"{emo:10s} "
            f"n={len(subset):2d} | "
            f"rescue={rescue:2d} | "
            f"hurt={hurt:2d} | "
            f"net={rescue-hurt:+d}"
        )

    # ---------------------------------------------------------
    # REACTION GATE ANALYSIS
    # ---------------------------------------------------------
    print("\n========== REACTION GATE ==========")

    for cat in [
        "face_rescue",
        "face_hurt",
        "both_correct",
        "both_wrong",
    ]:
        vals = np.array([
            r["face_gate_react"]
            for r in rows
            if r["category"] == cat
        ])

        if len(vals) == 0:
            continue

        print(
            f"{cat:15s}: "
            f"{vals.mean():.4f} ± "
            f"{vals.std(ddof=1) if len(vals) > 1 else 0.0:.4f}"
        )

    # ---------------------------------------------------------
    # TOP RESCUES
    # ---------------------------------------------------------
    print("\n========== TOP FACE RESCUES ==========")

    rescued = sorted(
        [
            r for r in rows
            if r["category"] == "face_rescue"
        ],
        key=lambda x:
            x["true_prob_delta_face_minus_text"],
        reverse=True,
    )

    for r in rescued[:15]:
        print(
            f"\nidx={r['sample_index']} "
            f"true={r['true_emotion']} "
            f"text={r['text_pred_emotion']} "
            f"face={r['face_pred_emotion']}"
        )
        print(
            f"ΔP(true)="
            f"{r['true_prob_delta_face_minus_text']:+.3f} "
            f"gate={r['face_gate_react']:.3f}"
        )
        print(
            f"context: {r['context_text']}"
        )
        print(
            f"target : {r['target_text']}"
        )

    # ---------------------------------------------------------
    # SAVE
    # ---------------------------------------------------------
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with out_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    summary_path = out_path.with_suffix(".summary.json")

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "n": n,
                **counts,
                "net_rescue": net,
            },
            f,
            indent=2,
        )

    print("\nSaved:")
    print(out_path)
    print(summary_path)


if __name__ == "__main__":
    main()
    