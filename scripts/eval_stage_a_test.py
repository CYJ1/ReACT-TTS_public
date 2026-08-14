from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from eval.metrics import emotion_accuracy_f1, prosody_metrics, vad_ccc
from react_tts.config import load_config
from react_tts.data.dialogue_dataset import DialogueStyleDataset
from react_tts.models.response_style_predictor import build_response_style_predictor


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--config", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--manifest_test", required=True)

    p.add_argument(
        "--condition",
        required=True,
        choices=["temporal", "text_only", "static", "full"],
    )

    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    p.add_argument("--out_json", required=True)
    p.add_argument("--out_npz", required=True)

    return p.parse_args()


def build_test_dataset(cfg, manifest_test):
    d = cfg.data

    return DialogueStyleDataset(
        manifest_test,
        tokenizer_spec=d.tokenizer,
        vocab_size=cfg.model.text.vocab_size,
        num_context_turns=d.num_context_turns,
        max_text_len=d.max_text_len,
        reaction_num_frames=d.reaction_num_frames,
        face_feat_dim=cfg.model.reaction.face_feat_dim,
        synthetic=False,
    )


def configure_condition(cfg, condition):
    # Start from the trained condition represented by the checkpoint.
    if condition == "temporal":
        # proposed model: temporal listener reaction, no explicit delta
        cfg.model.reaction.use_listener_face = True
        cfg.model.reaction.temporal_face = True
        cfg.model.reaction.use_reaction_delta = False

    elif condition == "text_only":
        cfg.model.reaction.use_listener_face = False

    elif condition == "static":
        cfg.model.reaction.use_listener_face = True
        cfg.model.reaction.temporal_face = False

    elif condition == "full":
        # temporal + explicit reaction delta
        cfg.model.reaction.use_listener_face = True
        cfg.model.reaction.temporal_face = True
        cfg.model.reaction.use_reaction_delta = True

    else:
        raise ValueError(condition)


def load_model(cfg, ckpt_path, device):
    model = build_response_style_predictor(cfg).to(device)

    state = torch.load(ckpt_path, map_location=device)

    # Compatible with both raw state_dict and {"model": state_dict}.
    if isinstance(state, dict) and "model" in state:
        state = state["model"]

    model.load_state_dict(state)
    model.eval()

    return model


@torch.no_grad()
def run_eval(model, loader, device, num_classes, do_mismatch):
    emo_true = []

    emo_pred_correct = []
    emo_prob_correct = []

    vad_pred = []
    vad_true = []

    prosody_pred = []
    prosody_true = []

    gate_values = []

    emo_pred_mismatch = []
    emo_prob_mismatch = []

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}

        # --------------------------------------------------
        # Correct listener
        # --------------------------------------------------
        out = model(
            batch,
            random_listener_override=False,
        )

        logits = out["emotion_logits"]
        probs = torch.softmax(logits, dim=-1)

        emo_true.append(
            batch["emotion_id"].cpu().numpy()
        )

        emo_pred_correct.append(
            logits.argmax(dim=-1).cpu().numpy()
        )

        emo_prob_correct.append(
            probs.cpu().numpy()
        )

        vad_pred.append(
            out["vad"].cpu().numpy()
        )
        vad_true.append(
            batch["vad"].cpu().numpy()
        )

        prosody_pred.append(
            out["prosody"].cpu().numpy()
        )
        prosody_true.append(
            batch["prosody"].cpu().numpy()
        )

        if "gate_react" in out:
            gate_values.append(
                out["gate_react"].cpu().numpy()
            )

        # --------------------------------------------------
        # Same checkpoint, mismatched listener
        # Only for proposed temporal model.
        # --------------------------------------------------
        if do_mismatch:
            out_m = model(
                batch,
                random_listener_override=True,
            )

            logits_m = out_m["emotion_logits"]
            probs_m = torch.softmax(
                logits_m,
                dim=-1,
            )

            emo_pred_mismatch.append(
                logits_m.argmax(dim=-1).cpu().numpy()
            )

            emo_prob_mismatch.append(
                probs_m.cpu().numpy()
            )

    y_true = np.concatenate(emo_true)
    y_pred = np.concatenate(emo_pred_correct)
    probs_correct = np.concatenate(emo_prob_correct)

    v_pred = np.concatenate(vad_pred)
    v_true = np.concatenate(vad_true)

    p_pred = np.concatenate(prosody_pred)
    p_true = np.concatenate(prosody_true)

    metrics = emotion_accuracy_f1(
        y_pred,
        y_true,
        num_classes,
    )

    metrics.update(
        vad_ccc(
            v_pred,
            v_true,
        )
    )

    metrics.update(
        prosody_metrics(
            p_pred,
            p_true,
        )
    )

    if gate_values:
        metrics["mean_gate_react"] = float(
            np.concatenate(gate_values).mean()
        )

    result = {
        "metrics_correct": metrics,
    }

    arrays = {
        "y_true": y_true,
        "y_pred_correct": y_pred,
        "prob_correct": probs_correct,
        "vad_true": v_true,
        "vad_pred": v_pred,
        "prosody_true": p_true,
        "prosody_pred": p_pred,
    }

    if do_mismatch:
        y_pred_m = np.concatenate(
            emo_pred_mismatch
        )

        probs_m = np.concatenate(
            emo_prob_mismatch
        )

        mismatch_metrics = emotion_accuracy_f1(
            y_pred_m,
            y_true,
            num_classes,
        )

        result["metrics_mismatch"] = mismatch_metrics

        result["delta_macro_f1_correct_minus_mismatch"] = (
            metrics["macro_f1"]
            - mismatch_metrics["macro_f1"]
        )

        result["delta_accuracy_correct_minus_mismatch"] = (
            metrics["accuracy"]
            - mismatch_metrics["accuracy"]
        )

        arrays["y_pred_mismatch"] = y_pred_m
        arrays["prob_mismatch"] = probs_m

    return result, arrays


def make_jsonable(x):
    if isinstance(x, dict):
        return {
            k: make_jsonable(v)
            for k, v in x.items()
        }

    if isinstance(x, (list, tuple)):
        return [
            make_jsonable(v)
            for v in x
        ]

    if isinstance(x, np.ndarray):
        return x.tolist()

    if isinstance(x, (np.floating, np.integer)):
        return x.item()

    return x


def main():
    args = parse_args()

    cfg = load_config(args.config)

    # Critical: Test is real data.
    cfg.data.synthetic = False

    configure_condition(
        cfg,
        args.condition,
    )

    test_ds = build_test_dataset(
        cfg,
        args.manifest_test,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
    )

    print("[condition]", args.condition)
    print("[checkpoint]", args.ckpt)
    print("[manifest_test]", args.manifest_test)
    print("[num_test]", len(test_ds))

    if len(test_ds) != 261:
        raise RuntimeError(
            f"Expected frozen Test size 261, got {len(test_ds)}"
        )

    model = load_model(
        cfg,
        args.ckpt,
        args.device,
    )

    # Counterfactual only for proposed Temporal No-Delta.
    do_mismatch = (
        args.condition == "temporal"
    )

    result, arrays = run_eval(
        model,
        test_loader,
        args.device,
        cfg.data.num_emotion_classes,
        do_mismatch=do_mismatch,
    )

    result["condition"] = args.condition
    result["checkpoint"] = args.ckpt
    result["manifest_test"] = args.manifest_test
    result["num_test"] = len(test_ds)

    print("\n========== TEST RESULT ==========")
    print(
        json.dumps(
            make_jsonable(result),
            indent=2,
        )
    )

    out_json = Path(args.out_json)
    out_npz = Path(args.out_npz)

    out_json.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    out_npz.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with out_json.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            make_jsonable(result),
            f,
            indent=2,
        )

    np.savez_compressed(
        out_npz,
        **arrays,
    )

    print("\nSaved:")
    print(out_json)
    print(out_npz)


if __name__ == "__main__":
    main()
