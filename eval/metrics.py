"""Evaluation metrics for Stage A (style prediction) results, §19-20 of the
design doc. Operates on plain numpy arrays collected over a validation
split, independent of the training-time torch losses in `react_tts/losses.py`.
"""
from __future__ import annotations

import numpy as np


def emotion_accuracy_f1(
    preds: np.ndarray,
    targets: np.ndarray,
    num_classes: int,
) -> dict:
    """Emotion classification metrics.

    Returns overall accuracy, macro-F1 across all configured classes,
    per-class F1, class support, and confusion matrix.

    Confusion matrix convention:
        rows    = ground-truth classes
        columns = predicted classes
    """
    preds = np.asarray(preds).reshape(-1)
    targets = np.asarray(targets).reshape(-1)

    if preds.shape != targets.shape:
        raise ValueError(
            f"pred/target shape mismatch: "
            f"{preds.shape} vs {targets.shape}"
        )

    acc = float((preds == targets).mean())

    f1s = []
    supports = []

    confusion = np.zeros(
        (num_classes, num_classes),
        dtype=np.int64,
    )

    for true, pred in zip(targets, preds):
        true = int(true)
        pred = int(pred)

        if not (
            0 <= true < num_classes
            and 0 <= pred < num_classes
        ):
            raise ValueError(
                f"Emotion class outside valid range: "
                f"target={true}, pred={pred}, "
                f"num_classes={num_classes}"
            )

        confusion[true, pred] += 1

    for c in range(num_classes):
        tp = int(
            ((preds == c) & (targets == c)).sum()
        )
        fp = int(
            ((preds == c) & (targets != c)).sum()
        )
        fn = int(
            ((preds != c) & (targets == c)).sum()
        )

        precision = (
            tp / (tp + fp)
            if (tp + fp) > 0
            else 0.0
        )
        recall = (
            tp / (tp + fn)
            if (tp + fn) > 0
            else 0.0
        )

        f1 = (
            2.0 * precision * recall
            / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        f1s.append(float(f1))
        supports.append(
            int((targets == c).sum())
        )

    return {
        "accuracy": acc,
        "macro_f1": float(np.mean(f1s)),
        "per_class_f1": f1s,
        "class_support": supports,
        "confusion_matrix": confusion.tolist(),
    }


def concordance_correlation_coefficient(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Per-dimension CCC. pred/target: [N, D] -> [D]."""
    pred_mean, target_mean = pred.mean(axis=0), target.mean(axis=0)
    pred_var, target_var = pred.var(axis=0), target.var(axis=0)
    covariance = ((pred - pred_mean) * (target - target_mean)).mean(axis=0)
    return (2 * covariance) / (pred_var + target_var + (pred_mean - target_mean) ** 2 + 1e-8)


def vad_ccc(pred: np.ndarray, target: np.ndarray) -> dict:
    ccc = concordance_correlation_coefficient(pred, target)
    dims = ["valence", "arousal", "dominance"][: pred.shape[1]]
    return {f"ccc_{d}": float(c) for d, c in zip(dims, ccc)} | {"ccc_mean": float(ccc.mean())}


def pearson_correlation(pred: np.ndarray, target: np.ndarray) -> float:
    pred, target = pred.flatten(), target.flatten()
    if pred.std() < 1e-8 or target.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(pred, target)[0, 1])


def mean_absolute_error(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - target)))


def prosody_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    """pred/target: [N, 4] = [f0_mean, f0_std, energy, speaking_rate]."""
    names = ["f0_mean", "f0_std", "energy", "speaking_rate"]
    out = {}
    for i, name in enumerate(names):
        out[f"{name}_corr"] = pearson_correlation(pred[:, i], target[:, i])
        out[f"{name}_mae"] = mean_absolute_error(pred[:, i], target[:, i])
    return out


def style_cosine_similarity(pred_emb: np.ndarray, target_emb: np.ndarray) -> float:
    num = (pred_emb * target_emb).sum(axis=1)
    denom = np.linalg.norm(pred_emb, axis=1) * np.linalg.norm(target_emb, axis=1) + 1e-8
    return float((num / denom).mean())
