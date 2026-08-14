import csv
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from preprocessing.extract_prosody_features import extract_raw_prosody


ROOT = Path("/workspace/REACT_TTS_rev/data")

MANIFEST_DIR = ROOT / "meld_full_v2/final/manifests"
MELD_ROOT = ROOT / "meld_raw/MELD.Raw"

CSV_PATHS = {
    "train": MELD_ROOT / "train_sent_emo.csv",
    "dev": MELD_ROOT / "dev_sent_emo.csv",
    "test": MELD_ROOT / "test_sent_emo.csv",
}

VIDEO_DIRS = {
    "train": MELD_ROOT / "train_splits",
    "dev": MELD_ROOT / "dev_splits_complete",
    "test": MELD_ROOT / "output_repeated_splits_test",
}


def norm_text(x):
    x = str(x).strip()
    return re.sub(r"\s+", " ", x)


def load_csv_by_dialogue(path):
    by_dialogue = defaultdict(list)

    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            by_dialogue[int(r["Dialogue_ID"])].append(r)

    for did in by_dialogue:
        by_dialogue[did].sort(
            key=lambda x: int(x["Utterance_ID"])
        )

    return by_dialogue


def resolve_target(sample, by_dialogue):
    did = int(sample["dialogue_id"])

    candidates = [
        r
        for r in by_dialogue[did]
        if norm_text(r["Utterance"])
        == norm_text(sample["target_text"])
        and r["Speaker"].strip()
        == sample["target_speaker_id"].strip()
        and r["Emotion"].strip().lower()
        == sample["target_emotion"].strip().lower()
    ]

    if len(candidates) == 1:
        return candidates[0]

    # Resolve duplicate target utterances using context.
    resolved = []

    for cand in candidates:
        uid = int(cand["Utterance_ID"])

        prev = [
            r
            for r in by_dialogue[did]
            if int(r["Utterance_ID"]) < uid
        ]

        prev = prev[-len(sample["context_turns"]):]

        if len(prev) != len(sample["context_turns"]):
            continue

        ok = True

        for csv_turn, manifest_turn in zip(
            prev,
            sample["context_turns"],
        ):
            if (
                norm_text(csv_turn["Utterance"])
                != norm_text(manifest_turn["text"])
                or csv_turn["Speaker"].strip()
                != manifest_turn["speaker_id"].strip()
            ):
                ok = False
                break

        if ok:
            resolved.append(cand)

    if len(resolved) != 1:
        raise RuntimeError(
            f"Could not uniquely resolve target: "
            f"dialogue={did}, "
            f"text={sample['target_text']!r}, "
            f"candidates={len(candidates)}, "
            f"context_resolved={len(resolved)}"
        )

    return resolved[0]


def extract_split(split):
    manifest_path = MANIFEST_DIR / f"{split}.jsonl"

    samples = [
        json.loads(line)
        for line in open(manifest_path, encoding="utf-8")
    ]

    by_dialogue = load_csv_by_dialogue(
        CSV_PATHS[split]
    )

    raw_features = []
    metadata = []

    failures = []

    print(
        f"\n=== EXTRACTING {split.upper()} "
        f"({len(samples)} samples) ===",
        flush=True,
    )

    for i, sample in enumerate(samples):
        try:
            target = resolve_target(
                sample,
                by_dialogue,
            )

            did = int(target["Dialogue_ID"])
            uid = int(target["Utterance_ID"])

            video_path = (
                VIDEO_DIRS[split]
                / f"dia{did}_utt{uid}.mp4"
            )

            if not video_path.exists():
                raise FileNotFoundError(
                    str(video_path)
                )

            num_words = max(
                len(sample["target_text"].split()),
                1,
            )

            raw = extract_raw_prosody(
                str(video_path),
                num_words=num_words,
            )

            vec = np.asarray(
                [
                    raw.f0_mean,
                    raw.f0_std,
                    raw.energy,
                    raw.speaking_rate,
                ],
                dtype=np.float64,
            )

            if not np.all(np.isfinite(vec)):
                raise ValueError(
                    f"Non-finite prosody: {vec}"
                )

            raw_features.append(vec)

            metadata.append(
                {
                    "dialogue_id": did,
                    "utterance_id": uid,
                    "video_path": str(video_path),
                }
            )

        except Exception as e:
            failures.append(
                {
                    "index": i,
                    "dialogue_id": sample.get(
                        "dialogue_id"
                    ),
                    "target_text": sample.get(
                        "target_text"
                    ),
                    "error": repr(e),
                }
            )

        if (i + 1) % 50 == 0 or i + 1 == len(samples):
            print(
                f"[{split}] "
                f"{i + 1}/{len(samples)} "
                f"success={len(raw_features)} "
                f"failed={len(failures)}",
                flush=True,
            )

    if failures:
        print(
            "\nPROSODY EXTRACTION FAILURES:",
            json.dumps(
                failures[:20],
                ensure_ascii=False,
                indent=2,
            ),
        )

        raise RuntimeError(
            f"{split}: "
            f"{len(failures)} prosody extractions failed. "
            "Manifest was NOT modified."
        )

    if len(raw_features) != len(samples):
        raise RuntimeError(
            f"{split}: extracted "
            f"{len(raw_features)} / {len(samples)}"
        )

    return (
        samples,
        np.stack(raw_features, axis=0),
        metadata,
    )


def main():
    split_data = {}

    for split in ["train", "dev", "test"]:
        split_data[split] = extract_split(split)

    # -------------------------------------------------
    # Fit normalization ONLY on the training split.
    # -------------------------------------------------
    train_raw = split_data["train"][1]

    mean = train_raw.mean(axis=0)
    std = train_raw.std(axis=0)

    # Fail rather than silently creating invalid targets.
    if np.any(std < 1e-8):
        raise RuntimeError(
            f"Near-zero training std detected: {std}"
        )

    names = [
        "f0_mean",
        "f0_std",
        "energy",
        "speaking_rate",
    ]

    print("\n=== TRAIN-DERIVED GLOBAL PROSODY STATS ===")

    for name, mu, sigma in zip(
        names,
        mean,
        std,
    ):
        print(
            f"{name:15s} "
            f"mean={mu:.8f} "
            f"std={sigma:.8f}"
        )

    # Diagnostics for possible unvoiced samples.
    for split in ["train", "dev", "test"]:
        raw = split_data[split][1]

        print(
            f"{split}: "
            f"zero_f0_mean="
            f"{int((raw[:, 0] <= 0).sum())}/"
            f"{len(raw)}, "
            f"zero_f0_std="
            f"{int((raw[:, 1] <= 0).sum())}/"
            f"{len(raw)}"
        )

    stats = {
        "normalization": (
            "global z-score using training split only"
        ),
        "feature_order": names,
        "train_mean": mean.tolist(),
        "train_std": std.tolist(),
        "train_samples": int(
            len(split_data["train"][1])
        ),
    }

    stats_path = (
        MANIFEST_DIR
        / "prosody_stats_train.json"
    )

    with open(
        stats_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            stats,
            f,
            indent=2,
            ensure_ascii=False,
        )

    # -------------------------------------------------
    # Apply same TRAIN statistics to ALL splits.
    # -------------------------------------------------
    for split in ["train", "dev", "test"]:
        samples, raw, metadata = split_data[split]

        normalized = (
            raw - mean[None, :]
        ) / std[None, :]

        output_path = (
            MANIFEST_DIR
            / f"{split}_prosody_fixed.jsonl"
        )

        with open(
            output_path,
            "w",
            encoding="utf-8",
        ) as f:
            for sample, vec in zip(
                samples,
                normalized,
            ):
                new_sample = dict(sample)

                new_sample["target_prosody"] = [
                    float(x)
                    for x in vec
                ]

                f.write(
                    json.dumps(
                        new_sample,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        print(
            f"\n[{split}] wrote: {output_path}"
        )
        print(
            " normalized mean:",
            normalized.mean(axis=0),
        )
        print(
            " normalized std :",
            normalized.std(axis=0),
        )
        print(
            " all-zero rows  :",
            int(
                np.all(
                    normalized == 0,
                    axis=1,
                ).sum()
            ),
        )

    print("\nDONE.")
    print(
        "Original manifests were NOT overwritten."
    )
    print(
        f"Stats saved to: {stats_path}"
    )


if __name__ == "__main__":
    main()
