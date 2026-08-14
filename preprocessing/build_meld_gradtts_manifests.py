from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


SPLIT_INFO = {
    "train": {
        "csv": "train_sent_emo.csv",
        "video_dir": "train_splits",
    },
    "dev": {
        "csv": "dev_sent_emo.csv",
        "video_dir": "dev_splits_complete",
    },
    "test": {
        "csv": "test_sent_emo.csv",
        "video_dir": "output_repeated_splits_test",
    },
}


def read_csv_rows(csv_path: Path):
    rows = []

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            rows.append(row)

    return rows


def normalize_dialogue_id(x):
    return str(int(x))


def normalize_utterance_id(x):
    return str(int(x))


def make_audio_path(
    meld_root: Path,
    split: str,
    dialogue_id: str,
    utterance_id: str,
):
    video_dir = SPLIT_INFO[split]["video_dir"]

    return (
        meld_root
        / video_dir
        / f"dia{dialogue_id}_utt{utterance_id}.mp4"
    )


def build_csv_index(rows):
    """
    Build indices for target recovery.

    Primary matching uses:
      dialogue_id + target speaker + target text

    If that is ambiguous, we additionally use the ordered preceding
    context turns from the strict ReACT-TTS manifest.
    """

    exact = {}
    by_dialogue_speaker = {}
    by_dialogue = {}

    for row in rows:
        did = normalize_dialogue_id(row["Dialogue_ID"])
        speaker = row["Speaker"].strip()
        utterance = row["Utterance"].strip()

        exact.setdefault(
            (did, speaker, utterance),
            [],
        ).append(row)

        by_dialogue_speaker.setdefault(
            (did, speaker),
            [],
        ).append(row)

        by_dialogue.setdefault(
            did,
            [],
        ).append(row)

    # CSV order should already correspond to utterance order,
    # but explicitly sorting by Utterance_ID makes recovery deterministic.
    for did in by_dialogue:
        by_dialogue[did].sort(
            key=lambda r: int(r["Utterance_ID"])
        )

    return exact, by_dialogue_speaker, by_dialogue

def write_jsonl(path: Path, items):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(
                json.dumps(
                    item,
                    ensure_ascii=False,
                )
                + "\n"
            )


def build_stage_b(
    meld_root: Path,
    split: str,
):
    csv_path = meld_root / SPLIT_INFO[split]["csv"]
    rows = read_csv_rows(csv_path)

    output = []
    missing_audio = 0

    for row in rows:
        did = normalize_dialogue_id(row["Dialogue_ID"])
        uid = normalize_utterance_id(row["Utterance_ID"])

        audio_path = make_audio_path(
            meld_root,
            split,
            did,
            uid,
        )

        if not audio_path.exists():
            missing_audio += 1
            continue

        text = row["Utterance"].strip()
        speaker = row["Speaker"].strip()
        emotion = row["Emotion"].strip().lower()

        if not text:
            continue

        output.append(
            {
                "dialogue_id": did,
                "utterance_id": uid,
                "target_text": text,
                "target_speaker_id": speaker,
                "target_emotion": emotion,
                "target_audio_path": str(
                    audio_path.resolve()
                ),
            }
        )

    print(
        f"[Stage B / {split}] "
        f"CSV={len(rows)} "
        f"usable={len(output)} "
        f"missing_audio={missing_audio}"
    )

    return output


def _norm_text(text):
    # Preserve wording but make harmless whitespace differences irrelevant.
    return " ".join(str(text).strip().split())


def recover_target_row(
    sample,
    exact_index,
    fallback_index,
    dialogue_index,
):
    did = str(sample["dialogue_id"])
    speaker = sample["target_speaker_id"].strip()
    text = _norm_text(sample["target_text"])

    exact_key = (
        did,
        speaker,
        sample["target_text"].strip(),
    )

    candidates = exact_index.get(
        exact_key,
        [],
    )

    if len(candidates) == 1:
        return candidates[0], "exact"

    # ---------------------------------------------------------
    # Ambiguous target text:
    # use the ordered dialogue context to identify which
    # occurrence is the actual response.
    # ---------------------------------------------------------
    dialogue_rows = dialogue_index.get(did, [])
    context_turns = sample.get("context_turns", [])

    context_matches = []

    for candidate in candidates:
        candidate_uid = int(candidate["Utterance_ID"])

        # Find the candidate's position in this dialogue.
        target_pos = None
        for i, row in enumerate(dialogue_rows):
            if int(row["Utterance_ID"]) == candidate_uid:
                target_pos = i
                break

        if target_pos is None:
            continue

        n_ctx = len(context_turns)

        # Target must have enough preceding turns to match the
        # context stored in the ReACT-TTS manifest.
        if target_pos < n_ctx:
            continue

        preceding = dialogue_rows[
            target_pos - n_ctx : target_pos
        ]

        ok = True

        for manifest_turn, csv_turn in zip(
            context_turns,
            preceding,
        ):
            if (
                manifest_turn["speaker_id"].strip()
                != csv_turn["Speaker"].strip()
            ):
                ok = False
                break

            if (
                _norm_text(manifest_turn["text"])
                != _norm_text(csv_turn["Utterance"])
            ):
                ok = False
                break

        if ok:
            context_matches.append(candidate)

    if len(context_matches) == 1:
        return context_matches[0], "context_disambiguated"

    if len(context_matches) > 1:
        raise RuntimeError(
            "Context still ambiguous: "
            f"dialogue={did}, speaker={speaker}, "
            f"text={text!r}, matches={len(context_matches)}"
        )

    # ---------------------------------------------------------
    # Final fallback for cases where the original strict subset
    # contains formatting differences in the target text.
    # ---------------------------------------------------------
    candidates = fallback_index.get(
        (did, speaker),
        [],
    )

    text_matches = [
        r
        for r in candidates
        if _norm_text(r["Utterance"]) == text
    ]

    if len(text_matches) == 1:
        return text_matches[0], "fallback_text"

    # Try context matching again over the fallback text matches.
    context_matches = []

    for candidate in text_matches:
        candidate_uid = int(candidate["Utterance_ID"])

        target_pos = None
        for i, row in enumerate(dialogue_rows):
            if int(row["Utterance_ID"]) == candidate_uid:
                target_pos = i
                break

        if target_pos is None:
            continue

        n_ctx = len(context_turns)

        if target_pos < n_ctx:
            continue

        preceding = dialogue_rows[
            target_pos - n_ctx : target_pos
        ]

        ok = True

        for manifest_turn, csv_turn in zip(
            context_turns,
            preceding,
        ):
            if (
                manifest_turn["speaker_id"].strip()
                != csv_turn["Speaker"].strip()
                or _norm_text(manifest_turn["text"])
                != _norm_text(csv_turn["Utterance"])
            ):
                ok = False
                break

        if ok:
            context_matches.append(candidate)

    if len(context_matches) == 1:
        return context_matches[0], "context_fallback"

    raise RuntimeError(
        "Could not uniquely recover target utterance: "
        f"dialogue={did}, speaker={speaker}, text={text!r}, "
        f"exact_candidates={len(candidates)}, "
        f"context_matches={len(context_matches)}"
    )

def build_stage_c(
    meld_root: Path,
    strict_manifest: Path,
    split: str,
):
    csv_path = meld_root / SPLIT_INFO[split]["csv"]
    rows = read_csv_rows(csv_path)

    exact_index, fallback_index, dialogue_index = build_csv_index(rows)

    recovered = []
    match_stats = {}

    with strict_manifest.open(
        encoding="utf-8"
    ) as f:
        for line in f:
            if not line.strip():
                continue

            sample = json.loads(line)

            row, method = recover_target_row(
                sample,
                exact_index,
                fallback_index,
                dialogue_index,
            )

            match_stats[method] = (
                match_stats.get(method, 0) + 1
            )

            did = normalize_dialogue_id(
                row["Dialogue_ID"]
            )
            uid = normalize_utterance_id(
                row["Utterance_ID"]
            )

            audio_path = make_audio_path(
                meld_root,
                split,
                did,
                uid,
            )

            if not audio_path.exists():
                raise FileNotFoundError(
                    f"Recovered target audio missing: {audio_path}"
                )

            out = dict(sample)

            out["target_utterance_id"] = uid
            out["target_audio_path"] = str(
                audio_path.resolve()
            )

            recovered.append(out)

    print(
        f"[Stage C / {split}] "
        f"recovered={len(recovered)} "
        f"match_stats={match_stats}"
    )

    return recovered


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--meld_root",
        default="../data/meld_raw/MELD.Raw",
    )

    parser.add_argument(
        "--strict_manifest_root",
        default="../data/meld_full_v2/final/manifests",
    )

    parser.add_argument(
        "--out_root",
        default="../data/meld_gradtts",
    )

    args = parser.parse_args()

    meld_root = Path(args.meld_root).resolve()
    strict_root = Path(
        args.strict_manifest_root
    ).resolve()
    out_root = Path(args.out_root).resolve()

    print("MELD root:", meld_root)
    print("Strict manifest root:", strict_root)
    print("Output root:", out_root)

    for split in [
        "train",
        "dev",
        "test",
    ]:
        stage_b = build_stage_b(
            meld_root,
            split,
        )

        write_jsonl(
            out_root
            / "stage_b"
            / f"{split}.jsonl",
            stage_b,
        )

        strict_manifest = (
            strict_root
            / f"{split}.jsonl"
        )

        stage_c = build_stage_c(
            meld_root,
            strict_manifest,
            split,
        )

        write_jsonl(
            out_root
            / "stage_c"
            / f"{split}.jsonl",
            stage_c,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
