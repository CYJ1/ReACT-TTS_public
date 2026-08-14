from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for x in rows:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")


def convert(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-y",
        "-loglevel", "error",
        "-i", src,
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-acodec", "pcm_s16le",
        str(dst),
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    return result.returncode == 0, result.stderr.strip()


def process_manifest(src_manifest, dst_manifest, wav_root, split):
    rows = load_jsonl(src_manifest)

    kept = []
    failed = []

    for i, row in enumerate(rows):
        did = row["dialogue_id"]
        uid = row.get("utterance_id", row.get("target_utterance_id"))

        if uid is None:
            raise RuntimeError(
                f"No utterance id for sample: {row}"
            )

        wav_path = (
            Path(wav_root)
            / split
            / f"dia{did}_utt{uid}.wav"
        )

        if wav_path.exists() and wav_path.stat().st_size > 1000:
            ok = True
            err = ""
        else:
            ok, err = convert(
                row["target_audio_path"],
                wav_path,
            )

        if ok:
            out = dict(row)
            out["target_wav_path"] = str(wav_path.resolve())
            kept.append(out)
        else:
            if wav_path.exists():
                wav_path.unlink()

            failed.append({
                "dialogue_id": did,
                "utterance_id": uid,
                "speaker": row["target_speaker_id"],
                "audio": row["target_audio_path"],
                "error": err,
            })

        if (i + 1) % 500 == 0:
            print(
                f"[{split}] {i+1}/{len(rows)} "
                f"kept={len(kept)} failed={len(failed)}"
            )

    write_jsonl(dst_manifest, kept)

    print(
        f"\n[{split}] original={len(rows)} "
        f"usable={len(kept)} "
        f"failed={len(failed)}"
    )

    return failed


def main():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--root",
        default="../data/meld_gradtts",
    )

    args = p.parse_args()

    root = Path(args.root)
    wav_root = root / "wav16k"

    all_failed = {}

    # Stage B
    for split, filename in [
        ("train", "train.jsonl"),
        ("dev", "dev_seen.jsonl"),
    ]:
        src = root / "stage_b" / filename
        dst = root / "stage_b_cached" / filename

        failed = process_manifest(
            src,
            dst,
            wav_root,
            f"stage_b_{split}",
        )

        all_failed[f"stage_b_{split}"] = failed

    # Stage C
    for split, filename in [
        ("train", "train.jsonl"),
        ("dev", "dev_seen.jsonl"),
        ("test", "test_seen.jsonl"),
    ]:
        src = root / "stage_c" / filename
        dst = root / "stage_c_cached" / filename

        failed = process_manifest(
            src,
            dst,
            wav_root,
            f"stage_c_{split}",
        )

        all_failed[f"stage_c_{split}"] = failed

    fail_path = root / "audio_decode_failures.json"

    with fail_path.open("w", encoding="utf-8") as f:
        json.dump(
            all_failed,
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\nSaved failure report:", fail_path)


if __name__ == "__main__":
    main()
