from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from preprocessing.extract_speaker_embedding import embed_speaker_from_wavs
except ModuleNotFoundError:
    from extract_speaker_embedding import embed_speaker_from_wavs


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def safe_name(name: str) -> str:
    keep = []
    for c in name:
        if c.isalnum() or c in "-_":
            keep.append(c)
        else:
            keep.append("_")
    return "".join(keep)


def decode_to_wav(src, out_path):
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-i",
            src,
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-acodec",
            "pcm_s16le",
            str(out_path),
        ],
        check=True,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--train_manifest",
        default="../data/meld_gradtts/stage_b/train.jsonl",
    )
    p.add_argument(
        "--out_dir",
        default="../data/meld_gradtts/speaker_embeddings_train",
    )
    p.add_argument(
        "--max_refs_per_speaker",
        type=int,
        default=10,
    )
    args = p.parse_args()

    rows = load_jsonl(args.train_manifest)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_speaker = defaultdict(list)
    for row in rows:
        by_speaker[row["target_speaker_id"]].append(
            row["target_audio_path"]
        )

    table = {}
    failed = []

    print("Train speakers:", len(by_speaker))

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)

        for i, (speaker, paths) in enumerate(sorted(by_speaker.items())):
            refs = paths[: args.max_refs_per_speaker]
            wav_paths = []

            try:
                for j, src in enumerate(refs):
                    wav = td / f"{i}_{j}.wav"
                    decode_to_wav(src, wav)
                    wav_paths.append(str(wav))

                emb = embed_speaker_from_wavs(wav_paths)

                out = out_dir / f"{safe_name(speaker)}.npy"
                np.save(out, emb.astype(np.float32))

                table[speaker] = str(out.resolve())

                print(
                    f"[{i+1}/{len(by_speaker)}] "
                    f"{speaker!r}: refs={len(wav_paths)}"
                )

            except Exception as e:
                failed.append((speaker, str(e)))
                print(f"FAILED {speaker!r}: {e}")

    table_path = out_dir / "speaker_table.json"
    with table_path.open("w", encoding="utf-8") as f:
        json.dump(table, f, indent=2, ensure_ascii=False)

    print("\nSaved:", table_path)
    print("Successful speakers:", len(table))
    print("Failed speakers:", len(failed))

    if failed:
        for x in failed[:20]:
            print("FAIL:", x)


if __name__ == "__main__":
    main()
