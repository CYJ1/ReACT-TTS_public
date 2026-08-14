from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from faster_whisper import WhisperModel
from resemblyzer import VoiceEncoder, preprocess_wav


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--temporal_dir",
        default="outputs/grad_stage_c_test252",
    )
    p.add_argument(
        "--text_dir",
        default="outputs/grad_stage_c_text_test252",
    )
    p.add_argument(
        "--output_csv",
        default="results/generated_speech_eval.csv",
    )
    p.add_argument(
        "--output_json",
        default="results/generated_speech_eval_summary.json",
    )

    p.add_argument("--start_idx", type=int, default=0)
    p.add_argument("--end_idx", type=int, default=None)

    p.add_argument(
        "--whisper_model",
        default="small.en",
    )
    p.add_argument(
        "--whisper_device",
        default="cuda",
    )
    p.add_argument(
        "--whisper_compute_type",
        default="float16",
    )

    return p.parse_args()


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def levenshtein(ref, hyp):
    n = len(ref)
    m = len(hyp)

    dp = list(range(m + 1))

    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i

        for j in range(1, m + 1):
            tmp = dp[j]

            cost = 0 if ref[i - 1] == hyp[j - 1] else 1

            dp[j] = min(
                dp[j] + 1,
                dp[j - 1] + 1,
                prev + cost,
            )

            prev = tmp

    return dp[m]


def wer_counts(ref_text, hyp_text):
    ref_words = normalize_text(ref_text).split()
    hyp_words = normalize_text(hyp_text).split()

    if len(ref_words) == 0:
        return 0, 0

    return (
        levenshtein(ref_words, hyp_words),
        len(ref_words),
    )


def cer_counts(ref_text, hyp_text):
    ref = normalize_text(ref_text).replace(" ", "")
    hyp = normalize_text(hyp_text).replace(" ", "")

    if len(ref) == 0:
        return 0, 0

    return (
        levenshtein(list(ref), list(hyp)),
        len(ref),
    )


def transcribe(model, wav_path):
    segments, _ = model.transcribe(
        str(wav_path),
        beam_size=5,
        language="en",
        vad_filter=True,
    )

    text = " ".join(
        seg.text.strip()
        for seg in segments
    )

    return text.strip()


def cosine_similarity(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    denom = (
        np.linalg.norm(a)
        * np.linalg.norm(b)
    )

    if denom <= 1e-12:
        return float("nan")

    return float(np.dot(a, b) / denom)


def get_speaker_embedding(encoder, wav_path):
    wav = preprocess_wav(Path(wav_path))

    if len(wav) == 0:
        raise RuntimeError(
            f"Empty audio: {wav_path}"
        )

    emb = encoder.embed_utterance(wav)
    return emb


def index_files(directory):
    directory = Path(directory)

    wavs = {}
    metas = {}

    for p in (directory / "wav").glob("*.wav"):
        m = re.match(r"idx(\d+)_", p.name)
        if m:
            wavs[int(m.group(1))] = p

    for p in (directory / "meta").glob("*.json"):
        m = re.match(r"idx(\d+)_", p.name)
        if m:
            metas[int(m.group(1))] = p

    return wavs, metas


def resolve_gt_audio(meta):
    # Cached manifest should normally contain target_wav_path.
    candidates = [
        meta.get("target_wav_path"),
        meta.get("target_audio_path"),
    ]

    for p in candidates:
        if p and os.path.exists(p):
            return p

    raise FileNotFoundError(
        "Could not find GT target audio. "
        f"target_wav_path={meta.get('target_wav_path')} "
        f"target_audio_path={meta.get('target_audio_path')}"
    )


def safe_mean(xs):
    xs = [
        x for x in xs
        if np.isfinite(x)
    ]

    return (
        float(np.mean(xs))
        if xs
        else float("nan")
    )


def main():
    args = parse_args()

    out_csv = Path(args.output_csv)
    out_json = Path(args.output_json)

    out_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    out_json.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporal_wavs, temporal_meta = index_files(
        args.temporal_dir
    )

    text_wavs, text_meta = index_files(
        args.text_dir
    )

    common = sorted(
        set(temporal_wavs)
        & set(text_wavs)
        & set(temporal_meta)
        & set(text_meta)
    )

    start = args.start_idx
    end = (
        len(common)
        if args.end_idx is None
        else min(args.end_idx, len(common))
    )

    indices = common[start:end]

    print("Common paired samples:", len(common))
    print(
        f"Evaluating position range [{start}, {end}) "
        f"= {len(indices)} samples"
    )

    print(
        "Loading Faster-Whisper:",
        args.whisper_model,
    )

    whisper = WhisperModel(
        args.whisper_model,
        device=args.whisper_device,
        compute_type=args.whisper_compute_type,
    )

    print("Loading Resemblyzer...")
    speaker_encoder = VoiceEncoder(
        device="cpu"
    )

    rows = []

    temporal_word_err = 0
    temporal_word_ref = 0
    text_word_err = 0
    text_word_ref = 0

    temporal_char_err = 0
    temporal_char_ref = 0
    text_char_err = 0
    text_char_ref = 0

    temp_sims = []
    text_sims = []

    failures = []

    for pos, idx in enumerate(indices, start=1):

        try:
            with open(temporal_meta[idx]) as f:
                meta = json.load(f)

            target_text = meta["target_text"]
            gt_path = resolve_gt_audio(meta)

            temp_path = temporal_wavs[idx]
            text_path = text_wavs[idx]

            # ---------------------------------------------
            # ASR
            # ---------------------------------------------
            temp_hyp = transcribe(
                whisper,
                temp_path,
            )

            text_hyp = transcribe(
                whisper,
                text_path,
            )

            tw_err, tw_ref = wer_counts(
                target_text,
                temp_hyp,
            )

            xw_err, xw_ref = wer_counts(
                target_text,
                text_hyp,
            )

            tc_err, tc_ref = cer_counts(
                target_text,
                temp_hyp,
            )

            xc_err, xc_ref = cer_counts(
                target_text,
                text_hyp,
            )

            temporal_word_err += tw_err
            temporal_word_ref += tw_ref

            text_word_err += xw_err
            text_word_ref += xw_ref

            temporal_char_err += tc_err
            temporal_char_ref += tc_ref

            text_char_err += xc_err
            text_char_ref += xc_ref

            temp_wer = (
                tw_err / tw_ref
                if tw_ref else 0.0
            )
            text_wer = (
                xw_err / xw_ref
                if xw_ref else 0.0
            )

            temp_cer = (
                tc_err / tc_ref
                if tc_ref else 0.0
            )
            text_cer = (
                xc_err / xc_ref
                if xc_ref else 0.0
            )

            # ---------------------------------------------
            # Speaker similarity
            # ---------------------------------------------
            gt_emb = get_speaker_embedding(
                speaker_encoder,
                gt_path,
            )

            temp_emb = get_speaker_embedding(
                speaker_encoder,
                temp_path,
            )

            text_emb = get_speaker_embedding(
                speaker_encoder,
                text_path,
            )

            temp_sim = cosine_similarity(
                gt_emb,
                temp_emb,
            )

            text_sim = cosine_similarity(
                gt_emb,
                text_emb,
            )

            temp_sims.append(temp_sim)
            text_sims.append(text_sim)

            row = {
                "idx": idx,
                "dialogue_id": meta.get("dialogue_id"),
                "utterance_id": meta.get(
                    "target_utterance_id",
                    meta.get("utterance_id"),
                ),
                "speaker": meta.get(
                    "target_speaker_id"
                ),
                "emotion": meta.get(
                    "target_emotion"
                ),
                "reference_text": target_text,
                "temporal_asr": temp_hyp,
                "text_asr": text_hyp,
                "temporal_wer": temp_wer,
                "text_wer": text_wer,
                "temporal_cer": temp_cer,
                "text_cer": text_cer,
                "temporal_spk_sim": temp_sim,
                "text_spk_sim": text_sim,
                "temporal_wav": str(temp_path),
                "text_wav": str(text_path),
                "gt_wav": str(gt_path),
            }

            rows.append(row)

            print(
                f"[{pos}/{len(indices)}] "
                f"idx={idx} "
                f"WER T={temp_wer:.3f} "
                f"X={text_wer:.3f} | "
                f"SPK T={temp_sim:.3f} "
                f"X={text_sim:.3f}"
            )

        except Exception as e:
            failures.append(
                {
                    "idx": idx,
                    "error": repr(e),
                }
            )

            print(
                f"[{pos}/{len(indices)}] "
                f"FAIL idx={idx}: {repr(e)}"
            )

    # ---------------------------------------------------------
    # CSV
    # ---------------------------------------------------------
    if rows:
        with open(
            out_csv,
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

    temporal_wer = (
        temporal_word_err
        / temporal_word_ref
        if temporal_word_ref
        else float("nan")
    )

    text_wer = (
        text_word_err
        / text_word_ref
        if text_word_ref
        else float("nan")
    )

    temporal_cer = (
        temporal_char_err
        / temporal_char_ref
        if temporal_char_ref
        else float("nan")
    )

    text_cer = (
        text_char_err
        / text_char_ref
        if text_char_ref
        else float("nan")
    )

    summary = {
        "num_pairs": len(rows),
        "num_failures": len(failures),

        "temporal": {
            "wer": temporal_wer,
            "cer": temporal_cer,
            "speaker_similarity_mean": safe_mean(
                temp_sims
            ),
        },

        "text_only": {
            "wer": text_wer,
            "cer": text_cer,
            "speaker_similarity_mean": safe_mean(
                text_sims
            ),
        },

        "delta_temporal_minus_text": {
            "wer": temporal_wer - text_wer,
            "cer": temporal_cer - text_cer,
            "speaker_similarity": (
                safe_mean(temp_sims)
                - safe_mean(text_sims)
            ),
        },

        "failures": failures,
    }

    with open(
        out_json,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            summary,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print("============ SUMMARY ============")
    print("Pairs:", len(rows))
    print("Failures:", len(failures))
    print()
    print(
        f"Temporal WER: {temporal_wer:.4f}"
    )
    print(
        f"Text-only WER: {text_wer:.4f}"
    )
    print()
    print(
        f"Temporal CER: {temporal_cer:.4f}"
    )
    print(
        f"Text-only CER: {text_cer:.4f}"
    )
    print()
    print(
        "Temporal speaker sim:",
        f"{safe_mean(temp_sims):.4f}",
    )
    print(
        "Text-only speaker sim:",
        f"{safe_mean(text_sims):.4f}",
    )
    print()
    print("CSV:", out_csv)
    print("JSON:", out_json)
    print("=================================")


if __name__ == "__main__":
    main()
