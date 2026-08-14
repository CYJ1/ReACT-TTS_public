"""Real target-speaker embedding for the TTS backbone's `speaker_embedding`
condition, using Resemblyzer (a pretrained d-vector-style speaker encoder).

Chosen specifically because its pretrained weights ship *inside the pip
package* (downloaded from PyPI at install time), unlike most speaker/face
encoders whose weights are hosted on GitHub releases or a model hub -- which
matters in network-restricted environments (see README "Pretrained encoders").

    pip install resemblyzer "setuptools<81"   # resemblyzer's webrtcvad dep needs pkg_resources

Output dim is 256, matching `configs/stage_b.yaml`'s `model.speaker_emb_dim`.
"""
from __future__ import annotations

from typing import List

import numpy as np

SPEAKER_EMB_DIM = 256
_encoder_singleton = None


def _get_voice_encoder():
    global _encoder_singleton
    if _encoder_singleton is None:
        from resemblyzer import VoiceEncoder  # local import: optional dependency

        _encoder_singleton = VoiceEncoder()
    return _encoder_singleton


def embed_speaker_from_wav(wav_path: str) -> np.ndarray:
    """Single reference utterance -> [256] speaker embedding."""
    from resemblyzer import preprocess_wav

    encoder = _get_voice_encoder()
    wav = preprocess_wav(wav_path)
    return encoder.embed_utterance(wav).astype(np.float32)


def embed_speaker_from_wavs(wav_paths: List[str]) -> np.ndarray:
    """Multiple reference utterances for the same speaker -> a single
    averaged [256] embedding (more robust than any one utterance; standard
    practice for speaker-conditioning in TTS). L2-renormalized after
    averaging.
    """
    from resemblyzer import preprocess_wav

    encoder = _get_voice_encoder()
    embeds = []
    for path in wav_paths:
        wav = preprocess_wav(path)
        embeds.append(encoder.embed_utterance(wav))
    mean_embed = np.mean(embeds, axis=0)
    mean_embed = mean_embed / (np.linalg.norm(mean_embed) + 1e-8)
    return mean_embed.astype(np.float32)


def build_speaker_embedding_table(speaker_to_wavs: dict, out_dir: str) -> dict:
    """speaker_to_wavs: {speaker_id: [wav_path, ...]} -> writes
    `<out_dir>/<speaker_id>.npy` for each speaker and returns
    {speaker_id: npy_path}, ready to plug into manifest
    `speaker_embedding_path` fields (see `react_tts/data/tts_dataset.py`).
    """
    import os

    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for speaker_id, wav_paths in speaker_to_wavs.items():
        embedding = embed_speaker_from_wavs(wav_paths)
        path = os.path.join(out_dir, f"{speaker_id}.npy")
        np.save(path, embedding)
        paths[speaker_id] = path
    return paths


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Build per-speaker reference embeddings from wav files.")
    parser.add_argument("--speaker_wavs_json", required=True, help='JSON file: {"speaker_id": ["a.wav", "b.wav"]}')
    parser.add_argument("--out_dir", default="data/speaker_embeddings")
    args = parser.parse_args()

    with open(args.speaker_wavs_json) as f:
        speaker_to_wavs = json.load(f)
    paths = build_speaker_embedding_table(speaker_to_wavs, args.out_dir)
    print(f"Wrote {len(paths)} speaker embeddings to {args.out_dir}")
