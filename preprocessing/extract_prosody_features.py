"""Extract F0 / energy / speaking-rate prosody features from a target
utterance's audio, with speaker-level normalization (design doc §13,
"F0_norm = (F0 - mu_speaker) / sigma_speaker").

Two-pass usage:
    1. `extract_raw_prosody` for every utterance -> collect per-speaker stats.
    2. `SpeakerNormalizer.fit(...)` then `.normalize(...)` per utterance.

Requires `librosa`, `pyworld`, `soundfile` (optional deps, only needed here).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass
class RawProsody:
    f0_mean: float
    f0_std: float
    energy: float
    speaking_rate: float  # words (or syllables) per second


def extract_raw_prosody(wav_path: str, num_words: int, sample_rate: int = 22050) -> RawProsody:
    import librosa
    import pyworld as pw

    y, sr = librosa.load(wav_path, sr=sample_rate)
    duration = max(len(y) / sr, 1e-3)

    y64 = y.astype(np.float64)
    f0, t = pw.dio(y64, sr, frame_period=10.0)
    f0 = pw.stonemask(y64, f0, t, sr)
    voiced = f0[f0 > 0]
    f0_mean = float(voiced.mean()) if len(voiced) > 0 else 0.0
    f0_std = float(voiced.std()) if len(voiced) > 0 else 0.0

    energy = float(np.sqrt(np.mean(y ** 2)))
    speaking_rate = num_words / duration

    return RawProsody(f0_mean=f0_mean, f0_std=f0_std, energy=energy, speaking_rate=speaking_rate)


@dataclass
class SpeakerNormalizer:
    _stats: Dict[str, dict] = field(default_factory=dict)

    def fit(self, per_speaker_values: Dict[str, List[RawProsody]]) -> None:
        for speaker_id, values in per_speaker_values.items():
            f0_means = np.array([v.f0_mean for v in values if v.f0_mean > 0])
            energies = np.array([v.energy for v in values])
            rates = np.array([v.speaking_rate for v in values])
            self._stats[speaker_id] = {
                "f0_mu": float(f0_means.mean()) if len(f0_means) else 0.0,
                "f0_sigma": float(f0_means.std() + 1e-6) if len(f0_means) else 1.0,
                "energy_mu": float(energies.mean()),
                "energy_sigma": float(energies.std() + 1e-6),
                "rate_mu": float(rates.mean()),
                "rate_sigma": float(rates.std() + 1e-6),
            }

    def normalize(self, speaker_id: str, prosody: RawProsody) -> List[float]:
        s = self._stats.get(speaker_id)
        if s is None:
            # unseen speaker: fall back to un-normalized (paper should warn/skip these)
            return [prosody.f0_mean, prosody.f0_std, prosody.energy, prosody.speaking_rate]
        f0_mean_norm = (prosody.f0_mean - s["f0_mu"]) / s["f0_sigma"]
        energy_norm = (prosody.energy - s["energy_mu"]) / s["energy_sigma"]
        rate_norm = (prosody.speaking_rate - s["rate_mu"]) / s["rate_sigma"]
        # f0_std left un-normalized (already a within-utterance variability measure)
        return [f0_mean_norm, prosody.f0_std, energy_norm, rate_norm]


def build_speaker_normalizer(utterances: List[dict]) -> SpeakerNormalizer:
    """utterances: [{"speaker_id": ..., "wav_path": ..., "num_words": ...}, ...]."""
    per_speaker: Dict[str, List[RawProsody]] = defaultdict(list)
    for u in utterances:
        raw = extract_raw_prosody(u["wav_path"], u["num_words"])
        per_speaker[u["speaker_id"]].append(raw)
    normalizer = SpeakerNormalizer()
    normalizer.fit(per_speaker)
    return normalizer
