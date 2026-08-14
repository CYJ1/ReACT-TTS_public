"""Expand phoneme-level encoder outputs to frame-level using durations.

Grad-TTS/Glow-TTS normally learn durations via Monotonic Alignment Search
(MAS) against the target mel. Reproducing exact MAS (a dynamic-programming
alignment search, typically implemented in Cython/Numba) is out of scope for
this MVP; instead this module expects durations to already be available per
sample -- either extracted offline with a forced aligner (e.g. MFA) and
stored in the manifest, or the model's own predicted durations at inference
time. See `preprocessing/` and the `docs`/paper's Limitations section.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence


class LengthRegulator(nn.Module):
    def forward(self, h_ph: torch.Tensor, durations: torch.Tensor, phoneme_mask: torch.Tensor, max_len: int | None = None):
        """
        h_ph: [B, L, H], durations: [B, L] non-negative ints, phoneme_mask: [B, L] bool.
        Returns expanded [B, T, H] and mel_mask [B, T] (T = max frame length in batch, or max_len).
        """
        B = h_ph.size(0)
        durations = durations.clamp(min=0) * phoneme_mask.to(durations.dtype)
        expanded_list = []
        for b in range(B):
            dur_b = durations[b].round().long()
            if dur_b.sum() == 0:
                dur_b = dur_b.clone()
                dur_b[0] = 1  # avoid a fully-empty sequence
            expanded_list.append(torch.repeat_interleave(h_ph[b], dur_b, dim=0))

        lengths = torch.tensor([e.size(0) for e in expanded_list], device=h_ph.device)
        expanded = pad_sequence(expanded_list, batch_first=True)  # [B, T, H]
        if max_len is not None:
            if expanded.size(1) < max_len:
                pad = max_len - expanded.size(1)
                expanded = torch.nn.functional.pad(expanded, (0, 0, 0, pad))
            else:
                expanded = expanded[:, :max_len]
                lengths = lengths.clamp(max=max_len)

        T = expanded.size(1)
        mel_mask = torch.arange(T, device=h_ph.device).unsqueeze(0) < lengths.unsqueeze(1)
        return expanded, mel_mask
