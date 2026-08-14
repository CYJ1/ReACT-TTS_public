"""Text -> phoneme-id frontend for inference.

Prefers `g2p_en` (CMUdict-based English G2P) if installed; falls back to a
deterministic character-level tokenizer otherwise, so `scripts/inference.py`
runs without an extra dependency. Swap in a stronger phonemizer (and retrain
`phoneme_vocab_size` accordingly) for real experiments -- this default is
not linguistically accurate.
"""
from __future__ import annotations

from typing import List

_ARPABET = [
    "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH", "EH", "ER", "EY", "F", "G", "HH", "IH", "IY",
    "JH", "K", "L", "M", "N", "NG", "OW", "OY", "P", "R", "S", "SH", "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH",
]
_PUNCT = [" ", ",", ".", "!", "?", "'", "-"]
PAD_ID = 0


class TextFrontend:
    def __init__(self):
        self._vocab = {sym: i + 1 for i, sym in enumerate(_ARPABET + _PUNCT)}
        self.vocab_size = len(self._vocab) + 1  # + pad
        try:
            from g2p_en import G2p

            self._g2p = G2p()
        except ImportError:
            self._g2p = None

    def text_to_phoneme_ids(self, text: str) -> List[int]:
        if self._g2p is not None:
            symbols = self._g2p(text)
            ids = []
            for sym in symbols:
                stripped = "".join(c for c in sym if not c.isdigit())  # drop stress markers
                ids.append(self._vocab.get(stripped, self._vocab.get(sym, PAD_ID)))
            return ids
        # character-level fallback
        return [self._vocab.get(ch, PAD_ID) for ch in text.lower() if ch.isalpha() or ch in _PUNCT]
