"""Stage B: the expressive TTS backbone itself (Face-TTS / Grad-TTS lineage),
independent of the conversational planner. Conditioning is deliberately
*not* concatenated into one vector (see README "Conditioning placement"):

  - z_speaker (identity/timbre)         -> global AdaLN/FiLM cond, everywhere in the decoder
  - z_style   (predicted response style) -> global AdaLN/FiLM cond, everywhere in the decoder
  - z_prosody (F0/energy/rate targets)   -> NOT used by this simplified backbone's duration
    predictor directly; exposed for an auxiliary prosody loss instead (see
    react_tts/losses.py). A full pitch/energy predictor (as in FastSpeech2)
    can be added the same way as the duration predictor if needed.
  - dialogue context                     -> already folded into z_style by Stage A;
    this backbone only sees the target text + speaker + style.
"""
from __future__ import annotations

import torch
from torch import nn

from react_tts.tts.diffusion_decoder import DiffusionDecoder
from react_tts.tts.duration_predictor import DurationPredictor
from react_tts.tts.length_regulator import LengthRegulator
from react_tts.tts.phoneme_encoder import PhonemeEncoder


class TTSBackbone(nn.Module):
    def __init__(
        self,
        phoneme_vocab_size: int,
        n_mels: int = 80,
        speaker_emb_dim: int = 256,
        style_emb_dim: int = 256,
        encoder_cfg=None,
        duration_cfg=None,
        decoder_cfg=None,
    ):
        super().__init__()
        encoder_cfg = encoder_cfg or {}
        duration_cfg = duration_cfg or {}
        decoder_cfg = decoder_cfg or {}

        hidden_dim = encoder_cfg.get("hidden_dim", 256)
        self.encoder = PhonemeEncoder(
            vocab_size=phoneme_vocab_size,
            hidden_dim=hidden_dim,
            num_layers=encoder_cfg.get("num_layers", 4),
            num_heads=encoder_cfg.get("num_heads", 4),
            ffn_dim=encoder_cfg.get("ffn_dim", 1024),
            dropout=encoder_cfg.get("dropout", 0.1),
        )

        cond_dim = speaker_emb_dim + style_emb_dim
        self.cond_dim = cond_dim

        self.duration_predictor = DurationPredictor(
            hidden_dim=duration_cfg.get("hidden_dim", hidden_dim),
            kernel_size=duration_cfg.get("kernel_size", 3),
            num_layers=duration_cfg.get("num_layers", 2),
            dropout=duration_cfg.get("dropout", 0.5),
            cond_dim=cond_dim,
        )
        self.length_regulator = LengthRegulator()
        self.mu_proj = nn.Linear(hidden_dim, n_mels)

        self.decoder = DiffusionDecoder(
            n_mels=n_mels,
            hidden_dim=decoder_cfg.get("hidden_dim", 128),
            num_res_blocks=decoder_cfg.get("num_res_blocks", 2),
            channel_mults=tuple(decoder_cfg.get("channel_mults", [1, 2, 2])),
            cond_dim=cond_dim,
            diffusion_steps=decoder_cfg.get("diffusion_steps", 50),
            beta_min=decoder_cfg.get("beta_min", 0.05),
            beta_max=decoder_cfg.get("beta_max", 20.0),
        )

    def _cond(self, speaker_emb: torch.Tensor, style_emb: torch.Tensor) -> torch.Tensor:
        return torch.cat([speaker_emb, style_emb], dim=-1)

    def forward_train(
        self,
        phoneme_ids: torch.Tensor,
        phoneme_mask: torch.Tensor,
        durations: torch.Tensor,
        target_mel: torch.Tensor,     # [B, n_mels, T]
        mel_mask: torch.Tensor,       # [B, T]
        speaker_emb: torch.Tensor,
        style_emb: torch.Tensor,
    ) -> dict:
        cond = self._cond(speaker_emb, style_emb)
        h_ph = self.encoder(phoneme_ids, phoneme_mask)

        log_dur_pred = self.duration_predictor(h_ph, phoneme_mask, cond)
        with torch.no_grad():
            dur_target = torch.log(durations.clamp(min=1).to(h_ph.dtype))
        duration_loss = ((log_dur_pred - dur_target) ** 2 * phoneme_mask.to(h_ph.dtype)).sum()
        duration_loss = duration_loss / phoneme_mask.sum().clamp(min=1.0)

        expanded, expanded_mask = self.length_regulator(h_ph, durations, phoneme_mask, max_len=target_mel.size(-1))
        mu = self.mu_proj(expanded).transpose(1, 2)  # [B, n_mels, T]

        diff_loss = self.decoder.compute_loss(target_mel, mu, mel_mask, cond)

        return {"diffusion_loss": diff_loss, "duration_loss": duration_loss, "mu": mu}

    @torch.no_grad()
    def infer(
        self,
        phoneme_ids: torch.Tensor,
        phoneme_mask: torch.Tensor,
        speaker_emb: torch.Tensor,
        style_emb: torch.Tensor,
        num_steps: int | None = None,
        length_scale: float = 1.0,
    ) -> dict:
        cond = self._cond(speaker_emb, style_emb)
        h_ph = self.encoder(phoneme_ids, phoneme_mask)
        log_dur_pred = self.duration_predictor(h_ph, phoneme_mask, cond)
        durations = torch.exp(log_dur_pred) * length_scale
        durations = durations.round().clamp(min=1) * phoneme_mask.to(durations.dtype)

        expanded, mel_mask = self.length_regulator(h_ph, durations, phoneme_mask)
        mu = self.mu_proj(expanded).transpose(1, 2)  # [B, n_mels, T]

        mel = self.decoder.sample(mu, mel_mask, cond, num_steps=num_steps)
        return {"mel": mel, "mel_mask": mel_mask, "durations": durations}
