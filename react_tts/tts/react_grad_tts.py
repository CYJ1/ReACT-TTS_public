from __future__ import annotations

import math
import random

import torch
import torch.nn as nn

from react_tts.tts.grad_tts import monotonic_align
from react_tts.tts.grad_tts.text_encoder import TextEncoder
from react_tts.tts.grad_tts.diffusion import Diffusion
from react_tts.tts.grad_tts.utils import (
    sequence_mask,
    generate_path,
    duration_loss,
    fix_len_compatibility,
)


class ReactGradTTS(nn.Module):
    """
    Grad-TTS acoustic backbone for ReACT-TTS.

    Conditioning:
        speaker_embedding : [B, 256]
        style_embedding   : [B, 256]
        --------------------------------
        cond              : [B, 512]

    The same global condition is used by both the text encoder and
    diffusion decoder, following the conditioning structure inherited
    from Face-TTS / Grad-TTS.
    """

    def __init__(
        self,
        n_vocab: int,
        n_feats: int = 128,
        speaker_emb_dim: int = 256,
        style_emb_dim: int = 256,
        n_enc_channels: int = 192,
        filter_channels: int = 768,
        filter_channels_dp: int = 256,
        n_heads: int = 2,
        n_enc_layers: int = 6,
        enc_kernel: int = 3,
        enc_dropout: float = 0.0,
        window_size: int = 4,
        dec_dim: int = 64,
        beta_min: float = 0.05,
        beta_max: float = 20.0,
        pe_scale: float = 1000.0,
    ):
        super().__init__()

        self.n_vocab = n_vocab
        self.n_feats = n_feats
        self.speaker_emb_dim = speaker_emb_dim
        self.style_emb_dim = style_emb_dim
        self.cond_dim = speaker_emb_dim + style_emb_dim

        self.encoder = TextEncoder(
            n_vocab=n_vocab,
            n_feats=n_feats,
            n_channels=n_enc_channels,
            filter_channels=filter_channels,
            filter_channels_dp=filter_channels_dp,
            n_heads=n_heads,
            n_layers=n_enc_layers,
            kernel_size=enc_kernel,
            p_dropout=enc_dropout,
            window_size=window_size,
            spk_emb_dim=self.cond_dim,
            multi_spks=1,
        )

        # perceptual_loss=False because the Face-TTS SyncNet perceptual
        # objective is not part of ReACT-TTS.
        self.decoder = Diffusion(
            n_feats=n_feats,
            dim=dec_dim,
            multi_spks=1,
            spk_emb_dim=self.cond_dim,
            beta_min=beta_min,
            beta_max=beta_max,
            pe_scale=pe_scale,
            config={"perceptual_loss": False},
        )

    def make_condition(
        self,
        speaker_embedding: torch.Tensor,
        style_embedding: torch.Tensor,
    ) -> torch.Tensor:
        if speaker_embedding.ndim != 2:
            raise ValueError(
                f"speaker_embedding must be [B,D], got {speaker_embedding.shape}"
            )
        if style_embedding.ndim != 2:
            raise ValueError(
                f"style_embedding must be [B,D], got {style_embedding.shape}"
            )

        if speaker_embedding.size(-1) != self.speaker_emb_dim:
            raise ValueError(
                f"Expected speaker dim {self.speaker_emb_dim}, "
                f"got {speaker_embedding.size(-1)}"
            )

        if style_embedding.size(-1) != self.style_emb_dim:
            raise ValueError(
                f"Expected style dim {self.style_emb_dim}, "
                f"got {style_embedding.size(-1)}"
            )

        return torch.cat(
            [speaker_embedding, style_embedding],
            dim=-1,
        )

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        speaker_embedding: torch.Tensor,
        style_embedding: torch.Tensor,
        n_timesteps: int = 10,
        temperature: float = 1.0,
        stoc: bool = False,
        length_scale: float = 1.0,
    ):
        """
        Inference.

        x          : [B, T_text]
        x_lengths  : [B]
        outputs mel: [B, n_feats, T_mel]
        """

        cond = self.make_condition(
            speaker_embedding,
            style_embedding,
        )

        mu_x, logw, x_mask = self.encoder(
            x,
            x_lengths,
            cond,
        )

        w = torch.exp(logw) * x_mask
        w_ceil = torch.ceil(w) * length_scale

        y_lengths = torch.clamp_min(
            torch.sum(w_ceil, [1, 2]),
            1,
        ).long()

        y_max_length = int(y_lengths.max())
        y_max_length_ = fix_len_compatibility(y_max_length)

        y_mask = sequence_mask(
            y_lengths,
            y_max_length_,
        ).unsqueeze(1).to(x_mask.dtype)

        attn_mask = (
            x_mask.unsqueeze(-1)
            * y_mask.unsqueeze(2)
        )

        attn = generate_path(
            w_ceil.squeeze(1),
            attn_mask.squeeze(1),
        ).unsqueeze(1)

        mu_y = torch.matmul(
            attn.squeeze(1).transpose(1, 2),
            mu_x.transpose(1, 2),
        )
        mu_y = mu_y.transpose(1, 2)

        z = (
            mu_y
            + torch.randn_like(mu_y) / temperature
        )

        decoder_outputs = self.decoder(
            z,
            y_mask,
            mu_y,
            n_timesteps,
            stoc,
            cond,
        )

        decoder_outputs = [
            output[:, :, :y_max_length]
            for output in decoder_outputs
        ]

        return {
            "mel": decoder_outputs[-1],
            "decoder_trajectory": decoder_outputs,
            "encoder_mel": mu_y[:, :, :y_max_length],
            "attention": attn[:, :, :, :y_max_length],
            "durations": w_ceil,
            "mel_lengths": y_lengths,
            "condition": cond,
        }

    def compute_loss(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        y: torch.Tensor,
        y_lengths: torch.Tensor,
        speaker_embedding: torch.Tensor,
        style_embedding: torch.Tensor,
        out_size: int | None = None,
    ):
        """
        Grad-TTS training objective.

        No external phoneme durations are needed:
        MAS estimates the monotonic text-mel alignment online.
        """

        cond = self.make_condition(
            speaker_embedding,
            style_embedding,
        )

        mu_x, logw, x_mask = self.encoder(
            x,
            x_lengths,
            cond,
        )

        y_max_length = y.shape[-1]

        y_mask = sequence_mask(
            y_lengths,
            y_max_length,
        ).unsqueeze(1).to(x_mask.dtype)

        attn_mask = (
            x_mask.unsqueeze(-1)
            * y_mask.unsqueeze(2)
        )

        # -----------------------------------------------------
        # Monotonic Alignment Search
        # -----------------------------------------------------
        with torch.no_grad():
            const = (
                -0.5
                * math.log(2 * math.pi)
                * self.n_feats
            )

            factor = -0.5 * torch.ones(
                mu_x.shape,
                dtype=mu_x.dtype,
                device=mu_x.device,
            )

            y_square = torch.matmul(
                factor.transpose(1, 2),
                y ** 2,
            )

            y_mu_double = torch.matmul(
                2.0 * (factor * mu_x).transpose(1, 2),
                y,
            )

            mu_square = torch.sum(
                factor * (mu_x ** 2),
                1,
            ).unsqueeze(-1)

            log_prior = (
                y_square
                - y_mu_double
                + mu_square
                + const
            )

            attn = monotonic_align.maximum_path(
                log_prior,
                attn_mask.squeeze(1),
            ).detach()

        logw_target = torch.log(
            1e-8 + torch.sum(
                attn.unsqueeze(1),
                -1,
            )
        ) * x_mask

        dur_loss = duration_loss(
            logw,
            logw_target,
            x_lengths,
        )

        # -----------------------------------------------------
        # Optional random mel crop for memory-efficient training
        # -----------------------------------------------------
        if out_size is not None:
            out_size = int(out_size)

            max_offset = (
                y_lengths - out_size
            ).clamp(min=0)

            out_offset = []

            for max_off in max_offset.tolist():
                if max_off > 0:
                    out_offset.append(
                        random.randint(0, int(max_off))
                    )
                else:
                    out_offset.append(0)

            out_offset = torch.tensor(
                out_offset,
                dtype=torch.long,
                device=y.device,
            )

            attn_cut = torch.zeros(
                attn.shape[0],
                attn.shape[1],
                out_size,
                dtype=attn.dtype,
                device=attn.device,
            )

            y_cut = torch.zeros(
                y.shape[0],
                self.n_feats,
                out_size,
                dtype=y.dtype,
                device=y.device,
            )

            y_cut_lengths = []

            for i in range(y.shape[0]):
                cut_len = min(
                    int(y_lengths[i]),
                    out_size,
                )

                lower = int(out_offset[i])
                upper = lower + cut_len

                y_cut[
                    i,
                    :,
                    :cut_len,
                ] = y[
                    i,
                    :,
                    lower:upper,
                ]

                attn_cut[
                    i,
                    :,
                    :cut_len,
                ] = attn[
                    i,
                    :,
                    lower:upper,
                ]

                y_cut_lengths.append(cut_len)

            y = y_cut
            attn = attn_cut

            y_lengths_cut = torch.tensor(
                y_cut_lengths,
                dtype=torch.long,
                device=y.device,
            )

            y_mask = sequence_mask(
                y_lengths_cut,
                out_size,
            ).unsqueeze(1).to(x_mask.dtype)

        # -----------------------------------------------------
        # Text prior expanded to mel length through MAS
        # -----------------------------------------------------
        mu_y = torch.matmul(
            attn.squeeze(1).transpose(1, 2),
            mu_x.transpose(1, 2),
        )
        mu_y = mu_y.transpose(1, 2)

        # Keep MAS-derived acoustic prior on the exact same
        # device/dtype as the target mel.
        mu_y = mu_y.to(device=y.device, dtype=y.dtype)
        y_mask = y_mask.to(device=y.device, dtype=y.dtype)

        # -----------------------------------------------------
        # Diffusion
        # -----------------------------------------------------
        diff_out = self.decoder.compute_loss(
            y,
            y_mask,
            mu_y,
            cond,
        )

        # perceptual_loss=False => (loss, xt)
        diff_loss = diff_out[0]

        # -----------------------------------------------------
        # Prior loss
        # -----------------------------------------------------
        prior_loss = torch.sum(
            0.5
            * (
                (y - mu_y) ** 2
                + math.log(2 * math.pi)
            )
            * y_mask
        )

        prior_loss = prior_loss / (
            torch.sum(y_mask).clamp(min=1.0)
            * self.n_feats
        )

        total_loss = (
            dur_loss
            + prior_loss
            + diff_loss
        )

        return {
            "loss": total_loss,
            "duration_loss": dur_loss,
            "prior_loss": prior_loss,
            "diffusion_loss": diff_loss,
            "attention": attn,
            "mu_y": mu_y,
            "condition": cond,
        }
