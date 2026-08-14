"""Score-based diffusion decoder, Grad-TTS lineage (Popov et al., 2021):
diffusion is defined toward a *data-dependent prior* `mu` (the length-regulated
encoder output projected to mel-channels), not toward standard Gaussian
noise. This is what lets the model start sampling from a linguistically/
prosodically informed prior rather than pure noise.

Forward SDE (closed form used for training):
    dX_t = 0.5 * beta_t * (mu - X_t) dt + sqrt(beta_t) dW_t,   t in [0, 1]
    beta_t = beta_min + t * (beta_max - beta_min)
    lambda(t) = integral_0^t beta_s ds = beta_min*t + 0.5*(beta_max-beta_min)*t^2
    X_t | X_0, mu ~ N( mu + (X_0 - mu) * exp(-0.5*lambda(t)),  (1 - exp(-lambda(t))) * I )

The network predicts the score s_theta(X_t, mu, t, cond) ~= grad_x log p(X_t | mu).
Training target: score = -noise / std_t  (weighted score matching, weight = var_t).

Reverse-time sampling (ancestral / Euler-Maruyama, N fixed steps as in the
Grad-TTS paper's Algorithm 1):
    dX_t = [0.5*beta_t*(mu - X_t) - beta_t*s_theta(X_t, mu, t, cond)] * dt
           + sqrt(beta_t * |dt|) * z          (z ~ N(0, I), integrating t: 1 -> 0)

Global conditioning (speaker identity + response style) is injected via
FiLM/AdaGN in every residual block of a small 2D U-Net operating on the mel
as a single-channel "image" [B, 1, n_mels, T], following Face-TTS/Grad-TTS's
treatment of the decoder as a 2D conv U-Net over (mel-bin, time).
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class TimestepEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: [B] float in [0, 1] -> [B, dim]."""
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / max(half - 1, 1)
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0) * 1000.0
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return self.mlp(emb)


class FiLMResBlock2D(nn.Module):
    """Same in/out channel count; channel transitions happen outside via 1x1 convs."""

    def __init__(self, channels: int, cond_dim: int, groups: int = 8):
        super().__init__()
        g = min(groups, channels)
        self.norm1 = nn.GroupNorm(g, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(g, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.film = nn.Linear(cond_dim, channels * 2)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift = self.film(cond).chunk(2, dim=-1)
        h = self.norm1(x)
        h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = self.conv1(self.act(h))
        h = self.norm2(h)
        h = self.conv2(self.act(h))
        return x + h


class ScoreUNet(nn.Module):
    def __init__(self, hidden_dim: int = 128, num_res_blocks: int = 2, channel_mults=(1, 2, 2), cond_dim: int = 256):
        super().__init__()
        self.time_embed = TimestepEmbedding(hidden_dim)
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim + hidden_dim, hidden_dim * 4), nn.SiLU(), nn.Linear(hidden_dim * 4, hidden_dim * 4)
        )
        self.in_conv = nn.Conv2d(2, hidden_dim, 3, padding=1)  # channels: [x_t, mu]
        self.num_levels = len(channel_mults)
        film_cond_dim = hidden_dim * 4

        self.down_blocks = nn.ModuleList()
        self.down_transitions = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        ch = hidden_dim
        for level, mult in enumerate(channel_mults):
            out_ch = hidden_dim * mult
            self.down_blocks.append(
                nn.ModuleList([FiLMResBlock2D(ch, film_cond_dim) for _ in range(num_res_blocks)])
            )
            self.down_transitions.append(nn.Conv2d(ch, out_ch, 1) if out_ch != ch else nn.Identity())
            ch = out_ch
            if level != self.num_levels - 1:
                self.downsamplers.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))
            else:
                self.downsamplers.append(nn.Identity())

        self.mid_blocks = nn.ModuleList([FiLMResBlock2D(ch, film_cond_dim) for _ in range(num_res_blocks)])

        self.up_blocks = nn.ModuleList()
        self.up_merges = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        for level in reversed(range(self.num_levels)):
            out_ch = hidden_dim * channel_mults[level]
            if level != self.num_levels - 1:
                self.upsamplers.append(nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1))
            else:
                self.upsamplers.append(nn.Identity())
            self.up_merges.append(nn.Conv2d(ch + out_ch, out_ch, 1))
            self.up_blocks.append(nn.ModuleList([FiLMResBlock2D(out_ch, film_cond_dim) for _ in range(num_res_blocks)]))
            ch = out_ch

        self.out_norm = nn.GroupNorm(min(8, ch), ch)
        self.out_conv = nn.Conv2d(ch, 1, 3, padding=1)

    def forward(self, x_t: torch.Tensor, mu: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """x_t, mu: [B, 1, n_mels, T]; t: [B]; cond: [B, cond_dim] -> score [B, 1, n_mels, T]."""
        time_emb = self.time_embed(t)
        film_cond = self.cond_mlp(torch.cat([cond, time_emb], dim=-1))

        h = self.in_conv(torch.cat([x_t, mu], dim=1))
        skips = []
        for blocks, transition, downsampler in zip(self.down_blocks, self.down_transitions, self.downsamplers):
            for block in blocks:
                h = block(h, film_cond)
            h = transition(h)
            skips.append(h)
            h = downsampler(h)

        for block in self.mid_blocks:
            h = block(h, film_cond)

        for blocks, merge, upsampler, skip in zip(self.up_blocks, self.up_merges, self.upsamplers, reversed(skips)):
            h = upsampler(h)
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = merge(torch.cat([h, skip], dim=1))
            for block in blocks:
                h = block(h, film_cond)

        h = self.out_conv(F.silu(self.out_norm(h)))
        return h


class DiffusionDecoder(nn.Module):
    def __init__(
        self,
        n_mels: int = 80,
        hidden_dim: int = 128,
        num_res_blocks: int = 2,
        channel_mults=(1, 2, 2),
        cond_dim: int = 256,
        diffusion_steps: int = 50,
        beta_min: float = 0.05,
        beta_max: float = 20.0,
    ):
        super().__init__()
        self.n_mels = n_mels
        self.diffusion_steps = diffusion_steps
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.net = ScoreUNet(hidden_dim, num_res_blocks, channel_mults, cond_dim)
        self._pad_multiple = 2 ** (len(channel_mults) - 1)

    def _lambda(self, t: torch.Tensor) -> torch.Tensor:
        return self.beta_min * t + 0.5 * (self.beta_max - self.beta_min) * t ** 2

    def _beta(self, t: torch.Tensor) -> torch.Tensor:
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def _pad_to_multiple(self, x: torch.Tensor):
        # x: [B, 1, n_mels, T]
        n_mels, T = x.shape[-2], x.shape[-1]
        pad_mels = (-n_mels) % self._pad_multiple
        pad_T = (-T) % self._pad_multiple
        x = F.pad(x, (0, pad_T, 0, pad_mels))
        return x, (n_mels, T)

    def compute_loss(self, x0: torch.Tensor, mu: torch.Tensor, mel_mask: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """x0, mu: [B, n_mels, T]; mel_mask: [B, T] bool; cond: [B, cond_dim]."""
        B = x0.size(0)
        device = x0.device
        t = torch.rand(B, device=device).clamp(min=1e-4)

        lam = self._lambda(t)[:, None, None]
        mean = mu + (x0 - mu) * torch.exp(-0.5 * lam)
        var = (1.0 - torch.exp(-lam)).clamp(min=1e-5)
        std = var.sqrt()

        noise = torch.randn_like(x0)
        x_t = mean + std * noise

        mask2d = mel_mask.unsqueeze(1).to(x0.dtype)  # [B, 1, T], broadcasts over the n_mels dim
        x_t_in, orig_shape = self._pad_to_multiple((x_t * mask2d).unsqueeze(1))
        mu_in, _ = self._pad_to_multiple((mu * mask2d).unsqueeze(1))
        pred_score = self.net(x_t_in, mu_in, t, cond)
        pred_score = pred_score[..., : orig_shape[0], : orig_shape[1]].squeeze(1)  # [B, n_mels, T]

        # weighted score matching: multiply residual by std (weight = var), as in Grad-TTS
        residual = pred_score * std + noise
        mask_full = mel_mask.unsqueeze(1).expand(-1, x0.size(1), -1).to(x0.dtype)
        loss = (residual ** 2 * mask_full).sum() / mask_full.sum().clamp(min=1.0)
        return loss

    @torch.no_grad()
    def sample(self, mu: torch.Tensor, mel_mask: torch.Tensor, cond: torch.Tensor, num_steps: int | None = None) -> torch.Tensor:
        """mu: [B, n_mels, T] prior (also used as the diffusion starting point, X_1 ~ N(mu, I))."""
        num_steps = num_steps or self.diffusion_steps
        x = mu + torch.randn_like(mu)

        dt = 1.0 / num_steps
        for i in reversed(range(num_steps)):
            t = torch.full((mu.size(0),), (i + 1) / num_steps, device=mu.device)
            beta_t = self._beta(t)[:, None, None]

            x_in, orig_shape = self._pad_to_multiple(x.unsqueeze(1))
            mu_in, _ = self._pad_to_multiple(mu.unsqueeze(1))
            score = self.net(x_in, mu_in, t, cond)
            score = score[..., : orig_shape[0], : orig_shape[1]].squeeze(1)

            drift = 0.5 * beta_t * (mu - x) - beta_t * score
            noise = torch.randn_like(x) if i > 0 else torch.zeros_like(x)
            x = x + drift * dt + torch.sqrt(beta_t * dt) * noise

        return x * mel_mask.unsqueeze(1).to(x.dtype)
