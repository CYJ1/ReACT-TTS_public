"""Shape/forward-pass tests for the Stage B TTS backbone, on synthetic data
with deliberately tiny dimensions (fast diffusion U-Net, few mel frames) so
the full training-loss + sampling code paths are exercised quickly.
"""
import torch
from torch.utils.data import DataLoader

from react_tts.config import Config
from react_tts.data.tts_dataset import TTSDataset
from react_tts.tts.backbone import TTSBackbone


def tiny_stage_b_config() -> Config:
    return Config(
        {
            "data": {"n_mels": 8, "max_text_len": 12, "max_mel_len": 32},
            "model": {
                "phoneme_vocab_size": 50,
                "n_mels": 8,
                "speaker_emb_dim": 16,
                "style_emb_dim": 16,
                "encoder": {"hidden_dim": 16, "num_layers": 2, "num_heads": 2, "ffn_dim": 32, "dropout": 0.1},
                "duration_predictor": {"hidden_dim": 16, "kernel_size": 3, "num_layers": 2, "dropout": 0.1},
                "decoder": {
                    "hidden_dim": 8, "num_res_blocks": 1, "channel_mults": [1, 2], "diffusion_steps": 4,
                    "beta_min": 0.05, "beta_max": 20.0,
                },
            },
        }
    )


def _build_backbone_and_batch(batch_size=2):
    cfg = tiny_stage_b_config()
    ds = TTSDataset(
        None, n_mels=cfg.data.n_mels, max_text_len=cfg.data.max_text_len, max_mel_len=cfg.data.max_mel_len,
        speaker_emb_dim=cfg.model.speaker_emb_dim, synthetic=True, synthetic_size=8,
        phoneme_vocab_size=cfg.model.phoneme_vocab_size,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    batch = next(iter(loader))

    m = cfg.model
    backbone = TTSBackbone(
        phoneme_vocab_size=m.phoneme_vocab_size, n_mels=m.n_mels, speaker_emb_dim=m.speaker_emb_dim,
        style_emb_dim=m.style_emb_dim, encoder_cfg=m.encoder, duration_cfg=m.duration_predictor, decoder_cfg=m.decoder,
    )
    speaker_emb = batch["speaker_embedding"]
    style_emb = torch.randn(batch_size, m.style_emb_dim)
    return backbone, batch, speaker_emb, style_emb, cfg


def test_forward_train_losses_finite():
    backbone, batch, speaker_emb, style_emb, _ = _build_backbone_and_batch()
    out = backbone.forward_train(
        phoneme_ids=batch["phoneme_ids"], phoneme_mask=batch["phoneme_mask"], durations=batch["durations"],
        target_mel=batch["target_mel"], mel_mask=batch["mel_mask"], speaker_emb=speaker_emb, style_emb=style_emb,
    )
    assert torch.isfinite(out["diffusion_loss"])
    assert torch.isfinite(out["duration_loss"])
    assert out["diffusion_loss"].item() >= 0


def test_infer_produces_correctly_shaped_mel():
    backbone, batch, speaker_emb, style_emb, cfg = _build_backbone_and_batch()
    backbone.eval()
    out = backbone.infer(
        phoneme_ids=batch["phoneme_ids"], phoneme_mask=batch["phoneme_mask"],
        speaker_emb=speaker_emb, style_emb=style_emb, num_steps=2,
    )
    mel = out["mel"]
    assert mel.dim() == 3
    assert mel.size(0) == batch["phoneme_ids"].size(0)
    assert mel.size(1) == cfg.model.n_mels
    assert torch.isfinite(mel).all()


def test_different_style_embeddings_change_output():
    backbone, batch, speaker_emb, _, cfg = _build_backbone_and_batch(batch_size=1)
    backbone.eval()
    torch.manual_seed(0)
    style_a = torch.randn(1, cfg.model.style_emb_dim)
    style_b = torch.randn(1, cfg.model.style_emb_dim)

    torch.manual_seed(123)
    out_a = backbone.infer(batch["phoneme_ids"], batch["phoneme_mask"], speaker_emb, style_a, num_steps=2)
    torch.manual_seed(123)
    out_b = backbone.infer(batch["phoneme_ids"], batch["phoneme_mask"], speaker_emb, style_b, num_steps=2)

    # style also shapes duration, so mel lengths can legitimately differ; compare on the shared prefix
    min_len = min(out_a["mel"].size(-1), out_b["mel"].size(-1))
    same_duration = out_a["mel"].size(-1) == out_b["mel"].size(-1)
    assert not same_duration or not torch.allclose(out_a["mel"][..., :min_len], out_b["mel"][..., :min_len])
