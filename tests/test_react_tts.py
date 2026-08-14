"""End-to-end tests: Stage A (planner) + Stage B (TTS backbone) wired
together via `ReactTTS`, including `style_source` switching and the
counterfactual-listener generation path (RQ3 probe). Tiny dims throughout
for speed; synthetic data only.
"""
import torch
from torch.utils.data import DataLoader

from react_tts.config import Config
from react_tts.data.tts_dataset import TTSDialogueDataset
from react_tts.models.react_tts import build_react_tts


def tiny_stage_a_config() -> Config:
    return Config(
        {
            "data": {
                "synthetic": True, "num_context_turns": 2, "max_text_len": 10, "reaction_num_frames": 4,
                "num_emotion_classes": 7, "tokenizer": "lite",
            },
            "model": {
                "text": {"vocab_size": 500, "hidden_dim": 16, "num_layers": 2, "num_heads": 2, "ffn_dim": 32, "max_turns": 6, "dropout": 0.1},
                "reaction": {
                    "face_feat_dim": 24, "hidden_dim": 16, "num_layers": 1, "num_heads": 2, "ffn_dim": 32, "dropout": 0.1,
                    "use_listener_face": True, "temporal_face": True, "use_reaction_delta": True, "random_listener": False,
                },
                "fusion": {"hidden_dim": 16, "num_heads": 2, "dropout": 0.1},
                "style": {"emotion_hidden": 16, "vad_dim": 3, "prosody_dim": 4},
            },
        }
    )


def tiny_stage_b_config() -> Config:
    return Config(
        {
            "data": {"n_mels": 8, "max_text_len": 12, "max_mel_len": 32},
            "model": {
                "phoneme_vocab_size": 50,
                "n_mels": 8,
                "speaker_emb_dim": 16,
                "style_emb_dim": 16,  # must match stage_a fusion.hidden_dim
                "encoder": {"hidden_dim": 16, "num_layers": 2, "num_heads": 2, "ffn_dim": 32, "dropout": 0.1},
                "duration_predictor": {"hidden_dim": 16, "kernel_size": 3, "num_layers": 2, "dropout": 0.1},
                "decoder": {
                    "hidden_dim": 8, "num_res_blocks": 1, "channel_mults": [1, 2], "diffusion_steps": 4,
                    "beta_min": 0.05, "beta_max": 20.0,
                },
            },
        }
    )


def _build_model_and_batch(batch_size=2):
    cfg_a, cfg_b = tiny_stage_a_config(), tiny_stage_b_config()
    ds = TTSDialogueDataset(
        None,
        dialogue_kwargs=dict(
            tokenizer_spec=cfg_a.data.tokenizer, vocab_size=cfg_a.model.text.vocab_size,
            num_context_turns=cfg_a.data.num_context_turns,
            max_text_len=cfg_a.data.max_text_len, reaction_num_frames=cfg_a.data.reaction_num_frames,
            face_feat_dim=cfg_a.model.reaction.face_feat_dim,
        ),
        tts_kwargs=dict(
            n_mels=cfg_b.model.n_mels, max_text_len=cfg_b.data.max_text_len, max_mel_len=cfg_b.data.max_mel_len,
            speaker_emb_dim=cfg_b.model.speaker_emb_dim, phoneme_vocab_size=cfg_b.model.phoneme_vocab_size,
        ),
        synthetic=True, synthetic_size=8,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    batch = next(iter(loader))
    model = build_react_tts(cfg_a, cfg_b)
    return model, batch, cfg_b


def test_forward_train_predicted_style():
    model, batch, cfg_b = _build_model_and_batch()
    out = model.forward_train(batch, style_source="predicted")
    assert torch.isfinite(out["diffusion_loss"])
    assert out["style_out"] is not None
    assert out["style_embedding"].shape == (batch["phoneme_ids"].size(0), cfg_b.model.style_emb_dim)


def test_forward_train_ground_truth_style():
    model, batch, cfg_b = _build_model_and_batch()
    batch["target_style_embedding"] = torch.randn(batch["phoneme_ids"].size(0), cfg_b.model.style_emb_dim)
    out = model.forward_train(batch, style_source="ground_truth")
    assert out["style_out"] is None
    assert torch.isfinite(out["diffusion_loss"])


def test_infer_shapes():
    model, batch, cfg_b = _build_model_and_batch(batch_size=1)
    model.eval()
    out = model.infer(batch, style_source="predicted", num_steps=2)
    assert out["mel"].shape[1] == cfg_b.model.n_mels


def test_counterfactual_generation_runs():
    model, batch, _ = _build_model_and_batch(batch_size=1)
    model.eval()
    neg = {"face_features": batch["cf_negative_face_features"], "frame_mask": batch["cf_negative_frame_mask"]}
    pos = {"face_features": batch["cf_positive_face_features"], "frame_mask": batch["cf_positive_frame_mask"]}
    out = model.counterfactual(batch, neg, pos, num_steps=2)
    assert "negative" in out and "positive" in out
    assert out["negative"]["mel"].shape == out["positive"]["mel"].shape


def test_freeze_style_predictor_no_grad():
    cfg_a, cfg_b = tiny_stage_a_config(), tiny_stage_b_config()
    model = build_react_tts(cfg_a, cfg_b, freeze_style_predictor=True)
    for p in model.style_predictor.parameters():
        assert not p.requires_grad
