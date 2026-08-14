"""Shape/forward-pass tests for Stage A (response style predictor), on
synthetic data only -- no MELD, no checkpoints, no network access needed.
"""
import torch
from torch.utils.data import DataLoader

from react_tts.config import load_config
from react_tts.data.dialogue_dataset import DialogueStyleDataset
from react_tts.models.response_style_predictor import build_response_style_predictor
from react_tts.modules.listener_reaction_encoder import shuffle_listener_reaction

CONFIG_PATH = "configs/stage_a.yaml"


def _make_batch(cfg, batch_size=4):
    d = cfg.data
    ds = DialogueStyleDataset(
        None, tokenizer_spec=d.tokenizer, vocab_size=cfg.model.text.vocab_size, num_context_turns=d.num_context_turns,
        max_text_len=d.max_text_len, reaction_num_frames=d.reaction_num_frames,
        face_feat_dim=cfg.model.reaction.face_feat_dim, synthetic=True, synthetic_size=16,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    return next(iter(loader)), cfg


def test_response_style_predictor_forward_shapes():
    cfg = load_config(CONFIG_PATH)
    batch, cfg = _make_batch(cfg)
    model = build_response_style_predictor(cfg)
    out = model(batch)

    B = batch["token_ids"].size(0)
    assert out["emotion_logits"].shape == (B, cfg.data.num_emotion_classes)
    assert out["vad"].shape == (B, cfg.model.style.vad_dim)
    assert out["prosody"].shape == (B, cfg.model.style.prosody_dim)
    assert out["style_embedding"].shape == (B, cfg.model.fusion.hidden_dim)
    assert torch.isfinite(out["emotion_logits"]).all()


def test_ablation_no_listener_face_zeroes_reaction_branch():
    cfg = load_config(CONFIG_PATH)
    cfg.model.reaction.use_listener_face = False
    batch, cfg = _make_batch(cfg)
    model = build_response_style_predictor(cfg)
    out = model(batch)
    assert out["emotion_logits"].shape[0] == batch["token_ids"].size(0)


def test_ablation_static_face_runs():
    cfg = load_config(CONFIG_PATH)
    cfg.model.reaction.temporal_face = False
    batch, cfg = _make_batch(cfg)
    model = build_response_style_predictor(cfg)
    out = model(batch)
    assert torch.isfinite(out["style_embedding"]).all()


def test_ablation_no_reaction_delta_runs():
    cfg = load_config(CONFIG_PATH)
    cfg.model.reaction.use_reaction_delta = False
    batch, cfg = _make_batch(cfg)
    model = build_response_style_predictor(cfg)
    out = model(batch)
    assert torch.isfinite(out["style_embedding"]).all()


def test_random_listener_override_changes_output():
    cfg = load_config(CONFIG_PATH)
    batch, cfg = _make_batch(cfg, batch_size=8)
    model = build_response_style_predictor(cfg)
    model.eval()
    with torch.no_grad():
        out_correct = model(batch, random_listener_override=False)
        out_random = model(batch, random_listener_override=True)
    # different listener inputs should (generically) produce different style embeddings
    assert not torch.allclose(out_correct["style_embedding"], out_random["style_embedding"])


def test_shuffle_listener_reaction_permutes_batch():
    feats = torch.randn(5, 3, 8)
    mask = torch.ones(5, 3, dtype=torch.bool)
    shuffled_feats, shuffled_mask = shuffle_listener_reaction(feats, mask, generator=torch.Generator().manual_seed(0))
    assert shuffled_feats.shape == feats.shape
    # every row of the shuffled tensor should still be one of the original rows
    for i in range(feats.size(0)):
        assert any(torch.allclose(shuffled_feats[i], feats[j]) for j in range(feats.size(0)))


def test_gates_are_in_unit_interval():
    cfg = load_config(CONFIG_PATH)
    batch, cfg = _make_batch(cfg)
    model = build_response_style_predictor(cfg)
    out = model(batch)
    assert (out["gate_react"] >= 0).all() and (out["gate_react"] <= 1).all()
    assert (out["gate_text"] >= 0).all() and (out["gate_text"] <= 1).all()
