import torch

from react_tts.losses import counterfactual_losses, prosody_loss, style_consistency_loss, vad_loss, weighted_sum


def test_style_consistency_loss_zero_for_identical():
    x = torch.randn(4, 16)
    loss = style_consistency_loss(x, x)
    assert loss.item() < 1e-5


def test_vad_loss_zero_for_identical():
    x = torch.randn(8, 3)
    loss = vad_loss(x, x)
    assert loss.item() < 1e-4


def test_prosody_loss_matches_l1():
    pred = torch.randn(4, 4)
    target = torch.randn(4, 4)
    expected = (pred - target).abs().mean()
    assert torch.isclose(prosody_loss(pred, target), expected)


def test_counterfactual_losses_shapes_and_bounds():
    content_neg = torch.randn(4, 16)
    content_pos = content_neg.clone()  # identical content -> distance ~0
    speaker_neg = torch.randn(4, 16)
    speaker_pos = speaker_neg.clone()
    style_neg = torch.randn(4, 16)
    style_pos = -style_neg  # maximally different direction -> large style distance

    out = counterfactual_losses(content_neg, content_pos, speaker_neg, speaker_pos, style_neg, style_pos, margin=0.2)
    assert out["counterfactual_content"].item() < 1e-4
    assert out["counterfactual_speaker"].item() < 1e-4
    assert out["counterfactual_separation"].item() < 1e-4  # style already well-separated beyond margin


def test_weighted_sum_respects_zero_weights():
    losses = {"a": torch.tensor(2.0), "b": torch.tensor(3.0)}
    total = weighted_sum(losses, {"a": 1.0, "b": 0.0})
    assert torch.isclose(total, torch.tensor(2.0))
