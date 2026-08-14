import torch

from react_tts.data.dialogue_dataset import DialogueStyleDataset
from react_tts.data.tts_dataset import TTSDataset, TTSDialogueDataset


def test_dialogue_dataset_shapes():
    ds = DialogueStyleDataset(None, num_context_turns=3, max_text_len=10, reaction_num_frames=6, face_feat_dim=32, synthetic=True, synthetic_size=5)
    item = ds[0]
    assert item["token_ids"].shape == (4, 10)  # num_context_turns + 1 target turn
    assert item["face_features"].shape == (6, 32)
    assert item["is_target"].sum().item() == 1
    assert item["listener_last_emotion_id"].item() >= -1


def test_tts_dataset_shapes():
    ds = TTSDataset(None, n_mels=8, max_text_len=15, max_mel_len=40, speaker_emb_dim=16, synthetic=True, synthetic_size=5, phoneme_vocab_size=30)
    item = ds[0]
    assert item["target_mel"].shape == (8, 40)
    assert item["phoneme_ids"].shape == (15,)
    assert item["mel_mask"].dtype == torch.bool


def test_tts_dialogue_dataset_merges_fields_and_counterfactual_pair():
    ds = TTSDialogueDataset(
        None,
        dialogue_kwargs=dict(num_context_turns=2, max_text_len=8, reaction_num_frames=4, face_feat_dim=16),
        tts_kwargs=dict(n_mels=8, max_text_len=10, max_mel_len=20, speaker_emb_dim=16, phoneme_vocab_size=20),
        synthetic=True, synthetic_size=5,
    )
    item = ds[0]
    for key in ["token_ids", "face_features", "phoneme_ids", "target_mel", "speaker_embedding"]:
        assert key in item
    assert "cf_negative_face_features" in item and "cf_positive_face_features" in item
    assert item["cf_negative_face_features"].shape == item["cf_positive_face_features"].shape
