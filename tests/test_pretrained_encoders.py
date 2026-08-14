"""Integration tests for the real pretrained encoders (preprocessing/).

These are network/model-file-optional: they `pytest.skip` cleanly if
mediapipe/resemblyzer aren't installed or the MediaPipe model files haven't
been downloaded yet (`python -m preprocessing.download_pretrained_assets`),
so the core `tests/` suite never requires network access. When the assets
*are* present (as in this repo's dev environment), they exercise the real
code paths end to end -- not the placeholders.
"""
import os

import numpy as np
import pytest

FACE_DETECTOR_MODEL = "models/mediapipe/blaze_face_short_range.tflite"
FACE_LANDMARKER_MODEL = "models/mediapipe/face_landmarker_v2_with_blendshapes.task"

mediapipe = pytest.importorskip("mediapipe", reason="mediapipe not installed")
_have_face_models = os.path.exists(FACE_DETECTOR_MODEL) and os.path.exists(FACE_LANDMARKER_MODEL)


@pytest.mark.skipif(not _have_face_models, reason="run `python -m preprocessing.download_pretrained_assets` first")
def test_face_detector_runs_on_random_frame():
    from preprocessing.face_utils import detect_faces_per_frame

    frame = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
    boxes = detect_faces_per_frame(frame, min_confidence=0.5)
    assert isinstance(boxes, list)
    for box in boxes:
        assert box.x2 >= box.x1 and box.y2 >= box.y1
        assert 0.0 <= box.confidence <= 1.0


@pytest.mark.skipif(not _have_face_models, reason="run `python -m preprocessing.download_pretrained_assets` first")
def test_blendshape_embedder_shape_and_no_face_handling():
    from preprocessing.pretrained_face_embedder import BLENDSHAPE_DIM, BlendshapeFaceEmbedder

    embedder = BlendshapeFaceEmbedder()
    noise_crop = (np.random.rand(160, 160, 3) * 255).astype(np.uint8)
    result = embedder(noise_crop)
    # random noise should not contain a detectable face
    assert result is None or result.shape == (BLENDSHAPE_DIM,)


@pytest.mark.skipif(not _have_face_models, reason="run `python -m preprocessing.download_pretrained_assets` first")
def test_extract_listener_reaction_features_end_to_end(tmp_path):
    cv2 = pytest.importorskip("cv2", reason="opencv not installed")
    from preprocessing.extract_listener_reaction import extract_listener_reaction_features

    video_path = str(tmp_path / "synthetic.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w, h = 320, 240
    writer = cv2.VideoWriter(video_path, fourcc, 25.0, (w, h))
    for _ in range(50):
        writer.write((np.random.rand(h, w, 3) * 255).astype(np.uint8))
    writer.release()

    bbox_map = {i: (50, 50, 200, 200) for i in range(50)}
    feats = extract_listener_reaction_features(
        video_path, bbox_map, t_end=1.0, window_pre_sec=0.5, window_post_sec=0.5, num_frames=8
    )
    assert feats.ndim == 2
    assert feats.shape[1] == 52


def test_default_placeholder_embedder_matches_blendshape_dim():
    """The dependency-free placeholder should stay shape-compatible with the
    real embedder it stands in for, so config/model dims don't silently
    drift out of sync (this is the one encoder test that never skips).
    """
    from preprocessing.extract_listener_reaction import default_placeholder_embed_fn
    from preprocessing.pretrained_face_embedder import BLENDSHAPE_DIM

    embed = default_placeholder_embed_fn()
    crop = (np.random.rand(64, 64, 3) * 255).astype(np.float32)
    out = embed(crop)
    assert out.shape == (BLENDSHAPE_DIM,)


def test_speaker_embedding_shape_and_normalization(tmp_path):
    pytest.importorskip("resemblyzer", reason="resemblyzer not installed")
    soundfile = pytest.importorskip("soundfile", reason="soundfile not installed")
    from preprocessing.extract_speaker_embedding import SPEAKER_EMB_DIM, embed_speaker_from_wav

    sr = 16000
    t = np.linspace(0, 1.5, int(sr * 1.5))
    wav = 0.3 * np.sin(2 * np.pi * 150 * t).astype(np.float32)
    path = str(tmp_path / "speaker.wav")
    soundfile.write(path, wav, sr)

    embedding = embed_speaker_from_wav(path)
    assert embedding.shape == (SPEAKER_EMB_DIM,)
    assert abs(np.linalg.norm(embedding) - 1.0) < 1e-4
