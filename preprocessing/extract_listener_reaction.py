"""Extract listener facial-reaction features from an already tracked face.

For MELD, each ``dia{D}_utt{U}.mp4`` file is an utterance-level clip that ends
at the end of that utterance. Therefore a nominal reaction window such as

    [T_end - 1 s, T_end + 1 s]

cannot be read from the previous-utterance clip after ``T_end``: those frames
simply do not exist in that file. ReACT-TTS therefore uses the *available
pre-response listener window* ending at the previous utterance boundary,
typically

    [T_end - window_pre_sec, T_end]

when ``window_post_sec=0``.

Face tracks are also produced only every ``FACE_DETECT_STRIDE`` frames.
Reaction timestamps therefore should not require an exact frame-index match
to a tracked bbox. This implementation uses the nearest tracked bbox within a
small tolerance, while requiring the image frame itself to exist.

The resulting feature array has shape
``[num_valid_reaction_frames, face_feat_dim]``. It may contain fewer than the
requested number of samples if the listener is not tracked or the
FaceLandmarker cannot embed some crops.
"""
from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np

from preprocessing.pretrained_face_embedder import BLENDSHAPE_DIM

_default_embedder_cache: dict = {}


def default_pretrained_embed_fn(
    model_path: Optional[str] = None,
) -> Callable[[np.ndarray], Optional[np.ndarray]]:
    """Return the cached 52-D MediaPipe blendshape embedder."""
    from preprocessing.pretrained_face_embedder import (
        DEFAULT_LANDMARKER_MODEL,
        BlendshapeFaceEmbedder,
    )

    key = model_path or DEFAULT_LANDMARKER_MODEL
    if key not in _default_embedder_cache:
        _default_embedder_cache[key] = BlendshapeFaceEmbedder(
            model_path=key
        )
    return _default_embedder_cache[key]


def default_placeholder_embed_fn(
    face_feat_dim: int = BLENDSHAPE_DIM,
    seed: int = 0,
) -> Callable[[np.ndarray], np.ndarray]:
    """Deterministic random projection for dependency-free pipeline tests."""
    rng = np.random.RandomState(seed)
    projection: Optional[np.ndarray] = None

    def embed(face_crop: np.ndarray) -> np.ndarray:
        nonlocal projection
        flat = face_crop.astype(np.float32).reshape(-1)
        if projection is None or projection.shape[0] != flat.shape[0]:
            projection = (
                rng.randn(flat.shape[0], face_feat_dim).astype(np.float32)
                / np.sqrt(flat.shape[0])
            )
        return flat @ projection

    return embed


def sample_reaction_frame_indices(
    fps: float,
    t_end: float,
    window_pre_sec: float,
    window_post_sec: float,
    num_frames: int,
) -> List[int]:
    """Uniformly sample requested reaction timestamps.

    For utterance-level MELD clips, callers should normally pass
    ``window_post_sec=0`` because frames after ``t_end`` belong outside the
    previous utterance clip.
    """
    start_t = max(0.0, t_end - window_pre_sec)
    end_t = max(start_t, t_end + window_post_sec)
    times = np.linspace(start_t, end_t, num_frames)
    return [int(round(t * fps)) for t in times]


def _nearest_key(
    sorted_keys: List[int],
    target: int,
    max_distance: int,
) -> Optional[int]:
    """Return nearest integer key when it is within ``max_distance``."""
    if not sorted_keys:
        return None

    # Small lists (reaction tracks) make this simple implementation adequate.
    nearest = min(sorted_keys, key=lambda k: abs(k - target))
    if abs(nearest - target) <= max_distance:
        return nearest
    return None


def _embed_bbox_in_frame(
    frame: np.ndarray,
    bbox,
    embed_fn,
) -> Optional[np.ndarray]:
    import cv2

    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = frame.shape[:2]

    x1 = max(0, min(w, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h, y1))
    y2 = max(0, min(h, y2))

    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    crop = cv2.resize(crop, (192, 192))
    return embed_fn(crop)


def extract_listener_reaction_features_from_frames(
    frames: dict,
    fps: float,
    listener_bbox_per_frame: dict,
    t_end: float,
    window_pre_sec: float = 1.0,
    window_post_sec: float = 0.0,
    num_frames: int = 16,
    face_feat_dim: int = BLENDSHAPE_DIM,
    embed_fn: Optional[Callable[[np.ndarray], Optional[np.ndarray]]] = None,
    bbox_max_distance_frames: int = 2,
) -> np.ndarray:
    """Extract reaction features while reusing already-decoded frames.

    The requested sample frame must exist in ``frames``. The listener bbox,
    however, may come from a nearby tracked frame (useful when tracking was
    performed with stride > 1).
    """
    embed_fn = embed_fn or default_pretrained_embed_fn()

    requested_indices = sample_reaction_frame_indices(
        fps,
        t_end,
        window_pre_sec,
        window_post_sec,
        num_frames,
    )

    available_frame_keys = sorted(frames.keys())
    bbox_keys = sorted(listener_bbox_per_frame.keys())

    if not available_frame_keys or not bbox_keys:
        return np.zeros((0, face_feat_dim), dtype=np.float32)

    min_frame = available_frame_keys[0]
    max_frame = available_frame_keys[-1]

    features = []

    for requested_idx in requested_indices:
        # Do not fabricate post-clip frames. If the requested frame does not
        # exist because the utterance-level clip has ended, skip it.
        if requested_idx < min_frame or requested_idx > max_frame:
            continue

        frame_idx = _nearest_key(
            available_frame_keys,
            requested_idx,
            max_distance=1,
        )
        if frame_idx is None:
            continue

        bbox_idx = _nearest_key(
            bbox_keys,
            frame_idx,
            max_distance=bbox_max_distance_frames,
        )
        if bbox_idx is None:
            continue

        frame = frames.get(frame_idx)
        bbox = listener_bbox_per_frame.get(bbox_idx)
        if frame is None or bbox is None:
            continue

        feat = _embed_bbox_in_frame(
            frame,
            bbox,
            embed_fn,
        )
        if feat is not None:
            features.append(feat)

    if not features:
        return np.zeros(
            (0, face_feat_dim),
            dtype=np.float32,
        )

    return np.stack(features).astype(np.float32)


def extract_listener_reaction_features(
    video_path: str,
    listener_bbox_per_frame: dict,
    t_end: float,
    window_pre_sec: float = 1.0,
    window_post_sec: float = 0.0,
    num_frames: int = 16,
    face_feat_dim: int = BLENDSHAPE_DIM,
    embed_fn: Optional[Callable[[np.ndarray], Optional[np.ndarray]]] = None,
    bbox_max_distance_frames: int = 2,
) -> np.ndarray:
    """Standalone variant for callers without already-decoded frames."""
    import cv2

    embed_fn = embed_fn or default_pretrained_embed_fn()

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    requested_indices = sample_reaction_frame_indices(
        fps,
        t_end,
        window_pre_sec,
        window_post_sec,
        num_frames,
    )
    bbox_keys = sorted(listener_bbox_per_frame.keys())

    features = []

    for frame_idx in requested_indices:
        bbox_idx = _nearest_key(
            bbox_keys,
            frame_idx,
            max_distance=bbox_max_distance_frames,
        )
        if bbox_idx is None:
            continue

        bbox = listener_bbox_per_frame.get(bbox_idx)
        if bbox is None:
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue

        feat = _embed_bbox_in_frame(
            frame,
            bbox,
            embed_fn,
        )
        if feat is not None:
            features.append(feat)

    cap.release()

    if not features:
        return np.zeros(
            (0, face_feat_dim),
            dtype=np.float32,
        )

    return np.stack(features).astype(np.float32)