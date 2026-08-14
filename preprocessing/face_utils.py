"""Face detection / tracking helpers for building the listener-reaction
windows described in the paper (README §"data strategy", design doc §16).

This module intentionally keeps the speaker/listener association simple and
conservative: MELD already ships per-utterance speaker labels, so we don't
need audio-visual active-speaker detection to know *who* is speaking --
only *which face in the frame* that speaker corresponds to, and which
persistent face is everybody else (the listener candidate). We restrict to
scenes with exactly two tracked faces (see `build_meld_dyadic_subset.py`),
which sidesteps the hard multi-party active-speaker-detection problem
entirely, at the cost of throwing away multi-party dialogues (documented
paper limitation).

Face detection uses MediaPipe's Tasks API (`vision.FaceDetector`) with the
`blaze_face_short_range.tflite` model -- NOT the legacy `mp.solutions.face_detection`
API, which was removed in mediapipe>=0.10.30. Run
`python -m preprocessing.download_pretrained_assets` once to fetch the
model file (served from storage.googleapis.com, no auth/GitHub needed).
Requires `opencv-python-headless` and `mediapipe` (see requirements.txt);
both are optional dependencies -- only needed for actually running this
preprocessing step, not for `react_tts/` model code or tests.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

DEFAULT_FACE_DETECTOR_MODEL = "models/mediapipe/blaze_face_short_range.tflite"
_detector_cache: dict = {}


@dataclass
class FaceBox:
    frame_idx: int
    timestamp: float  # seconds
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float

    def iou(self, other: "FaceBox") -> float:
        xa1, ya1 = max(self.x1, other.x1), max(self.y1, other.y1)
        xa2, ya2 = min(self.x2, other.x2), min(self.y2, other.y2)
        inter = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
        area_self = max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)
        area_other = max(0.0, other.x2 - other.x1) * max(0.0, other.y2 - other.y1)
        union = area_self + area_other - inter
        return inter / union if union > 0 else 0.0


@dataclass
class FaceTrack:
    track_id: int
    boxes: List[FaceBox]

    @property
    def start_time(self) -> float:
        return self.boxes[0].timestamp

    @property
    def end_time(self) -> float:
        return self.boxes[-1].timestamp

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def mean_box_area(self) -> float:
        areas = [(b.x2 - b.x1) * (b.y2 - b.y1) for b in self.boxes]
        return sum(areas) / len(areas)


def _get_face_detector(model_path: str, min_confidence: float):
    """Tasks API detectors are expensive to construct (loads a native
    shared library + model graph) -- cache one per (model_path, threshold).
    """
    key = (model_path, min_confidence)
    if key not in _detector_cache:
        import mediapipe as mp  # local import: optional dependency
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceDetectorOptions(base_options=base_options, min_detection_confidence=min_confidence)
        _detector_cache[key] = vision.FaceDetector.create_from_options(options)
    return _detector_cache[key]


def detect_faces_per_frame(frame, min_confidence: float = 0.5, model_path: str = DEFAULT_FACE_DETECTOR_MODEL) -> List[FaceBox]:
    """Detect faces in a single BGR frame (numpy array, e.g. from cv2.VideoCapture).

    Returns boxes in absolute pixel coordinates. Caller is responsible for
    supplying `frame_idx`/`timestamp` (this function only fills geometry +
    confidence).
    """
    import mediapipe as mp  # local import: optional dependency

    detector = _get_face_detector(model_path, min_confidence)
    rgb = np.ascontiguousarray(frame[:, :, ::-1])  # BGR -> RGB, C-contiguous for mp.Image
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)

    boxes = []
    for det in result.detections:
        bb = det.bounding_box  # already absolute pixel coords in the Tasks API
        x1, y1 = float(bb.origin_x), float(bb.origin_y)
        x2, y2 = x1 + bb.width, y1 + bb.height
        conf = det.categories[0].score if det.categories else min_confidence
        boxes.append(FaceBox(frame_idx=-1, timestamp=-1.0, x1=x1, y1=y1, x2=x2, y2=y2, confidence=conf))
    return boxes


def read_frames_sequential(video_path: str, start_sec: float, end_sec: float):
    """Decode [start_sec, end_sec) with exactly ONE seek + sequential
    `cap.read()` calls, instead of one `cap.set(POS_FRAMES, i)` per frame.

    Repeated `cv2.VideoCapture.set(CAP_PROP_POS_FRAMES)` is slow on
    compressed video (each seek can force a decode from the nearest
    preceding keyframe); sequential reads let the decoder proceed frame by
    frame with no re-seeking. This is the single video-I/O pass that
    `build_meld_dyadic_subset.py` reuses for face tracking, mouth-motion
    estimation, AND listener-reaction embedding, instead of opening/seeking
    the same clip three separate times.

    Takes seconds (not frame numbers) because fps is only known after
    opening the video -- resolved internally so callers don't need it first.

    Returns (fps, frames) where frames is {frame_idx: BGR ndarray}, for
    whatever frames were actually decodable in range (may end early if the
    clip is shorter than `end_sec`).
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    start_frame = max(0, int(start_sec * fps))
    end_frame = int(end_sec * fps)
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames = {}
    frame_idx = start_frame
    while frame_idx < end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        frames[frame_idx] = frame
        frame_idx += 1
    cap.release()
    return fps, frames


def build_face_tracks_from_frames(frames: dict, fps: float, stride: int = 1, min_confidence: float = 0.5) -> List[FaceTrack]:
    """Same tracking logic as before, but operating on already-decoded
    frames (see `read_frames_sequential`) instead of re-reading the video.
    """
    frame_indices = sorted(frames.keys())[::stride]
    per_frame_boxes = []
    for frame_idx in frame_indices:
        boxes = detect_faces_per_frame(frames[frame_idx], min_confidence=min_confidence)
        for b in boxes:
            b.frame_idx = frame_idx
            b.timestamp = frame_idx / fps
        per_frame_boxes.append(boxes)
    return track_faces(per_frame_boxes, max_gap_frames=int(fps))  # allow up to 1s occlusion


def track_faces(per_frame_boxes: List[List[FaceBox]], iou_threshold: float = 0.3, max_gap_frames: int = 5) -> List[FaceTrack]:
    """Greedy IoU tracker across frames -> list of FaceTrack.

    Simple and fast; sufficient for short (~2s) reaction windows where faces
    rarely cross paths. Not intended for long, crowded multi-party scenes
    (which are excluded from the dyadic subset anyway).
    """
    active: dict[int, FaceTrack] = {}
    finished: List[FaceTrack] = []
    next_id = 0
    last_seen_frame: dict[int, int] = {}

    for frame_idx, boxes in enumerate(per_frame_boxes):
        matched_track_ids = set()
        for box in boxes:
            best_track_id, best_iou = None, iou_threshold
            for track_id, track in active.items():
                iou = track.boxes[-1].iou(box)
                if iou > best_iou:
                    best_track_id, best_iou = track_id, iou
            if best_track_id is not None and best_track_id not in matched_track_ids:
                active[best_track_id].boxes.append(box)
                last_seen_frame[best_track_id] = frame_idx
                matched_track_ids.add(best_track_id)
            else:
                track = FaceTrack(track_id=next_id, boxes=[box])
                active[next_id] = track
                last_seen_frame[next_id] = frame_idx
                next_id += 1

        stale = [tid for tid, last in last_seen_frame.items() if frame_idx - last > max_gap_frames]
        for tid in stale:
            finished.append(active.pop(tid))
            last_seen_frame.pop(tid)

    finished.extend(active.values())
    return finished


def select_listener_track(tracks: List[FaceTrack], speaker_track_id: int, min_duration: float = 0.5) -> Optional[FaceTrack]:
    """Among tracks other than the (already identified) speaker, pick the
    listener candidate: `argmax [visibility + temporal continuity]`
    (design doc §16), approximated here as longest-duration, sufficiently
    visible non-speaker track.
    """
    candidates = [t for t in tracks if t.track_id != speaker_track_id and t.duration >= min_duration]
    if not candidates:
        return None
    return max(candidates, key=lambda t: (t.duration, t.mean_box_area))
