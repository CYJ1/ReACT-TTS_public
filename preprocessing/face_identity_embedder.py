"""Face identity embeddings for listener verification in ReACT-TTS.

This module is deliberately separate from `pretrained_face_embedder.py`.

- `pretrained_face_embedder.py` -> MediaPipe 52-D blendshape/expression features
- this module -> 512-D identity embeddings from FaceNet/InceptionResnetV1

The identity embedding is used only to verify whether a single face visible
during the previous utterance belongs to the speaker of the following target
utterance (i.e., the future respondent / listener candidate).

The implementation reuses MediaPipe face tracks already computed elsewhere
instead of running another face detector.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from preprocessing.face_utils import FaceTrack


class FaceIdentityEmbedder:
    """VGGFace2-pretrained FaceNet identity embedder."""

    def __init__(
        self,
        device: Optional[str] = None,
        max_frames: int = 5,
        crop_margin: float = 0.20,
    ):
        import torch
        from facenet_pytorch import InceptionResnetV1

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)
        self.max_frames = max(1, int(max_frames))
        self.crop_margin = max(0.0, float(crop_margin))

        self.model = (
            InceptionResnetV1(pretrained="vggface2")
            .eval()
            .to(self.device)
        )

    @staticmethod
    def _sample_boxes(boxes: Sequence, max_frames: int):
        if len(boxes) <= max_frames:
            return list(boxes)

        idx = np.linspace(0, len(boxes) - 1, max_frames)
        idx = np.unique(np.round(idx).astype(int))
        return [boxes[i] for i in idx]

    def _crop_to_tensor(self, frame_bgr: np.ndarray, box):
        import cv2
        import torch

        h, w = frame_bgr.shape[:2]

        bw = float(box.x2 - box.x1)
        bh = float(box.y2 - box.y1)
        mx = bw * self.crop_margin
        my = bh * self.crop_margin

        x1 = max(0, int(round(box.x1 - mx)))
        y1 = max(0, int(round(box.y1 - my)))
        x2 = min(w, int(round(box.x2 + mx)))
        y2 = min(h, int(round(box.y2 + my)))

        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        crop = cv2.resize(
            crop,
            (160, 160),
            interpolation=cv2.INTER_LINEAR,
        )
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop = np.ascontiguousarray(crop).astype(np.float32)

        tensor = torch.from_numpy(crop).permute(2, 0, 1)
        tensor = (tensor - 127.5) / 128.0
        return tensor

    def embed_track(
        self,
        frames: dict,
        track: FaceTrack,
    ) -> Optional[np.ndarray]:
        """Return an L2-normalized 512-D embedding for one face track."""
        import torch
        import torch.nn.functional as F

        tensors = []
        for box in self._sample_boxes(track.boxes, self.max_frames):
            frame = frames.get(box.frame_idx)
            if frame is None:
                continue

            tensor = self._crop_to_tensor(frame, box)
            if tensor is not None:
                tensors.append(tensor)

        if not tensors:
            return None

        batch = torch.stack(tensors, dim=0).to(self.device)

        with torch.inference_mode():
            emb = self.model(batch)
            emb = F.normalize(emb, p=2, dim=1)
            emb = emb.mean(dim=0, keepdim=True)
            emb = F.normalize(emb, p=2, dim=1)

        return emb[0].detach().cpu().numpy().astype(np.float32)

    @staticmethod
    def cosine_similarity(
        embedding_a: np.ndarray,
        embedding_b: np.ndarray,
    ) -> float:
        a = np.asarray(embedding_a, dtype=np.float32)
        b = np.asarray(embedding_b, dtype=np.float32)

        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom <= 0:
            return float("-inf")

        return float(np.dot(a, b) / denom)