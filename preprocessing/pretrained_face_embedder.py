"""Real pretrained facial-expression embedder for the listener-reaction
pipeline, replacing the random-projection placeholder that used to live in
`extract_listener_reaction.py`.

Uses MediaPipe FaceLandmarker (Tasks API) with the "v2 with blendshapes"
model: 52 ARKit-style blendshape coefficients (browDownLeft, mouthSmileLeft,
eyeBlinkRight, jawOpen, ...). This is deliberately an *expression* descriptor
rather than a face-*identity* embedding (e.g. FaceNet/ArcFace) -- for
listener-reaction modeling we want "what is the face doing", not "whose face
is it", and blendshapes give that directly and interpretably (which is also
convenient for the qualitative analysis in design doc §9/§16: e.g. tracking
mouthSmileLeft/Right collapsing, browDownLeft/Right rising).

Run `python -m preprocessing.download_pretrained_assets` once first to fetch
the model file (served from storage.googleapis.com; no GitHub/HF/auth
needed, unlike most face-recognition checkpoints).
"""
from __future__ import annotations

from typing import Optional

import numpy as np

DEFAULT_LANDMARKER_MODEL = "models/mediapipe/face_landmarker_v2_with_blendshapes.task"
BLENDSHAPE_DIM = 52


class BlendshapeFaceEmbedder:
    def __init__(self, model_path: str = DEFAULT_LANDMARKER_MODEL, min_face_confidence: float = 0.3):
        import mediapipe as mp  # local import: optional dependency
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False,
            num_faces=1,
            min_face_detection_confidence=min_face_confidence,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._mp = mp

    def __call__(self, face_crop_bgr: np.ndarray) -> Optional[np.ndarray]:
        """face_crop_bgr: HxWx3 uint8 BGR crop (e.g. from cv2). Returns a
        [52] float32 blendshape-score vector, or None if no face was found
        in the crop (caller should skip that frame, mirroring the existing
        "missing detection" handling elsewhere in this pipeline).
        """
        rgb = np.ascontiguousarray(face_crop_bgr[:, :, ::-1])
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)
        if not result.face_blendshapes:
            return None
        return np.array([c.score for c in result.face_blendshapes[0]], dtype=np.float32)

    def close(self):
        self._landmarker.close()
