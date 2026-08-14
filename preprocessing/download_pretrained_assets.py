"""One-time download of the pretrained MediaPipe model files used by
`face_utils.py` (face detection) and `extract_listener_reaction.py` (facial
expression / blendshape embedding).

Both files are served directly by Google's own CDN (`storage.googleapis.com`)
rather than a GitHub release or a model hub, which matters in network-
restricted environments: it's a plain file download, no auth, no redirect
through a blocked host.

    python -m preprocessing.download_pretrained_assets [--out_dir models/mediapipe]

Note: MediaPipe's Tasks API (used here; the legacy `mp.solutions` API was
removed in mediapipe>=0.10.30) loads a native shared library that needs
EGL/GLES at import time even for CPU-only inference. On Debian/Ubuntu:
    apt-get install -y libgl1 libegl1 libgles2
"""
from __future__ import annotations

import argparse
import os
import urllib.request

ASSETS = {
    "blaze_face_short_range.tflite": (
        "https://storage.googleapis.com/mediapipe-models/face_detector/"
        "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
    ),
    "face_landmarker_v2_with_blendshapes.task": (
        "https://storage.googleapis.com/mediapipe-assets/"
        "face_landmarker_v2_with_blendshapes.task"
    ),
}


def download_assets(out_dir: str = "models/mediapipe", force: bool = False) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for filename, url in ASSETS.items():
        dest = os.path.join(out_dir, filename)
        if os.path.exists(dest) and not force:
            print(f"[skip] {dest} already exists")
        else:
            print(f"[download] {url} -> {dest}")
            urllib.request.urlretrieve(url, dest)
        paths[filename] = dest
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="models/mediapipe")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    download_assets(args.out_dir, args.force)
