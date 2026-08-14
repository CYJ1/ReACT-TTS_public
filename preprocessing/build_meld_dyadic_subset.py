"""Build a listener-reaction-aware dyadic subset of MELD for ReACT-TTS.

The data-selection logic separates three concepts that were previously
conflated:

1. Dyadic dialogue criterion:
   context + target must contain exactly two labelled speakers.

2. Face-track persistence:
   a detected face track must be visible for at least
   `--min_track_visibility_ratio` of the decoded previous-turn window.

3. Reaction-feature validity:
   MediaPipe blendshape extraction must succeed for at least
   `--min_reaction_valid_ratio` of the requested reaction frames.

Listener selection is identity-based for both one-face and two-face
reaction shots. The visually inferred speaker of the following target
utterance is embedded with FaceNet. Previous-turn face tracks are embedded
and compared against that future respondent:

* 1 visible previous face: keep it only if it matches the target speaker.
* 2 visible previous faces: choose the face with the highest similarity to
  the target speaker, and keep it only if that best score passes
  `--identity_match_threshold`.
* 0 or >2 persistent visible faces: reject conservatively.

This makes the listener definition consistent across shot/reverse-shot and
two-face scenes: the listener in the previous turn is the person who becomes
the subsequent respondent.

The script prints detailed diagnostic counters, a reaction-valid-frame
histogram, and an identity-similarity histogram so thresholds can be chosen
on train/dev rather than tuned on test.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from collections import Counter, defaultdict
from typing import Dict, List

from preprocessing.extract_listener_reaction import (
    extract_listener_reaction_features_from_frames,
)
from preprocessing.extract_prosody_features import (
    RawProsody,
    SpeakerNormalizer,
    extract_raw_prosody,
)
from preprocessing.face_identity_embedder import FaceIdentityEmbedder
from preprocessing.face_utils import (
    FaceTrack,
    build_face_tracks_from_frames,
    read_frames_sequential,
)

MIN_TARGET_DURATION_SEC = 1.0
DEFAULT_TRACK_VISIBILITY_RATIO = 0.30
DEFAULT_REACTION_VALID_RATIO = 0.30
DEFAULT_IDENTITY_MATCH_THRESHOLD = 0.65
FACE_DETECT_STRIDE = 2

DEBUG_STATS = Counter()
IDENTITY_SIMILARITIES: List[float] = []
SINGLE_FACE_SIMILARITIES: List[float] = []
TWO_FACE_BEST_SIMILARITIES: List[float] = []
TWO_FACE_SECOND_SIMILARITIES: List[float] = []
REACTION_VALID_FRAMES = Counter()


def parse_meld_timestamp(ts: str) -> float:
    hms, millis = ts.split(",")
    h, m, s = (int(x) for x in hms.split(":"))
    return h * 3600 + m * 60 + s + int(millis) / 1000.0


def load_meld_csv(csv_path: str) -> Dict[str, List[dict]]:
    dialogues: Dict[str, List[dict]] = defaultdict(list)

    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if "StartTime" in row:
                row["StartTime_sec"] = parse_meld_timestamp(row["StartTime"])
            if "EndTime" in row:
                row["EndTime_sec"] = parse_meld_timestamp(row["EndTime"])
            dialogues[row["Dialogue_ID"]].append(row)

    for dialogue_id in dialogues:
        dialogues[dialogue_id].sort(
            key=lambda r: int(r["Utterance_ID"])
        )

    return dialogues


def clip_path(
    video_dir: str,
    dialogue_id: str,
    utterance_id: str,
) -> str:
    return os.path.join(
        video_dir,
        f"dia{dialogue_id}_utt{utterance_id}.mp4",
    )


def estimate_mouth_motion_energy_from_frames(
    frames: dict,
    bbox_per_frame: dict,
    start_frame: int,
    end_frame: int,
) -> float:
    import cv2
    import numpy as np

    prev_mouth = None
    energies = []

    for frame_idx in range(start_frame, end_frame):
        bbox = bbox_per_frame.get(frame_idx)
        frame = frames.get(frame_idx)

        if bbox is None or frame is None:
            prev_mouth = None
            continue

        x1, y1, x2, y2 = [int(v) for v in bbox]
        mouth = frame[
            y1 + (y2 - y1) // 2 : y2,
            x1:x2,
        ]

        if mouth.size == 0:
            continue

        mouth = cv2.resize(
            cv2.cvtColor(mouth, cv2.COLOR_BGR2GRAY),
            (32, 16),
        ).astype("float32")

        if prev_mouth is not None:
            energies.append(
                float(np.mean(np.abs(mouth - prev_mouth)))
            )

        prev_mouth = mouth

    return (
        sum(energies) / len(energies)
        if energies
        else 0.0
    )


def track_bbox_map(track: FaceTrack) -> dict:
    return {
        b.frame_idx: (b.x1, b.y1, b.x2, b.y2)
        for b in track.boxes
    }


def select_most_active_track(
    tracks: List[FaceTrack],
    frames: dict,
    start_frame: int,
    end_frame: int,
) -> FaceTrack:
    energies = []

    for track in tracks:
        energies.append(
            estimate_mouth_motion_energy_from_frames(
                frames,
                track_bbox_map(track),
                start_frame,
                end_frame,
            )
        )

    return tracks[
        max(
            range(len(tracks)),
            key=lambda i: energies[i],
        )
    ]


def infer_target_speaker_track(
    target_video: str,
    target_duration: float,
    min_track_visibility_ratio: float,
):
    """Infer the target-speaker face from the labelled target utterance."""
    try:
        fps, frames = read_frames_sequential(
            target_video,
            0.0,
            target_duration,
        )
    except Exception:
        DEBUG_STATS["single_face_target_decode_exception"] += 1
        return None, {}, None

    if not frames:
        DEBUG_STATS["single_face_target_no_frames"] += 1
        return None, frames, fps

    tracks = build_face_tracks_from_frames(
        frames,
        fps,
        stride=FACE_DETECT_STRIDE,
    )

    expected_frames = max(
        1,
        math.ceil(len(frames) / FACE_DETECT_STRIDE),
    )

    visible_tracks = [
        tr
        for tr in tracks
        if (
            len(tr.boxes) / expected_frames
            >= min_track_visibility_ratio
        )
    ]

    DEBUG_STATS[
        f"single_face_target_visible_tracks_{len(visible_tracks)}"
    ] += 1

    if not visible_tracks:
        DEBUG_STATS["single_face_target_no_visible_track"] += 1
        return None, frames, fps

    start_frame = min(frames.keys())
    end_frame = max(frames.keys()) + 1

    target_speaker_track = select_most_active_track(
        visible_tracks,
        frames,
        start_frame,
        end_frame,
    )

    return target_speaker_track, frames, fps


def infer_target_speaker_embedding(
    target_video: str,
    target_duration: float,
    min_track_visibility_ratio: float,
    identity_embedder: FaceIdentityEmbedder,
):
    """Return the FaceNet embedding of the visually inferred target speaker."""
    target_track, target_frames, _ = infer_target_speaker_track(
        target_video=target_video,
        target_duration=target_duration,
        min_track_visibility_ratio=min_track_visibility_ratio,
    )

    if target_track is None:
        DEBUG_STATS["target_speaker_not_found"] += 1
        return None

    target_embedding = identity_embedder.embed_track(
        target_frames,
        target_track,
    )

    if target_embedding is None:
        DEBUG_STATS["target_identity_failed"] += 1
        return None

    return target_embedding


def select_listener_by_future_identity(
    previous_tracks: List[FaceTrack],
    previous_frames: dict,
    target_video: str,
    target_duration: float,
    min_track_visibility_ratio: float,
    identity_embedder: FaceIdentityEmbedder,
    identity_match_threshold: float,
):
    """Select the previous-turn listener using future-respondent identity.

    The target utterance is spoken by the future respondent. We infer that
    speaker's face in the target clip, embed it, then compare it with all
    persistent face tracks from the previous utterance.

    Returns:
        (selected_track_or_none, similarities)

    For one previous track, the track is retained only if its similarity
    exceeds the threshold. For two tracks, the higher-scoring track is
    selected only if the best similarity exceeds the threshold.
    """
    target_embedding = infer_target_speaker_embedding(
        target_video=target_video,
        target_duration=target_duration,
        min_track_visibility_ratio=min_track_visibility_ratio,
        identity_embedder=identity_embedder,
    )

    if target_embedding is None:
        return None, []

    scored_tracks = []

    for track in previous_tracks:
        previous_embedding = identity_embedder.embed_track(
            previous_frames,
            track,
        )

        if previous_embedding is None:
            DEBUG_STATS["previous_identity_failed"] += 1
            continue

        similarity = identity_embedder.cosine_similarity(
            previous_embedding,
            target_embedding,
        )
        IDENTITY_SIMILARITIES.append(similarity)
        scored_tracks.append((similarity, track))

    if not scored_tracks:
        DEBUG_STATS["no_previous_identity_embedding"] += 1
        return None, []

    scored_tracks.sort(key=lambda x: x[0], reverse=True)
    similarities = [score for score, _ in scored_tracks]
    best_similarity, best_track = scored_tracks[0]

    if len(previous_tracks) == 1:
        SINGLE_FACE_SIMILARITIES.append(best_similarity)
        DEBUG_STATS["single_face_identity_comparisons"] += 1

        if best_similarity >= identity_match_threshold:
            DEBUG_STATS["single_face_identity_verified"] += 1
            return best_track, similarities

        DEBUG_STATS["single_face_identity_rejected"] += 1
        return None, similarities

    # Exactly two persistent previous-turn face tracks.
    TWO_FACE_BEST_SIMILARITIES.append(best_similarity)
    DEBUG_STATS["two_face_identity_comparisons"] += 1

    if len(scored_tracks) >= 2:
        TWO_FACE_SECOND_SIMILARITIES.append(scored_tracks[1][0])

    if best_similarity >= identity_match_threshold:
        DEBUG_STATS["two_face_identity_verified"] += 1
        return best_track, similarities

    DEBUG_STATS["two_face_identity_rejected"] += 1
    return None, similarities


def process_dialogue(
    dialogue_id: str,
    utterances: List[dict],
    video_dir: str,
    num_context_turns: int,
    window_pre_sec: float,
    window_post_sec: float,
    reaction_num_frames: int,
    face_feat_dim: int,
    min_track_visibility_ratio: float,
    min_reaction_valid_ratio: float,
    identity_embedder: FaceIdentityEmbedder,
    identity_match_threshold: float,
) -> List[dict]:
    samples = []

    for t in range(1, len(utterances)):
        DEBUG_STATS["candidate_windows"] += 1

        context = utterances[
            max(0, t - num_context_turns) : t
        ]
        target = utterances[t]

        speakers = (
            {u["Speaker"] for u in context}
            | {target["Speaker"]}
        )

        if len(speakers) != 2:
            DEBUG_STATS["skip_not_exactly_2_speakers"] += 1
            continue

        target_duration = (
            float(target["EndTime_sec"])
            - float(target["StartTime_sec"])
            if "EndTime_sec" in target
            else None
        )

        if (
            target_duration is not None
            and target_duration < MIN_TARGET_DURATION_SEC
        ):
            DEBUG_STATS["skip_target_too_short"] += 1
            continue

        prev = context[-1]

        prev_video = clip_path(
            video_dir,
            dialogue_id,
            prev["Utterance_ID"],
        )

        if not os.path.exists(prev_video):
            DEBUG_STATS["skip_prev_video_missing"] += 1
            continue

        prev_duration = max(
            0.0,
            float(prev.get("EndTime_sec", 0.0))
            - float(prev.get("StartTime_sec", 0.0)),
        )

        clip_t_end = prev_duration
        window_start_sec = max(
            0.0,
            clip_t_end - window_pre_sec,
        )
        window_end_sec = (
            clip_t_end + window_post_sec
        )

        try:
            fps, frames = read_frames_sequential(
                prev_video,
                window_start_sec,
                window_end_sec,
            )
        except Exception:
            DEBUG_STATS["skip_video_decode_exception"] += 1
            continue

        if not frames:
            DEBUG_STATS["skip_no_frames"] += 1
            continue

        tracks = build_face_tracks_from_frames(
            frames,
            fps,
            stride=FACE_DETECT_STRIDE,
        )

        expected_frames = max(
            1,
            math.ceil(len(frames) / FACE_DETECT_STRIDE),
        )

        DEBUG_STATS[f"raw_tracks_{len(tracks)}"] += 1

        visible_tracks = [
            tr
            for tr in tracks
            if (
                len(tr.boxes) / expected_frames
                >= min_track_visibility_ratio
            )
        ]

        DEBUG_STATS[
            f"visible_tracks_{len(visible_tracks)}"
        ] += 1

        if len(visible_tracks) == 0:
            DEBUG_STATS["skip_no_visible_tracks"] += 1
            continue

        if len(visible_tracks) > 2:
            DEBUG_STATS["skip_more_than_2_visible_tracks"] += 1
            continue

        if len(visible_tracks) == 1:
            DEBUG_STATS["single_face_candidates"] += 1
        else:
            DEBUG_STATS["two_face_candidates"] += 1

        target_video = clip_path(
            video_dir,
            dialogue_id,
            target["Utterance_ID"],
        )

        if not os.path.exists(target_video):
            DEBUG_STATS["target_video_missing"] += 1
            continue

        listener_track, identity_scores = select_listener_by_future_identity(
            previous_tracks=visible_tracks,
            previous_frames=frames,
            target_video=target_video,
            target_duration=float(target_duration),
            min_track_visibility_ratio=min_track_visibility_ratio,
            identity_embedder=identity_embedder,
            identity_match_threshold=identity_match_threshold,
        )

        if listener_track is None:
            if len(visible_tracks) == 1:
                DEBUG_STATS[
                    "skip_single_face_identity_not_verified"
                ] += 1
            else:
                DEBUG_STATS[
                    "skip_two_face_identity_not_verified"
                ] += 1
            continue

        listener_bbox_map = track_bbox_map(listener_track)

        if len(visible_tracks) == 1:
            sample_source = "single_face_identity_verified"
        else:
            sample_source = "two_face_identity_verified"

        reaction_feats = (
            extract_listener_reaction_features_from_frames(
                frames,
                fps,
                listener_bbox_map,
                t_end=clip_t_end,
                window_pre_sec=window_pre_sec,
                window_post_sec=window_post_sec,
                num_frames=reaction_num_frames,
                face_feat_dim=face_feat_dim,
            )
        )

        valid_reaction_frames = int(
            reaction_feats.shape[0]
        )
        REACTION_VALID_FRAMES[
            valid_reaction_frames
        ] += 1

        reaction_ratio = (
            valid_reaction_frames
            / reaction_num_frames
        )

        if (
            reaction_ratio
            < min_reaction_valid_ratio
        ):
            DEBUG_STATS["skip_reaction_validity"] += 1

            if (
                sample_source
                == "single_face_identity_verified"
            ):
                DEBUG_STATS[
                    "skip_single_face_reaction_validity"
                ] += 1
            else:
                DEBUG_STATS[
                    "skip_two_face_reaction_validity"
                ] += 1

            continue

        if (
            sample_source
            == "single_face_identity_verified"
        ):
            DEBUG_STATS["kept_single_face"] += 1
        else:
            DEBUG_STATS["kept_two_face"] += 1

        DEBUG_STATS["kept_total"] += 1

        samples.append(
            {
                "dialogue_id": dialogue_id,
                "context_turns": [
                    {
                        "text": u["Utterance"],
                        "speaker_id": u["Speaker"],
                    }
                    for u in context
                ],
                "context_emotions": [
                    u["Emotion"].lower()
                    for u in context
                ],
                "target_text": target["Utterance"],
                "target_speaker_id": target["Speaker"],
                "target_emotion": target["Emotion"].lower(),
                "_reaction_features": reaction_feats,
                "_target_audio_path": clip_path(
                    video_dir,
                    dialogue_id,
                    target["Utterance_ID"],
                ),
                "_target_speaker_raw": target["Speaker"],
                "listener_selection_source": sample_source,
            }
        )

    return samples


def _print_similarity_histogram(
    title: str,
    values: List[float],
):
    print(f"\n=== {title} ===")

    if not values:
        print("no similarities recorded")
        return

    bins = [
        (-1.0, 0.0),
        (0.0, 0.1),
        (0.1, 0.2),
        (0.2, 0.3),
        (0.3, 0.4),
        (0.4, 0.5),
        (0.5, 0.6),
        (0.6, 0.7),
        (0.7, 0.8),
        (0.8, 0.9),
        (0.9, 1.000001),
    ]

    for low, high in bins:
        count = sum(
            1
            for sim in values
            if low <= sim < high
        )
        label_high = 1.0 if high > 1.0 else high
        print(
            f"{low:>4.1f} <= sim < {label_high:>3.1f}: "
            f"{count}"
        )

    print(f"count: {len(values)}")
    print(f"mean: {sum(values) / len(values):.4f}")
    print(f"min: {min(values):.4f}")
    print(f"max: {max(values):.4f}")


def print_identity_histogram():
    _print_similarity_histogram(
        "ALL IDENTITY SIMILARITIES",
        IDENTITY_SIMILARITIES,
    )
    _print_similarity_histogram(
        "SINGLE-FACE IDENTITY SIMILARITIES",
        SINGLE_FACE_SIMILARITIES,
    )
    _print_similarity_histogram(
        "TWO-FACE BEST IDENTITY SIMILARITIES",
        TWO_FACE_BEST_SIMILARITIES,
    )
    _print_similarity_histogram(
        "TWO-FACE SECOND-BEST IDENTITY SIMILARITIES",
        TWO_FACE_SECOND_SIMILARITIES,
    )

    if (
        TWO_FACE_BEST_SIMILARITIES
        and TWO_FACE_SECOND_SIMILARITIES
        and len(TWO_FACE_BEST_SIMILARITIES)
        == len(TWO_FACE_SECOND_SIMILARITIES)
    ):
        margins = [
            best - second
            for best, second in zip(
                TWO_FACE_BEST_SIMILARITIES,
                TWO_FACE_SECOND_SIMILARITIES,
            )
        ]
        _print_similarity_histogram(
            "TWO-FACE IDENTITY MARGINS (BEST - SECOND)",
            margins,
        )


def print_reaction_histogram(
    reaction_num_frames: int,
):
    print("\n=== REACTION VALID-FRAME HISTOGRAM ===")

    total = sum(
        REACTION_VALID_FRAMES.values()
    )

    for valid_frames in range(
        reaction_num_frames + 1
    ):
        count = REACTION_VALID_FRAMES.get(
            valid_frames,
            0,
        )
        ratio = (
            valid_frames / reaction_num_frames
        )
        print(
            f"{valid_frames:>2}/{reaction_num_frames} "
            f"({ratio:>5.1%}): {count}"
        )

    print(f"total reaction candidates: {total}")


def main():
    DEBUG_STATS.clear()
    IDENTITY_SIMILARITIES.clear()
    SINGLE_FACE_SIMILARITIES.clear()
    TWO_FACE_BEST_SIMILARITIES.clear()
    TWO_FACE_SECOND_SIMILARITIES.clear()
    REACTION_VALID_FRAMES.clear()

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--meld_csv",
        required=True,
    )
    parser.add_argument(
        "--video_dir",
        required=True,
    )
    parser.add_argument(
        "--out_manifest",
        required=True,
    )
    parser.add_argument(
        "--out_features_dir",
        default=None,
        help=(
            "defaults to <out_manifest dir>/reaction_features"
        ),
    )

    parser.add_argument(
        "--num_context_turns",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--window_pre_sec",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--window_post_sec",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--reaction_num_frames",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--face_feat_dim",
        type=int,
        default=52,
    )
    parser.add_argument(
        "--limit_dialogues",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--min_track_visibility_ratio",
        type=float,
        default=DEFAULT_TRACK_VISIBILITY_RATIO,
        help=(
            "Minimum fraction of sampled frames in which a "
            "face track must be detected."
        ),
    )

    parser.add_argument(
        "--min_reaction_valid_ratio",
        type=float,
        default=DEFAULT_REACTION_VALID_RATIO,
        help=(
            "Minimum fraction of requested reaction frames "
            "for which blendshape features must be valid."
        ),
    )

    parser.add_argument(
        "--identity_match_threshold",
        type=float,
        default=DEFAULT_IDENTITY_MATCH_THRESHOLD,
        help=(
            "Cosine-similarity threshold for matching a "
            "previous-turn listener face to the future target speaker."
        ),
    )

    parser.add_argument(
        "--identity_device",
        type=str,
        default=None,
        help=(
            "FaceNet device: cpu, cuda, cuda:0, etc. "
            "Default: CUDA if available."
        ),
    )

    args = parser.parse_args()

    if not (
        0.0
        <= args.min_track_visibility_ratio
        <= 1.0
    ):
        parser.error(
            "--min_track_visibility_ratio must be between 0 and 1"
        )

    if not (
        0.0
        <= args.min_reaction_valid_ratio
        <= 1.0
    ):
        parser.error(
            "--min_reaction_valid_ratio must be between 0 and 1"
        )

    if not (
        -1.0
        <= args.identity_match_threshold
        <= 1.0
    ):
        parser.error(
            "--identity_match_threshold must be between -1 and 1"
        )

    out_features_dir = (
        args.out_features_dir
        or os.path.join(
            os.path.dirname(
                args.out_manifest
            ),
            "reaction_features",
        )
    )

    os.makedirs(
        out_features_dir,
        exist_ok=True,
    )
    os.makedirs(
        os.path.dirname(args.out_manifest),
        exist_ok=True,
    )

    print(
        "Loading FaceNet identity embedder "
        f"(device={args.identity_device or 'auto'})...",
        flush=True,
    )

    identity_embedder = FaceIdentityEmbedder(
        device=args.identity_device,
    )

    dialogues = load_meld_csv(
        args.meld_csv
    )
    dialogue_ids = list(
        dialogues.keys()
    )

    if args.limit_dialogues:
        dialogue_ids = dialogue_ids[
            : args.limit_dialogues
        ]

    all_samples = []
    start_time = time.time()

    for i, dialogue_id in enumerate(
        dialogue_ids,
        start=1,
    ):
        all_samples.extend(
            process_dialogue(
                dialogue_id=dialogue_id,
                utterances=dialogues[dialogue_id],
                video_dir=args.video_dir,
                num_context_turns=args.num_context_turns,
                window_pre_sec=args.window_pre_sec,
                window_post_sec=args.window_post_sec,
                reaction_num_frames=args.reaction_num_frames,
                face_feat_dim=args.face_feat_dim,
                min_track_visibility_ratio=args.min_track_visibility_ratio,
                min_reaction_valid_ratio=args.min_reaction_valid_ratio,
                identity_embedder=identity_embedder,
                identity_match_threshold=args.identity_match_threshold,
            )
        )

        elapsed = time.time() - start_time
        rate = elapsed / i
        remaining = (
            rate * (len(dialogue_ids) - i)
        )

        print(
            f"[{i}/{len(dialogue_ids)} dialogues] "
            f"{len(all_samples)} samples kept so far -- "
            f"{elapsed:.0f}s elapsed, "
            f"~{remaining:.0f}s remaining "
            f"({rate:.1f}s/dialogue)",
            flush=True,
        )

    raw_prosody_cache: Dict[
        int,
        RawProsody,
    ] = {}
    per_speaker: Dict[
        str,
        List[RawProsody],
    ] = defaultdict(list)

    prosody_success = 0
    prosody_failures = []

    for i, sample in enumerate(
    all_samples
        ):
        try:
            num_words = max(
                len(
                    sample[
                        "target_text"
                    ].split()
                ),
                1,
            )

            raw = extract_raw_prosody(
                sample[
                    "_target_audio_path"
                ],
                num_words,
            )

        except Exception as e:
            prosody_failures.append(
                {
                    "index": i,
                    "dialogue_id": sample.get(
                        "dialogue_id"
                    ),
                    "target_audio_path": sample.get(
                        "_target_audio_path"
                    ),
                    "error": repr(e),
                }
            )

            print(
                "[PROSODY ERROR] "
                f"{sample.get('_target_audio_path')}: "
                f"{e}",
                flush=True,
            )
            continue

        raw_prosody_cache[i] = raw

        per_speaker[
            sample[
                "_target_speaker_raw"
            ]
        ].append(raw)

        prosody_success += 1


    print(
    "\n=== PROSODY EXTRACTION ===",
    flush=True,
    )
    print(
    f"success: {prosody_success}/"
    f"{len(all_samples)}",
    flush=True,
    )
    print(
    f"failed: {len(prosody_failures)}",
    flush=True,
    )

    if prosody_failures:
        raise RuntimeError(
            f"Prosody extraction failed for "
            f"{len(prosody_failures)} / "
            f"{len(all_samples)} samples. "
            "Manifest was NOT written."
        )
    
    normalizer = SpeakerNormalizer()
    normalizer.fit(per_speaker)

    with open(
        args.out_manifest,
        "w",
        encoding="utf-8",
    ) as f:
        for i, sample in enumerate(
            all_samples
        ):
            import numpy as np

            feat_path = os.path.join(
                out_features_dir,
                f"{sample['dialogue_id']}_{i}.npy",
            )

            np.save(
                feat_path,
                sample.pop(
                    "_reaction_features"
                ),
            )

            sample[
                "listener_reaction_path"
            ] = feat_path

            raw = raw_prosody_cache.get(i)

            sample["target_prosody"] = (
                normalizer.normalize(
                    sample[
                        "_target_speaker_raw"
                    ],
                    raw,
                )
            )

            sample.pop(
                "_target_audio_path",
                None,
            )
            sample.pop(
                "_target_speaker_raw",
                None,
            )

            f.write(
                json.dumps(sample) + "\n"
            )

    print(
        f"Wrote {len(all_samples)} dyadic samples "
        f"to {args.out_manifest}"
    )

    print("\n=== DEBUG FILTER STATS ===")
    print(
        "min_track_visibility_ratio: "
        f"{args.min_track_visibility_ratio}"
    )
    print(
        "min_reaction_valid_ratio: "
        f"{args.min_reaction_valid_ratio}"
    )
    print(
        "identity_match_threshold: "
        f"{args.identity_match_threshold}"
    )

    for key, value in sorted(
        DEBUG_STATS.items()
    ):
        print(f"{key}: {value}")

    print("==========================")

    print_reaction_histogram(
        args.reaction_num_frames
    )
    print_identity_histogram()


if __name__ == "__main__":
    main()