from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

# MediaPipe landmark indices:
# https://google.github.io/mediapipe/solutions/hands.html

@dataclass
class GestureDebug:
    values: Dict[str, float]

def _euclid_xy(a, b) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)

def _hand_scale(landmarks) -> float:
    # wrist(0) -> index_tip(8) works well as a stable scale; add epsilon
    return _euclid_xy(landmarks.landmark[0], landmarks.landmark[8]) + 1e-9

def _finger_is_up(landmarks, tip_idx: int, pip_idx: int) -> bool:
    # image is mirrored in app (default), so Y still increases downward.
    return landmarks.landmark[tip_idx].y < landmarks.landmark[pip_idx].y

def classify(landmarks, rel_scale: float, fist_rel_thresh: float, ok_rel_thresh: float) -> Tuple[str, GestureDebug]:
    """Return (gesture_name, debug_values)."""
    up_index = _finger_is_up(landmarks, 8, 6)
    up_middle = _finger_is_up(landmarks, 12, 10)
    up_ring = _finger_is_up(landmarks, 16, 14)
    up_pinky = _finger_is_up(landmarks, 20, 18)

    fingers_up = sum([up_index, up_middle, up_ring, up_pinky])

    # thumb: use tip(4) vs ip(3) to detect extension in 'up' direction
    thumb_up = landmarks.landmark[4].y < landmarks.landmark[3].y

    # distances normalized by rel_scale
    d_index = _euclid_xy(landmarks.landmark[8], landmarks.landmark[0]) / rel_scale
    d_middle = _euclid_xy(landmarks.landmark[12], landmarks.landmark[0]) / rel_scale
    d_ring = _euclid_xy(landmarks.landmark[16], landmarks.landmark[0]) / rel_scale
    d_pinky = _euclid_xy(landmarks.landmark[20], landmarks.landmark[0]) / rel_scale
    d_thumb_index = _euclid_xy(landmarks.landmark[4], landmarks.landmark[8]) / rel_scale

    dbg = GestureDebug(
        values={
            "d_index": d_index,
            "d_middle": d_middle,
            "d_ring": d_ring,
            "d_pinky": d_pinky,
            "d_thumb_index": d_thumb_index,
            "scale": rel_scale,
            "fingers_up": float(fingers_up),
        }
    )

    # 1) OK: thumb + index tips close
    if d_thumb_index < ok_rel_thresh:
        return "OK", dbg

    # 2) Fist: all fingertips close to wrist
    if (d_index < fist_rel_thresh) and (d_middle < fist_rel_thresh) and (d_ring < fist_rel_thresh) and (d_pinky < fist_rel_thresh):
        return "Fist", dbg

    # 3) Peace (V): index + middle up; ring + pinky down
    if up_index and up_middle and (not up_ring) and (not up_pinky):
        return "Peace (V)", dbg

    # 4) Pointing: only index up
    if up_index and (not up_middle) and (not up_ring) and (not up_pinky):
        return "Pointing", dbg

    # 5) Open Palm: four fingers up and thumb extended
    if fingers_up == 4 and thumb_up:
        return "Open Palm", dbg

    # 6) Thumbs up/down: fingers down, thumb above or below wrist
    if (not up_index) and (not up_middle) and (not up_ring) and (not up_pinky):
        thumb_tip = landmarks.landmark[4]
        wrist = landmarks.landmark[0]
        if thumb_tip.y < wrist.y:
            return "Thumbs Up", dbg
        return "Thumbs Down", dbg

    return "Unknown", dbg

def get_default_scale(landmarks) -> float:
    return _hand_scale(landmarks)
