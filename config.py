from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

@dataclass(frozen=True)
class RuntimeConfig:
    # MediaPipe
    max_num_hands: int = 2
    model_complexity: int = 1
    min_detection_confidence: float = 0.6
    min_tracking_confidence: float = 0.6

    # Video
    camera_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    mirror: bool = True  # flip horizontally

    # Gesture smoothing + action rate limiting
    history_len: int = 7
    majority_min_count: int = 4  # votes required to accept a gesture
    action_cooldown_s: float = 1.1

    # Gesture thresholds (relative to hand scale)
    fist_rel_thresh: float = 0.60  # smaller => stricter
    ok_rel_thresh: float = 0.22    # smaller => stricter

    # Action safety
    start_armed: bool = False
    active_hand: str = "Right"  # "Right", "Left", or "Any" (which hand can trigger actions)

# Default mapping: gesture -> action token
GESTURE_TO_ACTION: Dict[str, str] = {
    "Open Palm": "PLAY_PAUSE",
    "Peace (V)": "NEXT",
    "Fist": "PREV",
    "Thumbs Up": "VOL_UP",
    "Thumbs Down": "VOL_DOWN",
    "OK": "CONFIRM",
    "Pointing": "CLICK",
    "Unknown": "NONE",
}
