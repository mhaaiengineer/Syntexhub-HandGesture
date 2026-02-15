from __future__ import annotations

import time
from collections import Counter, deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple

import cv2

@dataclass
class FPSCounter:
    last_t: float = time.time()
    fps: float = 0.0

    def tick(self) -> float:
        now = time.time()
        dt = now - self.last_t
        self.last_t = now
        if dt > 0:
            # simple exponential smoothing
            inst = 1.0 / dt
            self.fps = 0.85 * self.fps + 0.15 * inst if self.fps else inst
        return self.fps

class GestureSmoother:
    def __init__(self, history_len: int):
        self.buf: Deque[str] = deque(maxlen=history_len)

    def push(self, gesture: str) -> None:
        self.buf.append(gesture)

    def majority(self) -> Tuple[str, int]:
        if not self.buf:
            return "Unknown", 0
        g, c = Counter(self.buf).most_common(1)[0]
        return g, c

def draw_label(
    frame,
    text: str,
    x: int,
    y: int,
    bg: Tuple[int, int, int] = (0, 0, 0),
    fg: Tuple[int, int, int] = (255, 255, 255),
) -> None:
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(frame, (x, y - th - 10), (x + tw + 12, y + 6), bg, -1)
    cv2.putText(frame, text, (x + 6, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, fg, 2, cv2.LINE_AA)

def safe_imshow(win: str, frame) -> None:
    # guards against some OpenCV errors on headless environments
    try:
        cv2.imshow(win, frame)
    except Exception:
        pass
