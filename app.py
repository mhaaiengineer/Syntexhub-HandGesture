from __future__ import annotations

import time
from collections import deque
from typing import Optional, Tuple

import cv2
import mediapipe as mp

from .actions import perform
from .config import GESTURE_TO_ACTION, RuntimeConfig
from .gestures import classify, get_default_scale
from .utils import FPSCounter, GestureSmoother, draw_label

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

class HandGestureApp:
    def __init__(self, cfg: RuntimeConfig):
        self.cfg = cfg
        self.cap = cv2.VideoCapture(cfg.camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.frame_height)

        self.fps = FPSCounter()
        self.show_landmarks = True
        self.debug_mode = True
        self.armed = cfg.start_armed

        self.last_action_t = 0.0

        # per-hand smoothing (max 2)
        self.smoothers = [GestureSmoother(cfg.history_len), GestureSmoother(cfg.history_len)]

        # calibration
        self.calibrating = False
        self.calib_frames_left = 0
        self.calib_accum = 0.0
        self.calibrated_scale: Optional[float] = None

    def _toggle_calibration(self):
        self.calibrating = True
        self.calib_frames_left = 25
        self.calib_accum = 0.0

    def _handle_keys(self, key: int):
        if key in (ord('q'), 27):  # q or ESC
            return False
        if key == ord('s'):
            self.show_landmarks = not self.show_landmarks
        if key == ord('d'):
            self.debug_mode = not self.debug_mode
        if key == ord('a'):
            self.armed = not self.armed
        if key == ord('c'):
            self._toggle_calibration()
        return True

    def _hand_allowed(self, handed_label: str) -> bool:
        if self.cfg.active_hand.lower() == "any":
            return True
        return handed_label.lower() == self.cfg.active_hand.lower()

    def run(self):
        print("Controls: A=Arm/Disarm, C=Calibrate, D=Debug, S=Landmarks, Q=Quit")

        with mp_hands.Hands(
            max_num_hands=self.cfg.max_num_hands,
            model_complexity=self.cfg.model_complexity,
            min_detection_confidence=self.cfg.min_detection_confidence,
            min_tracking_confidence=self.cfg.min_tracking_confidence,
        ) as hands:
            while True:
                ok, frame = self.cap.read()
                if not ok:
                    print("Camera read error.")
                    break

                if self.cfg.mirror:
                    frame = cv2.flip(frame, 1)
                h, w, _ = frame.shape
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)

                # HUD (top-left)
                draw_label(frame, f"FPS: {self.fps.tick():.1f}", 10, 28, bg=(20, 20, 20))
                draw_label(frame, f"ACTIONS: {'ON' if self.armed else 'OFF'}", 10, 56, bg=(60, 20, 20) if self.armed else (20, 60, 20))

                if self.calibrating:
                    draw_label(frame, f"Calibrating... {self.calib_frames_left}", 10, 84, bg=(20, 20, 80))

                if results.multi_hand_landmarks and results.multi_handedness:
                    for idx, (hand_lm, handed) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
                        if idx > 1:
                            break
                        handed_label = handed.classification[0].label  # "Left" or "Right"
                        score = handed.classification[0].score

                        # scale
                        scale = get_default_scale(hand_lm)
                        if self.calibrated_scale:
                            rel_scale = self.calibrated_scale
                        else:
                            rel_scale = scale

                        gesture, dbg = classify(
                            hand_lm,
                            rel_scale=rel_scale,
                            fist_rel_thresh=self.cfg.fist_rel_thresh,
                            ok_rel_thresh=self.cfg.ok_rel_thresh,
                        )

                        self.smoothers[idx].push(gesture)
                        smooth_g, votes = self.smoothers[idx].majority()
                        if votes < self.cfg.majority_min_count:
                            smooth_g = "Unknown"

                        action_token = GESTURE_TO_ACTION.get(smooth_g, "NONE")

                        # calibration accumulation
                        if self.calibrating:
                            self.calib_accum += scale
                            self.calib_frames_left -= 1
                            if self.calib_frames_left <= 0:
                                self.calibrated_scale = self.calib_accum / 25.0
                                self.calibrating = False

                        # draw bbox + label
                        xs = [lm.x for lm in hand_lm.landmark]
                        ys = [lm.y for lm in hand_lm.landmark]
                        x_min, x_max = int(min(xs) * w) - 10, int(max(xs) * w) + 10
                        y_min, y_max = int(min(ys) * h) - 30, int(max(ys) * h) + 10
                        x_min, y_min = max(0, x_min), max(0, y_min)

                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 180, 0), 2)
                        label = f"{handed_label} ({int(score*100)}%) | {smooth_g} | {action_token}"
                        cv2.putText(frame, label, (x_min + 6, y_min + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

                        if self.show_landmarks:
                            mp_drawing.draw_landmarks(frame, hand_lm, mp_hands.HAND_CONNECTIONS)

                        if self.debug_mode:
                            y0 = y_max + 22
                            for k in ("d_index", "d_middle", "d_ring", "d_pinky", "d_thumb_index"):
                                if k in dbg.values:
                                    cv2.putText(frame, f"{k}: {dbg.values[k]:.3f}", (x_min + 6, y0),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
                                    y0 += 18

                        # perform action
                        now = time.time()
                        if self.armed and self._hand_allowed(handed_label):
                            if action_token != "NONE" and (now - self.last_action_t) > self.cfg.action_cooldown_s:
                                perform(action_token)
                                self.last_action_t = now

                cv2.imshow("Hand Gesture Recognition", frame)
                key = cv2.waitKey(1) & 0xFF
                if not self._handle_keys(key):
                    break

        self.cap.release()
        cv2.destroyAllWindows()
