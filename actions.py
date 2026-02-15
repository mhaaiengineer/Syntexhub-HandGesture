from __future__ import annotations

import time
from typing import Callable, Dict

import pyautogui

# Safety: moving mouse to top-left aborts PyAutoGUI
pyautogui.FAILSAFE = True

def action_play_pause() -> None:
    pyautogui.press("space")

def action_next() -> None:
    # slides / many players
    pyautogui.press("right")

def action_prev() -> None:
    pyautogui.press("left")

def action_vol_up() -> None:
    pyautogui.press("volumeup")

def action_vol_down() -> None:
    pyautogui.press("volumedown")

def action_confirm() -> None:
    pyautogui.press("enter")

def action_click() -> None:
    pyautogui.click()

ACTIONS: Dict[str, Callable[[], None]] = {
    "PLAY_PAUSE": action_play_pause,
    "NEXT": action_next,
    "PREV": action_prev,
    "VOL_UP": action_vol_up,
    "VOL_DOWN": action_vol_down,
    "CONFIRM": action_confirm,
    "CLICK": action_click,
    "NONE": lambda: None,
}

def perform(action_token: str) -> None:
    fn = ACTIONS.get(action_token, None)
    if fn:
        fn()
