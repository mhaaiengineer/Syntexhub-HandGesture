from __future__ import annotations

from src.app import HandGestureApp
from src.config import RuntimeConfig

if __name__ == "__main__":
    app = HandGestureApp(RuntimeConfig())
    app.run()
