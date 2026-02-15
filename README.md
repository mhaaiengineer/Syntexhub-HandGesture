# Syntexhub HandGesture
Real-time AI hand gesture recognition using MediaPipe + OpenCV. Detects hand landmarks, classifies common gestures (Open Palm, Peace, Fist, Thumbs Up/Down, OK, Pointing), and triggers system actions via PyAutoGUI (play/pause, volume, next/prev, click, etc.).
# ✋ AI Hand Gesture Recognition System
**Python + MediaPipe + OpenCV + PyAutoGUI**

This project detects **hand landmarks in real-time** (webcam), classifies common **static gestures** with lightweight rule-based logic, and triggers **system-level actions** like play/pause, volume control, slide navigation, and mouse clicks.

It satisfies the internship Project‑2 requirements: real-time hand landmarks, gesture classification, action mapping, and a runnable webcam demo script. 

---

## ✅ Features
- Real-time hand tracking (MediaPipe Hands)
- Gesture classification (rule-based; no training required)
- Actions via PyAutoGUI (keyboard / mouse)
- **Arm/Disarm** safety switch (actions off by default)
- Gesture smoothing (majority vote)
- Action cooldown (prevents spam)
- Optional calibration (for different hand sizes/distances)
- FPS counter + on-screen debug overlays
- Multi-hand detection (actions can be restricted to one hand)

---

## ✌️ Supported Gestures (default)
| Gesture | Meaning | Default Action |
|---|---|---|
| Open Palm | “Stop / pause” | Space (Play/Pause) |
| Peace (V) | “Next” | Right Arrow (Next slide) |
| Fist | “Back” | Left Arrow (Previous slide) |
| Thumbs Up | Increase | Volume Up |
| Thumbs Down | Decrease | Volume Down |
| OK | Confirm | Enter |
| Pointing (Index) | Select | Mouse Left Click |

> You can change all mappings in `src/config.py`.

---

## 🚀 Quick Start

### 1) Install dependencies
```bash
pip install -r requirements.txt
```

### 2) Run webcam demo (recommended)
```bash
python run_webcam.py
```

---

## 🎮 Controls (inside the window)
- **A**: Arm/Disarm actions (default: **disarmed**)
- **C**: Calibrate hand scale (helpful if fist/ok detection is inconsistent)
- **D**: Toggle debug overlay
- **S**: Toggle landmark drawing
- **Q**: Quit

---

## 📁 Project Structure
```
Syntexhub_HandGesture_Full/
  ├─ run_webcam.py
  ├─ requirements.txt
  ├─ README.md
  ├─ .gitignore
  ├─ src/
  │   ├─ app.py
  │   ├─ gestures.py
  │   ├─ actions.py
  │   ├─ config.py
  │   └─ utils.py
  └─ notebooks/
      └─ handgesture.ipynb
```

---

## ⚠️ Notes
- PyAutoGUI controls your system. Use **A** to arm only when ready.
- Keep your hand inside the frame with good lighting for best accuracy.
- If you’re on Linux, you may need extra system packages for PyAutoGUI (X11).

---

