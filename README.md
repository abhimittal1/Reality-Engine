# Reality Engine

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

A real-time, modular computer vision system that turns a webcam feed into an interactive visual effects canvas manipulated entirely by physical hand gestures.

## Features

Reality Engine implements five interactive visual effects powered by gesture recognition:

1. **Black Hole (Gravitational Lensing):** Spawn a localized gravitational distortion with a pinch gesture. The black hole pulls in surrounding light, generating an event horizon and an accretion disk of orbiting particles.
2. **Doctor Strange Portal:** Open a circular gateway to another dimension using a two-handed circle gesture. The portal is framed by a rotating, emitting spark boundary.
3. **Invisible Portal:** By capturing a background frame and combining it with MediaPipe Selfie Segmentation, the invisible portal cloaks only the human body entering the portal while leaving objects and background unaffected.
4. **Freeze Time:** Point and close your fist to extract and instantly "freeze" a segmented cutout of a human or object in the scene while the live camera continues behind it.
5. **Telekinesis:** Use a fast YOLO object detector to select items in the room. Pinch to grab the object, levitate it with simulated spring physics, and move it around the screen.

## How it Works

The application operates through a unified real-time pipeline:
1. **Camera Input:** Multi-threaded webcam capture (with automatic AVFoundation support on macOS) to ensure zero lag.
2. **Computer Vision:** `mediapipe.tasks.python.vision.HandLandmarker` tracks 21 hand joints, while `ImageSegmenter` and `yolov8n` handle human and object masking.
3. **Gesture Engine:** A debounced state machine translates raw landmarks into semantic events (Pinch, Fist, Open Palm, Point, Circle).
4. **Effect Engine:** Independent visual effects modules consume state and render composite outputs using `cv2.remap` and vectorized `ParticleSystem` physics.

## Installation & Prerequisites

**Prerequisites:**
- Python 3.9 or higher
- A working webcam

**Setup with `uv` (Recommended):**
```bash
# Create and activate a virtual environment
uv venv
source .venv/bin/activate

# Install the application and its dependencies
uv pip install -e .
```

**Setup with standard `pip`:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -e .
```

*Note: MediaPipe `.task` models and YOLO `.pt` weights are downloaded automatically on first run.*

## Usage

Start the main application loop:
```bash
python -m reality_engine.main
```

### Controls

The engine relies on hand gestures, but keyboard fallbacks and system toggles are available:
- `1`: Black Hole
- `2`: Portal
- `3`: Invisible Portal
- `4`: Freeze Time
- `5`: Telekinesis
- `0`: Disable Powers
- `b`: Capture Background (Required for Invisible Portal effect)
- `d`: Toggle Debug HUD (shows FPS, gestures, bounding boxes, and hand skeleton)
- `q`: Exit Application