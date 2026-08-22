# Product Requirements Document: Reality Engine

| | |
|---|---|
| **Status** | Draft v1.0 |
| **Owner** | Abhishek |
| **Last Updated** | August 21, 2026 |
| **Document Type** | PRD — Product & Technical Requirements |

---

## 1. Executive Summary

Reality Engine is a real-time computer vision application, built in Python, that turns a standard webcam into an interactive canvas the user can manipulate with their hands. Using hand tracking, gesture recognition, segmentation, and geometric transforms, the system layers a set of visual "powers" — a Doctor Strange–style portal, an invisibility effect, time-freeze, telekinesis, and a gravitational black hole — onto the live camera feed.

The project is scoped as a single, modular Python application (not five disconnected demos), built incrementally from a working camera pipeline through to a full effects engine. It is local-first: no mobile app, cloud backend, or multi-user support in V1.

---

## 2. Problem Statement

Most beginner-to-intermediate computer vision projects (face detection, object detection, hand tracking, pose estimation) are effective learning exercises but produce forgettable, disconnected demos. They rarely combine into something a viewer finds genuinely surprising or worth sharing.

Reality Engine addresses this gap by unifying several CV capabilities behind one premise: **the user's physical movements directly and visibly manipulate the reality shown on screen.** The system should feel responsive and visually impressive on its own merits, while remaining modular enough that each capability can be explained, tested, and extended independently.

---

## 3. Product Vision

Build a real-time "reality manipulation engine" where the webcam is a window into a world the user can reshape with gestures.

**High-level data flow:**

```
Real World → Webcam → Computer Vision (Hands / Humans / Objects)
           → Gesture Engine → Reality Engine
           → Effects (Portal, Freeze Time, Telekinesis, Black Hole)
           → Final Renderer → Screen
```

---

## 4. Target Users

| User | Description |
|---|---|
| **Primary** | Developers, students, AI/ML engineers, and CV enthusiasts building or extending the project. |
| **Secondary** | Viewers of a demo video, who should understand the effect intuitively without needing to understand the underlying technology. |

**Target reaction:** *"Wait — this is running from a Python program and a webcam?"*

---

## 5. Goals

### 5.1 Product Goals
- Real-time webcam application with smooth, interactive performance.
- Reliable detection and tracking of one or more hands.
- A custom gesture-recognition layer mapped to discrete actions.
- A modular effects architecture — new powers can be added without touching the core system.
- Visually polished output suitable for demos and portfolio use.

### 5.2 Learning Goals
Practical depth in: Python, OpenCV, NumPy, hand tracking, gesture recognition, object detection, image segmentation, masking, alpha compositing, coordinate systems, geometric transforms, real-time rendering, state management, and performance optimization.

### 5.3 Non-Goals (V1)
Explicitly out of scope for the first version:
- Mobile or web application
- Cloud backend, authentication, or database
- SaaS or multiplayer features
- AR glasses support
- Full 3D physics or high-fidelity depth estimation
- Film-grade VFX quality

Reality Engine V1 is a **local-first desktop Python application**.

---

## 6. Core User Experience

The application opens directly to a live camera view with a minimal on-screen status indicator (active power, debug info). The system continuously tracks hands, fingers, gestures, movement velocity, and — depending on the active effect — a human-segmentation mask or selected object. Gestures drive state transitions; there is no menu-driven or form-based interaction model.

---

## 7. Feature Specifications

### 7.1 Hand Tracking & Gesture Engine *(Foundational)*

The core input layer. Converts raw hand landmarks into high-level, named gesture events consumed by every other module.

| Gesture | Action |
|---|---|
| Pinch | Grab / create |
| Open palm | Release / activate |
| Fist | Freeze / destroy |
| Point | Select object |
| Hand movement | Move object or effect |
| Circular motion | Create portal |
| Two-hand movement | Resize portal |

**Requirements**
- Detect and distinguish left/right hands.
- Track landmark positions and frame-to-frame velocity.
- Recognize the gesture set above with debounce logic to prevent repeated/accidental triggers.

**Pipeline:** `Camera Frame → Hand Tracker → 21 Landmarks → Gesture Analyzer → GestureEvent`

```python
GestureEvent(type="PINCH", hand="RIGHT", position=(x, y), velocity=(vx, vy))
```

---

### 7.2 Portal Effect (Doctor Strange–style)

A circular opening in the live feed, positioned and resized by hand gestures, that reveals alternate content.

**User flow:** hand movement tracked → circular gesture detected → portal created → user repositions/resizes → animated border and particles render on top.

| Scope | Capability |
|---|---|
| **V1** | Circular portal that follows the hand; gesture-based resize; animated rotating border; particle effects; displays a predefined image or video. |
| **V2** | Portal renders a second live camera source; perspective-correct placement on a detected surface; objects can be moved through the portal. |

**Technical components:** portal center, portal radius, circular mask, portal content buffer, alpha compositing, particle renderer.

---

### 7.3 Invisible Portal

Creates the illusion that a person or object disappears within a defined region by substituting a previously captured background.

**User flow:** capture an empty-scene background → user enters frame → portal region created → pixels inside the region are replaced with the stored background.

| Scope | Capability |
|---|---|
| **V1** | Simple circular mask: `portal_area = background_frame`. |
| **Advanced** | Combine with human segmentation so only the person/object intersecting the mask is removed, rather than the full masked region. |

---

### 7.4 Freeze Time

Captures a selected object or person as a static image layer while the rest of the scene continues moving live.

**User flow:** object detected → user points to select → user closes fist to capture → object and mask extracted → stored as a frozen layer → rendered over all subsequent frames.

```python
FrozenObject(image, mask, position, timestamp, rotation, scale)
```

**Requirements:** support freezing single or multiple objects, freezing the user, removing a frozen object, and repositioning a frozen object after capture.

---

### 7.5 Telekinesis

Lets the user select an object and move it independently through the scene, as if levitating it.

**User flow:** point to select → pinch to grab → object attaches to the virtual hand → hand movement transforms the object → open palm releases it.

```python
ObjectTransform(x, y, rotation, scale)
```

**Requirements:** object selection, segmentation-based extraction, position/scale/rotation manipulation, smooth motion, and a deliberate movement delay to sell the "telekinetic" feel.

---

### 7.6 Real-Life Black Hole

Simulates gravitational lensing: a moving point that visibly warps and pulls in nearby pixels.

**User flow:** pinch to spawn → hand movement relocates the black hole → surrounding pixels distort continuously → fist destroys it.

**Technical approach:** for each pixel, compute distance from the black hole center, calculate a distortion offset, and remap the source image using `cv2.remap()` with NumPy-generated coordinate maps.

---

## 8. System Architecture

```
                    ┌─────────────┐
                    │   WEBCAM    │
                    └──────┬──────┘
                           ▼
                    ┌─────────────┐
                    │ FRAME INPUT │
                    └──────┬──────┘
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
    HAND TRACKER     OBJECT DETECTOR   SEGMENTATION
          └────────────────┼────────────────┘
                           ▼
                    ┌─────────────┐
                    │GESTURE ENGINE│
                    └──────┬──────┘
                           ▼
                    ┌─────────────┐
                    │  APP STATE  │
                    └──────┬──────┘
              ┌────────────┼────────────┐
              ▼            ▼            ▼
           PORTAL        FREEZE      TELEKINESIS
              └────────────┼────────────┘
                           ▼
                      BLACK HOLE
                           ▼
                    ┌─────────────┐
                    │  RENDERER   │
                    └──────┬──────┘
                           ▼
                       DISPLAY
```

**Governing principle:** vision modules detect reality; effects consume that state but never run their own CV models. This avoids duplicate computation and keeps effects independently testable.

```
HandTracker → AppState → {Portal, BlackHole, Telekinesis, ...}
```

### 8.1 Module Layout

```
camera/
  camera.py            # webcam I/O, resolution, FPS, buffering
vision/
  hand_tracker.py
  object_detector.py
  human_segmenter.py
  scene_analyzer.py
gestures/
  pinch.py
  fist.py
  open_hand.py
  swipe.py
  circle.py
effects/
  base.py
  portal.py
  invisible_portal.py
  freeze_time.py
  telekinesis.py
  black_hole.py
```

**Effect interface:**
```python
class BaseEffect:
    def update(self, state): pass
    def render(self, frame): return frame
```

### 8.2 Application & Effect State

```python
class AppState:
    frame
    timestamp
    hands
    gestures
    detected_objects
    selected_object
    active_effects
    background_frame
    frozen_objects
    portal_state
    black_hole_state
```

```python
PortalState(active=False, center=None, radius=0, rotation=0)
BlackHoleState(active=False, center=None, radius=0, strength=0)
TelekinesisState(selected_object=None, grabbed=False, offset=None)
frozen_objects: list[FrozenObject]
```

### 8.3 Rendering Order

```
1. Camera frame
2. Background effects
3. Black hole distortion
4. Invisible portal
5. Frozen objects
6. Telekinesis objects
7. Portal
8. Particles
9. UI / debug overlay
```

A future layer system (`Camera → Background → Distortion → Objects → Portals → Particles → UI`) is recommended once more than two or three effects are active simultaneously.

---

## 9. Technology Stack

| Layer | Choice |
|---|---|
| Core | Python 3.11+, OpenCV, NumPy |
| Computer vision | MediaPipe (hands), YOLO (objects, added later), a segmentation model (added later) |
| Optional advanced graphics | PyOpenGL / ModernGL, PyGame |

**Recommendation:** start with Python + OpenCV + NumPy + MediaPipe only. Add object detection and segmentation when a specific feature phase requires them — avoid front-loading the full model stack.

---

## 10. Non-Functional Requirements

| Requirement | Target |
|---|---|
| Input resolution | 720p |
| Minimum usable frame rate | 20 FPS |
| Target frame rate | 30 FPS |
| Gesture-to-response latency | < 100 ms |
| Concurrent users in frame | 1 (primary) |

**Optimization strategy:** not every module needs to run every frame.

| Module | Suggested cadence |
|---|---|
| Hand tracking | Every frame |
| Object detection | Every 5–10 frames |
| Segmentation | On demand, only when an active effect needs it |

---

## 11. Risks & Assumptions

| Risk | Mitigation |
|---|---|
| Segmentation/object detection may be too slow for real-time use on modest hardware | Run at reduced cadence (Section 10); consider lighter model variants. |
| Gesture recognition may misfire in cluttered or low-light scenes | Add debounce/hysteresis logic; define minimum-confidence thresholds. |
| Effect layering (Section 8.3) may produce visual artifacts as more effects run concurrently | Introduce the explicit layer/compositing system once beyond 2–3 simultaneous effects. |
| Scope creep across six features could stall the MVP | Hold strictly to the phased roadmap (Section 12) and MVP checklist (Section 13). |

**Assumptions:** single-user, front-facing webcam setup; development and demo environment has adequate lighting; target hardware has a discrete or capable integrated GPU for real-time inference.

---

## 12. Development Roadmap

| Phase | Focus | Key Deliverables |
|---|---|---|
| **0 — Setup** | Working pipeline | Project structure, virtual environment, OpenCV camera input, FPS counter |
| **1 — Hand Tracking** | Foundational input | Landmarks, left/right detection, hand center, pinch/open/fist detection, velocity |
| **2 — Gesture Engine** | Reusable event system | `GestureEvent` model consumed by all later phases |
| **3 — Black Hole** | First visual effect | Spawn, move, destroy, pixel distortion, rotation, particles |
| **4 — Portal** | Second effect | Circle follows hand, resize, animated border, content rendering, particles |
| **5 — Invisible Portal** | Segmentation-driven effect | Background capture, circular mask, background replacement, accurate body removal |
| **6 — Freeze Time** | Layered compositing | Person freeze, capture, mask extraction, persistent rendering, multi-object support |
| **7 — Telekinesis** | Object manipulation | Detection, select/grab, extraction, move/release, scale and rotation |

**Success criterion per phase (example — hand tracking):** gesture is detected reliably → application logs the event → a visual marker renders on the tracked hand.

---

## 13. MVP Definition

The MVP intentionally favors breadth over polish — a working version of every core power rather than a perfected version of one.

- **Gesture Engine:** detects pinch, fist, and open palm; tracks hand position.
- **Black Hole:** can be created, follows the hand, visibly distorts nearby pixels.
- **Portal:** appears via gesture, can be moved, displays alternate content.
- **Invisible Portal:** background can be captured; portal reveals it correctly.
- **Freeze Time:** user can freeze their own image.
- **Telekinesis:** user can select and move at least one detected object.

---

## 14. Success Metrics

| Metric | Target |
|---|---|
| Sustained frame rate during single-effect use | ≥ 20 FPS (720p) |
| Gesture recognition reliability | Consistent triggering across repeated attempts in normal lighting |
| Phase completion | All 7 phases reach their stated success criteria |
| Demo reaction (qualitative) | A first-time viewer identifies the effect as "impossible-looking" without prompting |

---

## 15. Future Vision (Post-MVP)

Once the core engine is stable, additional powers can be added as independent effect modules without touching the core pipeline: fireball, lightning, ice freeze, ghost clone, delayed mirror, reality warp, multiple simultaneous black holes, time rewind, an independently-moving shadow, face transformation, and a portal to a second physical location.

---

## 16. Summary

Reality Engine is not a CRUD application with forms, accounts, or dashboards — it is a real-time interactive computer vision engine. The interaction model is intentionally simple:

```
Move hands → Recognize gestures → Trigger powers → Manipulate the visual world
```

**Recommended next step:** translate this PRD into a detailed technical architecture and implementation plan — exact library choices, data-flow contracts between modules, effect lifecycle hooks, the gesture state machine, and the rendering/compositing pipeline — then build sequentially from Phase 0 through Phase 7.