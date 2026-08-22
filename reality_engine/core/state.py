from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List, Tuple, Dict, Any
import numpy as np

class PowerMode(Enum):
    NONE = auto()
    BLACK_HOLE = auto()
    PORTAL = auto()
    INVISIBLE_PORTAL = auto()
    FREEZE_TIME = auto()
    TELEKINESIS = auto()

class GestureType(Enum):
    NONE = auto()
    PINCH = auto()
    OPEN_PALM = auto()
    FIST = auto()
    POINT = auto()
    CIRCLE = auto()
    TWO_HAND_EXPAND = auto()

@dataclass
class HandData:
    landmarks: np.ndarray             # (21, 3) normalized coords
    pixel_landmarks: np.ndarray       # (21, 2) pixel coords
    handedness: str                   # 'Left' or 'Right'
    palm_center: Tuple[int, int]
    pinch_distance: float
    is_pinching: bool
    is_fist: bool
    is_open_palm: bool
    is_pointing: bool
    velocity: Tuple[float, float]

@dataclass
class GestureEvent:
    type: GestureType
    hand: str                         # 'Left', 'Right', 'Both'
    position: Tuple[int, int]
    velocity: Tuple[float, float]
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class FrozenObject:
    id: str
    image: np.ndarray                 # BGR or BGRA crop
    mask: np.ndarray                  # Binary mask
    bbox: Tuple[int, int, int, int]   # x, y, w, h
    position: Tuple[int, int]         # Current render position
    scale: float = 1.0
    rotation: float = 0.0

@dataclass
class AppState:
    frame: Optional[np.ndarray] = None
    timestamp: float = 0.0
    fps: float = 0.0
    hands: List[HandData] = field(default_factory=list)
    recent_gestures: List[GestureEvent] = field(default_factory=list)
    active_power: PowerMode = PowerMode.NONE
    background_frame: Optional[np.ndarray] = None
    human_mask: Optional[np.ndarray] = None
    frozen_objects: List[FrozenObject] = field(default_factory=list)
    detected_objects: List[Any] = field(default_factory=list)
    debug_mode: bool = True
