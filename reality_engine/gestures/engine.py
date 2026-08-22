import time
from typing import Dict, Tuple

from reality_engine.config import Config
from reality_engine.core.state import AppState, GestureEvent, GestureType
from reality_engine.gestures.detectors import PinchDetector, FistDetector, OpenPalmDetector, PointDetector, CircleDetector

class GestureEngine:
    def __init__(self):
        self.detectors = {
            GestureType.PINCH: PinchDetector(),
            GestureType.FIST: FistDetector(),
            GestureType.OPEN_PALM: OpenPalmDetector(),
            GestureType.POINT: PointDetector(),
            GestureType.CIRCLE: CircleDetector(),
        }
        
        # Debounce tracking: hand_id -> {gesture_type: last_trigger_time}
        self.last_trigger: Dict[str, Dict[GestureType, float]] = {}
        
    def process(self, state: AppState):
        state.recent_gestures = []
        current_time = state.timestamp
        
        for idx, hand in enumerate(state.hands):
            h_id = f"{hand.handedness}_{idx}"
            
            if h_id not in self.last_trigger:
                self.last_trigger[h_id] = {}
                
            # Populate hand booleans
            hand.is_pinching = self.detectors[GestureType.PINCH].detect(hand, state)
            hand.is_fist = self.detectors[GestureType.FIST].detect(hand, state)
            hand.is_open_palm = self.detectors[GestureType.OPEN_PALM].detect(hand, state)
            hand.is_pointing = self.detectors[GestureType.POINT].detect(hand, state)
            
            # Fire events
            for g_type, detector in self.detectors.items():
                is_detected = False
                if g_type == GestureType.PINCH: is_detected = hand.is_pinching
                elif g_type == GestureType.FIST: is_detected = hand.is_fist
                elif g_type == GestureType.OPEN_PALM: is_detected = hand.is_open_palm
                elif g_type == GestureType.POINT: is_detected = hand.is_pointing
                elif g_type == GestureType.CIRCLE: is_detected = detector.detect(hand, state)
                
                if is_detected:
                    last_time = self.last_trigger[h_id].get(g_type, 0.0)
                    if current_time - last_time > Config.DEBOUNCE_TIME:
                        self.last_trigger[h_id][g_type] = current_time
                        
                        event = GestureEvent(
                            type=g_type,
                            hand=hand.handedness,
                            position=hand.palm_center,
                            velocity=hand.velocity
                        )
                        state.recent_gestures.append(event)
