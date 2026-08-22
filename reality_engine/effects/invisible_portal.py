import cv2
import numpy as np
from typing import Optional, Tuple

from reality_engine.effects.base import BaseEffect
from reality_engine.core.state import AppState, GestureType

class InvisiblePortalEffect(BaseEffect):
    def __init__(self):
        self.active = False
        self.center: Optional[Tuple[int, int]] = None
        self.radius = 200

    def reset(self) -> None:
        self.active = False
        self.center = None

    def update(self, state: AppState, dt: float) -> None:
        for event in state.recent_gestures:
            if event.type == GestureType.CIRCLE:
                self.active = True
                self.center = event.position
            elif event.type == GestureType.FIST:
                self.active = False
                
        if self.active:
            for hand in state.hands:
                if hand.is_pinching:
                    self.center = hand.palm_center
                    break

    def render(self, frame: np.ndarray, state: AppState) -> np.ndarray:
        if not self.active or self.center is None or state.background_frame is None:
            # Cannot render without a captured background
            if self.active and state.background_frame is None and state.debug_mode:
                cv2.putText(frame, "NO BACKGROUND! PRESS 'b'", (10, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            return frame
            
        h, w = frame.shape[:2]
        
        # Base circular portal mask
        portal_mask = np.zeros((h, w), dtype=np.float32)
        cv2.circle(portal_mask, self.center, self.radius, 1.0, -1)
        portal_mask = cv2.GaussianBlur(portal_mask, (21, 21), 0)
        
        # If we have a human mask from segmentation, we only apply invisibility to the human IN the portal
        if state.human_mask is not None:
            # Combine masks: Portal area AND Human presence
            final_mask = portal_mask * state.human_mask
        else:
            # Basic mode: full portal region is invisible
            final_mask = portal_mask
            
        mask_3d = np.stack([final_mask, final_mask, final_mask], axis=2)
        
        bg = cv2.resize(state.background_frame, (w, h))
        
        frame = frame * (1.0 - mask_3d) + bg * mask_3d
        return frame.astype(np.uint8)
