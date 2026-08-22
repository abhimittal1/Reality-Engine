import cv2
import numpy as np
import time
import math
from typing import Optional, Tuple

from reality_engine.effects.base import BaseEffect
from reality_engine.core.state import AppState, GestureType

class TelekinesisEffect(BaseEffect):
    def __init__(self):
        self.grabbed = False
        self.target_pos = (0, 0)
        self.current_pos = (0, 0)
        
        self.sprite: Optional[np.ndarray] = None
        self.mask: Optional[np.ndarray] = None
        self.orig_bbox: Optional[Tuple[int, int, int, int]] = None
        self.grab_offset = (0, 0)
        
        self.alpha = 0.1 # Spring constant
        
    def reset(self) -> None:
        self.grabbed = False
        self.sprite = None
        self.mask = None
        self.orig_bbox = None

    def update(self, state: AppState, dt: float) -> None:
        for event in state.recent_gestures:
            if event.type == GestureType.PINCH and not self.grabbed:
                self._try_grab(state, event.position)
            elif event.type == GestureType.OPEN_PALM and self.grabbed:
                self.reset()
                
        if self.grabbed:
            for hand in state.hands:
                if hand.is_pinching:
                    # Target position = hand position + grab offset
                    self.target_pos = (
                        hand.palm_center[0] - self.grab_offset[0],
                        hand.palm_center[1] - self.grab_offset[1]
                    )
                    break
                    
            # Spring physics update
            # P_curr = P_curr + alpha * (P_target - P_curr)
            dx = self.target_pos[0] - self.current_pos[0]
            dy = self.target_pos[1] - self.current_pos[1]
            
            # Subtle bobbing (levitation feel)
            bobbing_y = math.sin(state.timestamp * 5.0) * 5.0 if dx**2 + dy**2 < 1000 else 0
            
            self.current_pos = (
                self.current_pos[0] + dx * self.alpha,
                self.current_pos[1] + dy * self.alpha + bobbing_y * 0.1
            )
            
    def _try_grab(self, state, hand_pos):
        # We need detected objects to grab
        if not hasattr(state, 'detected_objects') or not state.detected_objects:
            return
            
        hx, hy = hand_pos
        for obj in state.detected_objects:
            x, y, w, h = obj.bbox
            if x <= hx <= x + w and y <= hy <= y + h:
                # Grabbed!
                self.grabbed = True
                self.orig_bbox = obj.bbox
                self.sprite = state.frame[y:y+h, x:x+w].copy()
                
                # Simple mask based on grab_cut or just bounding box for V1
                # To keep it fast, we'll use an elliptical mask
                self.mask = np.zeros((h, w), dtype=np.float32)
                cv2.ellipse(self.mask, (w//2, h//2), (w//2, h//2), 0, 0, 360, 1.0, -1)
                self.mask = cv2.GaussianBlur(self.mask, (15, 15), 0)
                
                self.current_pos = (float(x), float(y))
                self.target_pos = self.current_pos
                self.grab_offset = (hx - x, hy - y)
                break

    def render(self, frame: np.ndarray, state: AppState) -> np.ndarray:
        # If we have an object grabbed, we should inpaint its original location
        # and render it at its current location
        
        if self.grabbed and self.sprite is not None and self.orig_bbox is not None:
            ox, oy, ow, oh = self.orig_bbox
            h_im, w_im = frame.shape[:2]
            
            # 1. Inpaint original hole
            if state.background_frame is None:
                # Just blur the hole if no background
                roi = frame[oy:oy+oh, ox:ox+ow]
                frame[oy:oy+oh, ox:ox+ow] = cv2.GaussianBlur(roi, (51, 51), 0)
            else:
                bg = cv2.resize(state.background_frame, (w_im, h_im))
                frame[oy:oy+oh, ox:ox+ow] = bg[oy:oy+oh, ox:ox+ow]
                
            # 2. Render floating object
            cx, cy = int(self.current_pos[0]), int(self.current_pos[1])
            x1 = max(0, cx)
            y1 = max(0, cy)
            x2 = min(w_im, cx + ow)
            y2 = min(h_im, cy + oh)
            
            if x1 < x2 and y1 < y2:
                crop_x1 = x1 - cx
                crop_y1 = y1 - cy
                crop_x2 = crop_x1 + (x2 - x1)
                crop_y2 = crop_y1 + (y2 - y1)
                
                obj_img = self.sprite[crop_y1:crop_y2, crop_x1:crop_x2]
                obj_mask = self.mask[crop_y1:crop_y2, crop_x1:crop_x2]
                
                mask_3d = np.stack([obj_mask]*3, axis=2)
                roi = frame[y1:y2, x1:x2]
                frame[y1:y2, x1:x2] = roi * (1 - mask_3d) + obj_img * mask_3d
                
        return frame.astype(np.uint8)
