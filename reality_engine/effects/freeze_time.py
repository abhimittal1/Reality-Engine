import cv2
import numpy as np
import uuid
from typing import Optional, Tuple

from reality_engine.effects.base import BaseEffect
from reality_engine.core.state import AppState, GestureType, FrozenObject

class FreezeTimeEffect(BaseEffect):
    def __init__(self):
        pass

    def reset(self) -> None:
        pass

    def update(self, state: AppState, dt: float) -> None:
        # We listen for FIST to capture if we're in this mode
        # Also OPEN_PALM to clear all frozen objects for simplicity
        
        for event in state.recent_gestures:
            if event.type == GestureType.FIST:
                self._capture_freeze(state)
            elif event.type == GestureType.OPEN_PALM:
                state.frozen_objects.clear()

    def _capture_freeze(self, state: AppState):
        if state.human_mask is None or state.frame is None:
            return
            
        # Ensure mask is binary
        _, mask_bin = cv2.threshold(state.human_mask, 0.5, 1.0, cv2.THRESH_BINARY)
        if np.sum(mask_bin) < 100:
            return # Too small to freeze
            
        # Find bounding box
        coords = cv2.findNonZero((mask_bin * 255).astype(np.uint8))
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            # Add some padding
            pad = 20
            h_im, w_im = state.frame.shape[:2]
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(w_im, x + w + pad)
            y2 = min(h_im, y + h + pad)
            
            crop_img = state.frame[y1:y2, x1:x2].copy()
            crop_mask = mask_bin[y1:y2, x1:x2].copy()
            
            frozen = FrozenObject(
                id=str(uuid.uuid4()),
                image=crop_img,
                mask=crop_mask,
                bbox=(x1, y1, x2-x1, y2-y1),
                position=(x1, y1),
                scale=1.0,
                rotation=0.0
            )
            state.frozen_objects.append(frozen)

    def render(self, frame: np.ndarray, state: AppState) -> np.ndarray:
        # Render all frozen objects
        h_im, w_im = frame.shape[:2]
        
        for obj in state.frozen_objects:
            ox, oy = obj.position
            ow, oh = obj.bbox[2:]
            
            x1 = max(0, ox)
            y1 = max(0, oy)
            x2 = min(w_im, ox + ow)
            y2 = min(h_im, oy + oh)
            
            if x1 >= x2 or y1 >= y2:
                continue
                
            crop_x1 = x1 - ox
            crop_y1 = y1 - oy
            crop_x2 = crop_x1 + (x2 - x1)
            crop_y2 = crop_y1 + (y2 - y1)
            
            obj_img = obj.image[crop_y1:crop_y2, crop_x1:crop_x2]
            obj_mask = obj.mask[crop_y1:crop_y2, crop_x1:crop_x2]
            
            mask_3d = np.stack([obj_mask]*3, axis=2)
            
            roi = frame[y1:y2, x1:x2]
            frame[y1:y2, x1:x2] = roi * (1 - mask_3d) + obj_img * mask_3d
            
        return frame.astype(np.uint8)
