import cv2
import numpy as np

from reality_engine.config import Config
from reality_engine.core.state import AppState

class HUD:
    def __init__(self):
        self.font = Config.DEBUG_FONT
        self.scale = Config.DEBUG_FONT_SCALE
        self.color = Config.DEBUG_COLOR
        
    def render(self, frame: np.ndarray, state: AppState):
        h, w = frame.shape[:2]
        
        # Overlay Panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 40), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # FPS
        fps_text = f"FPS: {int(state.fps)}"
        cv2.putText(frame, fps_text, (10, 25), self.font, self.scale, self.color, 2)
        
        # Active Power
        power_text = f"POWER: {state.active_power.name}"
        text_size = cv2.getTextSize(power_text, self.font, self.scale, 2)[0]
        cv2.putText(frame, power_text, (w // 2 - text_size[0] // 2, 25), self.font, self.scale, (0, 255, 255), 2)
        
        # Controls Hint
        controls = "1-5: Powers | b: Bg | d: Debug | q: Quit"
        text_size2 = cv2.getTextSize(controls, self.font, 0.5, 1)[0]
        cv2.putText(frame, controls, (w - text_size2[0] - 10, 25), self.font, 0.5, (200, 200, 200), 1)
        
        # Gesture feedback
        if state.debug_mode:
            y_offset = 60
            for i, hand in enumerate(state.hands):
                gestures = []
                if hand.is_pinching: gestures.append("PINCH")
                if hand.is_fist: gestures.append("FIST")
                if hand.is_open_palm: gestures.append("OPEN PALM")
                if hand.is_pointing: gestures.append("POINT")
                
                info = f"{hand.handedness} Hand: {', '.join(gestures) if gestures else 'None'}"
                cv2.putText(frame, info, (10, y_offset), self.font, self.scale, (255, 100, 100), 2)
                y_offset += 30
