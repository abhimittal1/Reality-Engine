import time
import cv2
from typing import List

from reality_engine.config import Config
from reality_engine.core.state import AppState, PowerMode
from reality_engine.camera.camera import ThreadedCamera
from reality_engine.vision.hand_tracker import HandTracker
from reality_engine.vision.segmenter import HumanSegmenter
from reality_engine.vision.object_detector import ObjectDetector
from reality_engine.gestures.engine import GestureEngine
from reality_engine.effects.black_hole import BlackHoleEffect
from reality_engine.effects.portal import PortalEffect
from reality_engine.effects.invisible_portal import InvisiblePortalEffect
from reality_engine.effects.freeze_time import FreezeTimeEffect
from reality_engine.effects.telekinesis import TelekinesisEffect
from reality_engine.rendering.ui import HUD

class RealityEngine:
    def __init__(self):
        self.config = Config()
        self.state = AppState()
        self.camera = ThreadedCamera(
            src=self.config.CAMERA_ID,
            width=self.config.CAMERA_WIDTH,
            height=self.config.CAMERA_HEIGHT,
            target_fps=self.config.TARGET_FPS
        )
        
        # Modules will be initialized here as they are built
        self.hand_tracker = HandTracker()
        self.human_segmenter = HumanSegmenter()
        self.object_detector = ObjectDetector()
        self.gesture_engine = GestureEngine()
        self.hud = HUD()
        self.effects = {
            PowerMode.BLACK_HOLE: BlackHoleEffect(),
            PowerMode.PORTAL: PortalEffect(),
            PowerMode.INVISIBLE_PORTAL: InvisiblePortalEffect(),
            PowerMode.FREEZE_TIME: FreezeTimeEffect(),
            PowerMode.TELEKINESIS: TelekinesisEffect()
        }
        
        self.is_running = False
        
        # FPS calculation
        self.last_time = time.time()
        self.frame_count = 0
        
    def start(self):
        print("Starting Reality Engine...")
        self.camera.start()
        self.is_running = True
        self.run_loop()
        
    def stop(self):
        self.is_running = False
        self.camera.stop()
        cv2.destroyAllWindows()
        
    def run_loop(self):
        while self.is_running:
            grabbed, frame = self.camera.read()
            if not grabbed or frame is None:
                time.sleep(0.01)
                continue
                
            # Flip horizontally for selfie view
            frame = cv2.flip(frame, 1)
            self.state.frame = frame
            
            # Update timing
            current_time = time.time()
            dt = current_time - self.last_time
            self.last_time = current_time
            self.state.timestamp = current_time
            
            # FPS Calculation (smoothed)
            current_fps = 1.0 / dt if dt > 0 else 0
            self.state.fps = self.state.fps * 0.9 + current_fps * 0.1
            
            # Process Vision
            if self.hand_tracker:
                self.hand_tracker.process(self.state)
                
            if self.human_segmenter and self.state.active_power in (PowerMode.INVISIBLE_PORTAL, PowerMode.FREEZE_TIME):
                self.human_segmenter.process(self.state)
                
            if self.object_detector and self.state.active_power == PowerMode.TELEKINESIS:
                self.state.detected_objects = self.object_detector.process(self.state.frame)
                if self.state.debug_mode:
                    for obj in self.state.detected_objects:
                        x, y, w, h = obj.bbox
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
                        cv2.putText(frame, obj.class_name, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
            # Process Gestures
            if self.gesture_engine:
                self.gesture_engine.process(self.state)
                
            # Process Active Effect
            if self.state.active_power != PowerMode.NONE and self.state.active_power in self.effects:
                active_effect = self.effects[self.state.active_power]
                active_effect.update(self.state, dt)
                frame = active_effect.render(frame, self.state)
                
            # Render HUD
            self.hud.render(frame, self.state)
                
            # Display
            cv2.imshow("Reality Engine", frame)
            
            # Handle keyboard input
            self._handle_input()
            
    def _handle_input(self):
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            self.stop()
        elif key == ord('d'):
            self.state.debug_mode = not self.state.debug_mode
        elif key == ord('b'):
            # Save current frame as background
            self.state.background_frame = self.state.frame.copy()
            print("Background captured!")
        elif ord('1') <= key <= ord('5'):
            # Power selection
            power_idx = key - ord('1') + 1
            modes = list(PowerMode)
            if power_idx < len(modes):
                self.state.active_power = modes[power_idx]
                print(f"Switched to mode: {self.state.active_power.name}")
        elif key == ord('0'):
            self.state.active_power = PowerMode.NONE
            print("Powers disabled")
