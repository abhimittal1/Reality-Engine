import cv2
import numpy as np
import math
from typing import Optional, Tuple

from reality_engine.effects.base import BaseEffect
from reality_engine.core.state import AppState, GestureType
from reality_engine.rendering.particles import ParticleSystem

class BlackHoleEffect(BaseEffect):
    def __init__(self):
        self.active = False
        self.center: Optional[Tuple[int, int]] = None
        self.radius = 200 # ROI radius
        self.strength = 80 # Event horizon radius
        self.particles = ParticleSystem(max_particles=1000)
        
        # Precompute maps for remapping to speed up the process
        self.map_x = None
        self.map_y = None
        self._init_maps()

    def _init_maps(self):
        size = self.radius * 2
        # Coordinate grid relative to center
        Y, X = np.indices((size, size))
        Y = Y - self.radius
        X = X - self.radius
        
        R = np.sqrt(X**2 + Y**2)
        R[R == 0] = 1 # Prevent division by zero
        
        # Lensing equation: r_new = r * (1 - (strength/r)^2) for r > strength
        # Since we pull pixels IN, we map (x,y) to a source pixel farther out.
        # Simple gravity warp: source_r = r + strength^2 / r
        source_R = R + (self.strength**2) / R
        
        # Map back to cartesian
        map_x = (X * source_R / R) + self.radius
        map_y = (Y * source_R / R) + self.radius
        
        # Event horizon (black circle)
        map_x[R < self.strength] = -1
        map_y[R < self.strength] = -1
        
        self.map_x = map_x.astype(np.float32)
        self.map_y = map_y.astype(np.float32)

    def reset(self) -> None:
        self.active = False
        self.center = None

    def update(self, state: AppState, dt: float) -> None:
        self.particles.update(dt)
        
        for event in state.recent_gestures:
            if event.type == GestureType.PINCH:
                self.active = True
                self.center = event.position
            elif event.type == GestureType.FIST:
                self.active = False
                
        # Dragging when active
        if self.active:
            for hand in state.hands:
                if hand.is_pinching:
                    self.center = hand.palm_center
                    break
                    
            if self.center:
                # Emit vortex particles spiraling inward
                num_emit = 5
                for _ in range(num_emit):
                    angle = np.random.uniform(0, 2 * math.pi)
                    # Start farther out
                    r = np.random.uniform(self.strength, self.radius)
                    x = self.center[0] + math.cos(angle) * r
                    y = self.center[1] + math.sin(angle) * r
                    
                    # Velocity inward and tangential
                    tangent = angle + math.pi / 2
                    vx = math.cos(tangent) * 40 - math.cos(angle) * 30
                    vy = math.sin(tangent) * 40 - math.sin(angle) * 30
                    
                    color = (255, 200, 200) # slight bluish white
                    self.particles.emit(1, x, y, speed=50, color=color, lifetime_base=0.5)
                    last_idx = np.where(self.particles.active)[0][-1]
                    self.particles.velocities[last_idx] = [vx, vy]

    def render(self, frame: np.ndarray, state: AppState) -> np.ndarray:
        if not self.active or self.center is None:
            return frame
            
        h, w = frame.shape[:2]
        cx, cy = self.center
        
        # Calculate ROI bounds
        x1 = max(0, cx - self.radius)
        y1 = max(0, cy - self.radius)
        x2 = min(w, cx + self.radius)
        y2 = min(h, cy + self.radius)
        
        if x1 >= x2 or y1 >= y2:
            return frame
            
        roi = frame[y1:y2, x1:x2]
        roi_h, roi_w = roi.shape[:2]
        
        # Shift maps to match clipped ROI (if near edges)
        start_x = self.radius - (cx - x1)
        start_y = self.radius - (cy - y1)
        
        mx = self.map_x[start_y:start_y+roi_h, start_x:start_x+roi_w]
        my = self.map_y[start_y:start_y+roi_h, start_x:start_x+roi_w]
        
        distorted = cv2.remap(roi, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        
        frame[y1:y2, x1:x2] = distorted
        
        # Render particles
        self.particles.render(frame)
        
        return frame
