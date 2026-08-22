import cv2
import numpy as np
import os
import math
from typing import Optional, Tuple

from reality_engine.effects.base import BaseEffect
from reality_engine.core.state import AppState, GestureType
from reality_engine.config import Config
from reality_engine.rendering.particles import ParticleSystem

class PortalEffect(BaseEffect):
    def __init__(self):
        self.active = False
        self.center: Optional[Tuple[int, int]] = None
        self.radius = 150
        self.rotation = 0.0
        self.particles = ParticleSystem(max_particles=1500)
        
        # Load or generate alternate dimension image
        self.alt_dimension = None
        asset_path = "assets/portal_dimension.jpg"
        if os.path.exists(asset_path):
            self.alt_dimension = cv2.imread(asset_path)
            
        if self.alt_dimension is None:
            # Generate a galaxy-like dummy background
            self.alt_dimension = np.zeros((Config.CAMERA_HEIGHT, Config.CAMERA_WIDTH, 3), dtype=np.uint8)
            cv2.randu(self.alt_dimension, 0, 255)
            # Add some dark space blueish tint
            self.alt_dimension[:, :, 0] = self.alt_dimension[:, :, 0] // 2 + 100 # B
            self.alt_dimension[:, :, 1] = self.alt_dimension[:, :, 1] // 4 # G
            self.alt_dimension[:, :, 2] = self.alt_dimension[:, :, 2] // 4 # R
            
            # Smooth it out
            self.alt_dimension = cv2.GaussianBlur(self.alt_dimension, (21, 21), 0)

    def reset(self) -> None:
        self.active = False
        self.center = None

    def update(self, state: AppState, dt: float) -> None:
        self.rotation += 2.0 * dt # Rotate border
        self.particles.update(dt)
        
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
                    
            if self.center:
                # Emit new particles around the border
                num_emit = 20
                for _ in range(num_emit):
                    angle = np.random.uniform(0, 2 * math.pi)
                    r_noise = self.radius + np.random.uniform(-5, 5)
                    x = self.center[0] + math.cos(angle) * r_noise
                    y = self.center[1] + math.sin(angle) * r_noise
                    
                    # Velocity tangent to the circle
                    tangent = angle + math.pi / 2
                    vx = math.cos(tangent) * 50
                    vy = math.sin(tangent) * 50
                    
                    # Orange/Gold spark color
                    color = (0, 165 + np.random.randint(-40, 40), 255)
                    
                    # Using the particle system emit. ParticleSystem emit signature:
                    # emit(count, pos_x, pos_y, speed, color, lifetime_base)
                    # We will emit 1 particle at a time in this loop, or we could vectorize it.
                    self.particles.emit(1, x, y, speed=50, color=color, lifetime_base=0.5)
                    # Override velocity to make it tangential
                    last_idx = np.where(self.particles.active)[0][-1]
                    self.particles.velocities[last_idx] = [vx + np.random.uniform(-10, 10), vy + np.random.uniform(-10, 10)]

    def render(self, frame: np.ndarray, state: AppState) -> np.ndarray:
        if not self.active or self.center is None:
            return frame
            
        h, w = frame.shape[:2]
        
        # Create circular mask for the portal content
        mask = np.zeros((h, w), dtype=np.float32)
        cv2.circle(mask, self.center, self.radius, 1.0, -1)
        
        # Feather the edges
        mask = cv2.GaussianBlur(mask, (21, 21), 0)
        mask_3d = np.stack([mask, mask, mask], axis=2)
        
        # Ensure alt_dimension matches frame size
        alt_dim_resized = cv2.resize(self.alt_dimension, (w, h))
        
        # Composite alt dimension into frame
        frame = frame * (1.0 - mask_3d) + alt_dim_resized * mask_3d
        frame = frame.astype(np.uint8)
        
        # Render particles
        self.particles.render(frame)
                
        return frame
