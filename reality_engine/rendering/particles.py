import cv2
import numpy as np

class ParticleSystem:
    def __init__(self, max_particles=1000):
        self.max_particles = max_particles
        self.positions = np.zeros((max_particles, 2), dtype=np.float32)
        self.velocities = np.zeros((max_particles, 2), dtype=np.float32)
        self.colors = np.zeros((max_particles, 3), dtype=np.uint8)
        self.lifetimes = np.zeros(max_particles, dtype=np.float32)
        self.active = np.zeros(max_particles, dtype=bool)
        
    def emit(self, count, pos_x, pos_y, speed, color, lifetime_base=1.0):
        inactive_indices = np.where(~self.active)[0]
        count = min(count, len(inactive_indices))
        if count == 0:
            return
            
        indices = inactive_indices[:count]
        
        # Random directions
        angles = np.random.uniform(0, 2*np.pi, count)
        speeds = np.random.uniform(speed*0.5, speed*1.5, count)
        
        self.positions[indices, 0] = pos_x + np.random.uniform(-5, 5, count)
        self.positions[indices, 1] = pos_y + np.random.uniform(-5, 5, count)
        
        self.velocities[indices, 0] = np.cos(angles) * speeds
        self.velocities[indices, 1] = np.sin(angles) * speeds
        
        # Broadcasting color assignment
        for i, idx in enumerate(indices):
            self.colors[idx] = color
            
        self.lifetimes[indices] = np.random.uniform(lifetime_base*0.5, lifetime_base*1.5, count)
        self.active[indices] = True
        
    def update(self, dt):
        active_idx = np.where(self.active)[0]
        if len(active_idx) == 0:
            return
            
        self.positions[active_idx] += self.velocities[active_idx] * dt
        self.lifetimes[active_idx] -= dt
        
        # Expire dead particles
        dead_idx = active_idx[self.lifetimes[active_idx] <= 0]
        self.active[dead_idx] = False
        
    def render(self, frame):
        active_idx = np.where(self.active)[0]
        for idx in active_idx:
            x, y = int(self.positions[idx, 0]), int(self.positions[idx, 1])
            c = tuple(int(x) for x in self.colors[idx])
            # Draw as tiny circles or points depending on lifetime
            radius = max(1, int(self.lifetimes[idx] * 2))
            cv2.circle(frame, (x, y), radius, c, -1)
