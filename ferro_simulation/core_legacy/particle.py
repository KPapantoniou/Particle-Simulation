"""
particle.py
------------

Defines the Particle class representing a single ferromagnetic particle.

Attributes:
-----------
- position : th.ndarray
    3D position vector r = [x, y, z]
- velocity : np.ndarray
    3D velocity vector v = dr/dt
- acceleration : np.ndarray
    3D acceleration vector a = dv/dt
- magnetic_moment : np.ndarray
    Magnetic moment vector m
- mass : float
    Particle mass
- radius : float
    Particle radius (optional for collisions or visualization)
"""

# import numpy as np
import torch


class Particle:
    def __init__(self, position, velocity, magnetic_moment, mass=1.0, radius=1e-3, device='cuda'):
        
        self.device = device
      
        def _to_batched(t):
            if torch.is_tensor(t):
                t = t.to(device=self.device, dtype=torch.float32)
                if t.ndim ==1:
                    return t.unsqueeze(0)
                return t
            t = torch.tensor(t,device=self.device, dtype=torch.float32)
            if t.ndim==1:
                return t.unsqueeze(0)
            return t
        self.position = _to_batched(position)
       
        self.velocity = _to_batched(velocity)
        self.magnetic_moment = _to_batched(magnetic_moment)
        self.acceleration = torch.zeros_like(self.position, device=self.device, dtype=torch.float32)
        self.mass = mass
        self.radius = radius
        self.current_jitter = torch.zeros_like(self.position)

    def __repr__(self):
        # .tolist() makes it readable and removes the 'device=cuda' text
        pos = self.position.detach().cpu().tolist()
        vel = self.velocity.detach().cpu().tolist()
        return f"Particle(pos={pos}, vel={vel}, mass={self.mass:.2e})"
    
    def update_position(self, dt):
        """Update current position using current velocity and timestep dt"""
        self.position += (self.velocity * dt) 

    def update_velocity(self, dt):
        """Update particle velocity using current acceleration and timestep dt."""
        self.velocity += self.acceleration * dt

    def apply_force(self, force):
        """Compute acceleration from applied force: F = m * a"""
        self.acceleration = force / self.mass

    def apply_torque(self, torque):
        return
    
    def apply_jitter(self, intensity_factor=0.01, grid_limit=3e-3):
            diameter = 2 * self.radius
            limit = diameter * intensity_factor
            self.current_jitter = (torch.rand_like(self.position) * 2 - 1) * limit
            self.position += self.current_jitter
            self.position.clamp_(-grid_limit, grid_limit)

    def __repr__(self):
        return f"Particle(batch_size={self.position.shape[0]}, device={self.device})"