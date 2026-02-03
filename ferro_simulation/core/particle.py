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
        """
        Initialize a ferromagnetic particle.

        Parameters
        ----------
        position : array-like, shape (3,)
            Initial particle position [x, y, z]
        velocity : array-like, shape (3,)
            Initial particle velocity [vx, vy, vz]
        magnetic_moment : array-like, shape (3,)
            Particle magnetic moment vector [mx, my, mz]
        mass : float
            Particle mass
        radius : float
            Particle radius
        """
        # self.gamma = gamma
        self.device = device
        self.position = torch.tensor(position, device=self.device, dtype=torch.float32)
        self.velocity = torch.tensor(velocity, device=self.device, dtype=torch.float32)
        self.acceleration = torch.zeros(3, device=self.device, dtype=torch.float32)
        self.magnetic_moment = torch.tensor(magnetic_moment, device=self.device, dtype=torch.float32)
        self.mass = mass
        self.radius = radius

    def __repr__(self):
        # .tolist() makes it readable and removes the 'device=cuda' text
        pos = self.position.detach().cpu().tolist()
        vel = self.velocity.detach().cpu().tolist()
        return f"Particle(pos={pos}, vel={vel}, mass={self.mass:.2e})"
    
    def update_position(self, dt):
        """Update current position using current velocity and timestep dt"""
        self.position += self.velocity * dt

    def update_velocity(self, dt):
        """Update particle velocity using current acceleration and timestep dt."""
        self.velocity += self.acceleration * dt

    def apply_force(self, force):
        """Compute acceleration from applied force: F = m * a"""
        self.acceleration = force / self.mass

    def apply_torque(self, torque):
        return
