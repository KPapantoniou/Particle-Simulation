"""
dynamics.py
-----------

Integrate particle motion using forces and torques.

Supports:
- Translational dynamics: F = m * a
- Rotational dynamics: dm/dt = γ (m × B)
- Integration methods: Euler, RK4 (can extend)
"""

# import numpy as np
import torch 
from core.particle import Particle

class Dynamics:
    def __init__(self, gamma =1.0, method = "euler", device = 'cuda'):
        """
        Initialize dynamics integrator.

        Parameters
        ----------
        gamma : float
            Gyromagnetic ratio for rotational dynamics
        """
        self.gamma = gamma
        self.method =method.lower()
        self.device =device
        self.time = 0.0

    def step(self,particles, forces_obj, field_obj, dt):
        """
        Advance all particles one timestep.

        Parameters
        ----------
        particles : list of Particle
        forces_obj : Forces
        field_obj : Field
        dt : float
        """
        if self.method == "euler":
            self._euler_step(particles, forces_obj, field_obj, dt)
        elif self.method == "rk4":
            self._rk4_step(particles, forces_obj, field_obj, dt)
        else:
            raise ValueError(f"Unknown integration method: {self.method}")
        self.time +=dt
    
    
    def _euler_step(self, particles, forces_obj, field_obj, dt):
        F_list = forces_obj.compute_forces(particles, field_obj, dt)
        # print(particles,forces_obj,field_obj,dt)
        # B = field_obj.evaluate(p.position,t=self.time)
        for i, p in enumerate(particles):
            
            F = F_list[i]

            # p.apply_force(F)
            p.velocity = F/forces_obj.damping
            # p.update_velocity(dt)
            p.update_position(dt)
            # print(p)
           
            # print(F.detach().cpu().tolist())
            # p.magnetic_moment += self.gamma * torch.cross(p.magnetic_moment, B) * dt

    # def _rk4_step(self, particles, forces_obj, field_obj, dt):
    #     """
    #     Performs one RK4 step for all particles.
    #     particles: list of Particle objects (positions, velocities, moments on GPU)
    #     forces_obj: Forces instance (compute forces with new PyTorch version)
    #     field_obj: Field instance (compute B field)
    #     dt: timestep
    #     """
    #     for p in particles:
    #         # -----------------
    #         # k1
    #         # -----------------
    #         B1 = field_obj.evaluate(p.position, 0)            # B at current pos
    #         F1_list = forces_obj.compute_forces(particles, field_obj)     # returns list of forces
    #         F1 = F1_list[particles.index(p)]                  # pick particle's force

    #         p1 = p.position
    #         v1 = p.velocity 
    #         m1 = p.magnetic_moment

    #         # -----------------
    #         # k2
    #         # -----------------
    #         v2 = v1 + F1 / p.mass * dt / 2
    #         r2 = p1 + v1 * dt / 2
    #         m2 = m1 + self.gamma * torch.linalg.cross(m1, B1) * dt / 2

    #         B2 = field_obj.evaluate(r2, dt/2)
    #         F2_list = forces_obj.compute_forces(particles,field_obj)
    #         F2 = F2_list[particles.index(p)]

    #         # -----------------
    #         # k3
    #         # -----------------
    #         v3 = v1 + F2 / p.mass * dt / 2
    #         r3 = p.position + v2 * dt / 2
    #         m3 = m1 + self.gamma * torch.linalg.cross(m2, B2) * dt / 2

    #         B3 = field_obj.evaluate(r3, dt/2)
    #         F3_list = forces_obj.compute_forces(particles,field_obj)
    #         F3 = F3_list[particles.index(p)]

    #         # -----------------
    #         # k4
    #         # -----------------
    #         v4 = v1 + F3 / p.mass * dt
    #         r4 = p.position + v3 * dt
    #         m4 = m1 + self.gamma * torch.linalg.cross(m3, B3) * dt

    #         B4 = field_obj.evaluate(r4, dt)
    #         F4_list = forces_obj.compute_forces(particles,field_obj)
    #         F4 = F4_list[particles.index(p)]

    #         # -----------------
    #         # Combine RK4 increments
    #         # -----------------
    #         # Clip forces to prevent overflows
    #         F_total = torch.clamp(F1 + 2*F2 + 2*F3 + F4, min=-1e3, max=1e3)
    #         p.velocity += dt / 6 * F_total / p.mass

    #         v_total = v1 + 2*v2 + 2*v3 + v4
    #         p.position += dt / 6 * v_total

    #         torque_total = torch.linalg.cross(m1, B1) + 2*torch.linalg.cross(m2, B2) + 2*torch.linalg.cross(m3, B3) + torch.linalg.cross(m4, B4)
    #         p.magnetic_moment += dt / 6 * self.gamma * torch.clamp(torque_total, min=-1e3, max=1e3)
