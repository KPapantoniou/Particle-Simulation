"""
forces.py
---------

Compute forces and torques on ferromagnetic particles.

Supports:
- Torque from external magnetic field:  τ = m x B
- Dipole-dipole forces between particles: F_ij = 3 μ0/4πr^5 [...]
- Viscuous damping: F_damp = -y*v
"""

# import numpy as np 
import torch as th
from core.particle import Particle

MU0 = 4 * th.pi*1e-07 #vacum permeability approximation

class Forces:
    def __init__(self, damping = 1e-8, dipole_interactions=True, device='cuda'):
        """
        Initialize the Forces object.

        Parameters
        ----------
        damping : float
            Viscous damping coefficient
        dipole_interactions : bool
            If True, compute pairwise dipole-dipole forces
        """  
        self.damping = damping
        self.dipole_interactions = dipole_interactions 
        self.device=device

    # def torque(self, particle: Particle, B: th.ndarray):
    #     """
    #     Compute torque τ = m x B.

    #     Parameters
    #     ----------
    #     particle : Particle
    #     B : np.ndarray, shape (3,0)
    #         Magnetic field at particle position

    #     Returns
    #     -------
    #     np.ndarray, shape (3,0)
    #         Torque vector
    #     """
    #     return th.cross(particle.magnetic_moment, B)

    def magnetic_force(self, particle, field, eps=1e-6):
        """
        Compute magnetic force F = ∇(m · B)
        using central finite differences.
        """
        r = particle.position
        m = particle.magnetic_moment

        F = th.zeros(3, device=self.device)

        for d in range(3):
            dr = th.zeros(3, device=self.device)
            dr[d] = eps

            B_plus = field.evaluate(r + dr, 0.0)
            B_minus = field.evaluate(r - dr, 0.0)

            F[d] = (th.dot(m, B_plus) - th.dot(m, B_minus)) / (2 * eps)
            # print(B_plus,"\n",B_minus)

        return F

    def damping_force(self, particle: Particle):
        """
        Compute viscous damping force: F = -y v

        Parameters
        ----------
        particle: Particle

        Returns
        -------
        np.ndarray, shape (3,)
            Damping force
        """
        return -self.damping * particle.velocity
    
    def dipole_force(self, p1, p2):
            # r_vec = p2.position - p1.position
            r_vec = (p2.position - p1.position).to(device=self.device, dtype=th.float32)
            
            m1 = p1.magnetic_moment.to(device=self.device, dtype=th.float32)
            m2 = p2.magnetic_moment.to(device=self.device, dtype=th.float32)

            r_mag = th.norm(r_vec)
            r_mag = th.clamp(r_mag, min=1e-9)  # avoid division by zero
            r_hat = r_vec / r_mag

            # m1 = p1.magnetic_moment
            # m2 = p2.magnetic_moment

            # dipole-dipole force formula
            dot_mr1 = th.dot(m1, r_hat)
            dot_mr2 = th.dot(m2, r_hat)
            dot_mm = th.dot(m1, m2)

            prefactor = 3 * MU0 / (4 * th.pi * r_mag**4)
            F = prefactor * ((dot_mr1 * m2) + (dot_mr2 * m1) + (dot_mm * r_hat) - 5 * dot_mr1 * dot_mr2 * r_hat)

            # clip maximum force to avoid overflow
            F = th.clamp(F, min=-1e-6, max=1e-6)
            # print(self.device)
            return F

    def compute_forces(self, particles, field_obj, dt):
        N = len(particles)
        F_list = [th.zeros(3, device=self.device) for _ in range(N)]

        # --- external magnetic force ---
        for i, p in enumerate(particles):
            # B = field_obj.evaluate(p.position, dt)
            fm = self.magnetic_force(p, field_obj)
            # print(i,fm)
            F_list[i] +=fm


        # --- dipole-dipole (optional) ---
        if self.dipole_interactions:
            for i in range(N):
                for j in range(i+1, N):
                    F = self.dipole_force(particles[i], particles[j])
                    F_list[i] += F
                    F_list[j] -= F

        # --- damping ---
        for i, p in enumerate(particles):
            F_list[i] -= self.damping_force(p)

        return F_list

