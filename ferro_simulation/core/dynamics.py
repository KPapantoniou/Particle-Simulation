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

    def get_open_loop_currents(self,time):
    # Example: Rotate the magnetic field direction over time
        i1 = 1.0 * torch.cos(torch.tensor(time))
        i2 = 1.0 * torch.sin(torch.tensor(time))
        return [i1, i2]

    def close_loop_control(self, p, force_grid, grid_limit, target,k):
        
        F1 = force_grid[0]
        F2 = force_grid[1]
        F3 = force_grid[2]
        F4 = force_grid[3]
        # print(F1.shape,F1.shape)
        nx, ny, _ = F1.shape
        # for p in particles:

            # U = G*(-k*(particle.position-target))
        # print(p.position)
        idx_x = int(((p.position[0] + grid_limit) / (2 * grid_limit)) * (nx - 1))
        idx_y = int(((p.position[1] + grid_limit) / (2 * grid_limit)) * (ny - 1))

        idx_x = max(0, min(nx - 1, idx_x))
        idx_y = max(0, min(ny - 1, idx_y))
        G = torch.stack([
        F1[idx_x, idx_y],   
        F2[idx_x, idx_y],
        F3[idx_x, idx_y],
        F4[idx_x, idx_y]
        ], dim=1) 
        # print(G.shape)
        G_pinv = torch.linalg.pinv(G)
        e_x = -k*(p.position[0]-target[0])
        e_y = -k*(p.position[1]-target[1])
        e = torch.tensor([e_x, e_y], dtype=torch.float64)
        I = torch.matmul(G_pinv,self.gamma*e)
        I = torch.clamp(I,-1.5,1.5)
        return I


    def step(self, particles, force_grid, obj, dt, steps, grid_limit,I):
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
            self._euler_step(particles, force_grid, obj, dt, grid_limit, I)
        elif self.method == "rk4":
            self._rk4_step(particles, force_grid, obj, dt, steps)
        else:
            raise ValueError(f"Unknown integration method: {self.method}")
        self.time +=dt
    
    
    def _euler_step(self, particles, force_grid, obj, dt, grid_limit,I):
        
        # print(particles,forces_obj,field_obj,dt)
        # B = field_obj.evaluate(p.position,t=self.time)
        # print(force_grid.shape)
        F_total = torch.sum(force_grid * I.view(-1,1,1,1), dim=0)
        # print(F_total.shape)
        Fx = F_total[:,:,0]
        Fy = F_total[:,:,1]
        

        

        nx, ny = Fx.shape
        if grid_limit is None:
            grid_limit = 1.0
        
        # for _ in range(steps):
        for i,p in enumerate(particles):
            
            
            idx_x = int(((p.position[0] + grid_limit) / (2 * grid_limit)) * (nx - 1))
            idx_y = int(((p.position[1] + grid_limit) / (2 * grid_limit)) * (ny - 1))

            idx_x = max(0, min(nx - 1, idx_x))
            idx_y = max(0, min(ny - 1, idx_y))

            fx = Fx[idx_x,idx_y]
            fy = Fy[idx_x,idx_y]
            # print(f"Force at center: {fx:.2e}, {fy:.2e}")
            

            p.velocity[0] = fx/ obj.damping
            p.velocity[1] = fy/obj.damping

            p.update_position(dt)

        


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
