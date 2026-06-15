
import torch 
from core.particle import Particle

class Dynamics:
    def __init__(self, gamma =1.0, method = "euler", device = 'cuda', damping=1.0):
    
        self.gamma = gamma
        self.method =method.lower()
        self.device =device
        self.damping = damping
        self.time = 0.0

    def get_open_loop_currents(self,time):
    # Example: Rotate the magnetic field direction over time
        time_t = torch.tensor(time, device=self.device)
        i1 = 1.0 * torch.cos(time_t)
        i2 = 1.0 * torch.sin(time_t)
        return [i1, i2]

    def close_loop_control(self, p, force_grid, grid_limit, target,k):    
        F1, F2, F3, F4 = force_grid

        nx, ny, _ = F1.shape
        # for p in particles:

            # U = G*(-k*(particle.position-target))
        # print(p.position)
        pos = p.position
        device = pos.device
        if F1.device != device:
            raise ValueError("force_grid must be on the same device as particle state")

        if pos.ndim == 3:
            pos_ctl = pos[:, 0, :]
        else:
            pos_ctl = pos
        idx_x = ((pos_ctl[:, 0] + grid_limit) / (2 * grid_limit) * (nx - 1)).long()
        idx_y = ((pos_ctl[:, 1] + grid_limit) / (2 * grid_limit) * (ny - 1)).long()
        idx_x = idx_x.clamp(0, nx - 1)
        idx_y = idx_y.clamp(0, ny - 1)

        f1 = F1[idx_x, idx_y]  # (B,2)
        f2 = F2[idx_x, idx_y]
        f3 = F3[idx_x, idx_y]
        f4 = F4[idx_x, idx_y]

        # Build control matrix per batch: (B,2,4)
        G = torch.stack([f1, f2, f3, f4], dim=2)

        # Batched pseudo-inverse: (B,4,2)
        G_pinv = torch.linalg.pinv(G)

        # target should be (B,3)
        e_x = -k * (pos_ctl[:, 0] - target[:, 0])
        e_y = -k * (pos_ctl[:, 1] - target[:, 1])
        e = torch.stack([e_x, e_y], dim=1)  # (B,2)

        e = e.to(dtype=G_pinv.dtype, device=G_pinv.device)
        gamma = torch.as_tensor(self.gamma, dtype=G_pinv.dtype, device=G_pinv.device)
        I = torch.bmm(G_pinv, (gamma * e).unsqueeze(-1)).squeeze(-1)
        I = torch.clamp(I, -2, 2)
        return I



    def step(self, particles, force_grid, obj, dt, steps, grid_limit,I):
     
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
        if isinstance(force_grid, (list, tuple)):
            force_grid = torch.stack(force_grid, dim=0)
        if force_grid.ndim == 3:
            force_grid = force_grid.unsqueeze(0)
        if force_grid.ndim != 4:
            raise ValueError(
                f"force_grid must have shape (C,Nx,Ny,2) or (Nx,Ny,2), got {tuple(force_grid.shape)}"
            )
        if force_grid.device != I.device:
            raise ValueError("force_grid and currents must be on the same device")

        nx, ny = force_grid.shape[1], force_grid.shape[2]
        if grid_limit is None:
            grid_limit = 1.0

        pos = particles.position
        particles.apply_jitter(intensity_factor=0.01)
        vel = particles.velocity
        if pos.ndim == 2:
            pos = pos.unsqueeze(1)
        if vel.ndim == 2:
            vel = vel.unsqueeze(1)

        idx_x = ((pos[..., 0] + grid_limit) / (2 * grid_limit) * (nx - 1)).long()
        idx_y = ((pos[..., 1] + grid_limit) / (2 * grid_limit) * (ny - 1)).long()
        idx_x = idx_x.clamp(0, nx - 1)
        idx_y = idx_y.clamp(0, ny - 1)

        sampled = force_grid[:, idx_x, idx_y].permute(1, 2, 0, 3)
        force = (I[:, None, :, None] * sampled).sum(dim=2)

        vel[..., 0] = force[..., 0] / self.damping
        vel[..., 1] = force[..., 1] / self.damping
        particles.update_position(dt)


            


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
    #        = torch.clamp(F1 + 2*F2 + 2*F3 + F4, min=-1e3, max=1e3)
    #         p.velocity += dt / 6  / p.mass

    #         v_total = v1 + 2*v2 + 2*v3 + v4
    #         p.position += dt / 6 * v_total

    #         torque_total = torch.linalg.cross(m1, B1) + 2*torch.linalg.cross(m2, B2) + 2*torch.linalg.cross(m3, B3) + torch.linalg.cross(m4, B4)
    #         p.magnetic_moment += dt / 6 * self.gamma * torch.clamp(torque_total, min=-1e3, max=1e3)
