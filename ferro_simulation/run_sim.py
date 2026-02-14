import copy
import torch as th


def run_simulation(
    mode,
    particles,
    F_basis,
    U_basis,
    target,
    grid_limit,
    dt,
    k,
    dynamics_obj,
    forces_obj,
    steps,
    batch_size=1,
    record_potential=True,
    potential_stride=1,
    history_device="cpu",
    history_stride=1,
    pot_csv_path=None,
    record_positions=True,
):
    if isinstance(particles, (list, tuple)):
        if len(particles) != 1:
            raise ValueError("Pass a single Particle with batched dimensions.")
        particles = particles[0]
    if target.ndim == 1:
        target = target.unsqueeze(0).repeat(batch_size, 1)
    local_particles = copy.deepcopy(particles)
    device = local_particles.position.device
    F_basis = F_basis.to(device=device)
    U_basis = U_basis.to(device=device)
    target = target.to(device=device)
    pos_hist = [] if record_positions else None
    pot_hist, current_hist = [], []
    hist_device = th.device(history_device) if not isinstance(history_device, th.device) else history_device
    I_fixed = None
    target_cp = copy.deepcopy(target)
    stride = max(1, int(history_stride))
    pot_stride = max(1, int(potential_stride))
    csv_writer = None
    csv_file = None

    if local_particles.position.ndim == 1:
        local_particles.position = local_particles.position.unsqueeze(0).repeat(batch_size, 1)
    if local_particles.velocity.ndim == 1:
        local_particles.velocity = local_particles.velocity.unsqueeze(0).repeat(batch_size, 1)
    if local_particles.magnetic_moment.ndim == 1:
        local_particles.magnetic_moment = local_particles.magnetic_moment.unsqueeze(0).repeat(batch_size, 1)

    if mode == "open":
        I_fixed = dynamics_obj.close_loop_control(
            local_particles, F_basis, grid_limit, target_cp, k
        )

    if pot_csv_path:
        import csv
        csv_file = open(pot_csv_path, "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["#nx", U_basis.shape[1], "ny", U_basis.shape[2]])

    for i in range(int(steps)):
        if mode == "closed":
            I = dynamics_obj.close_loop_control(
                local_particles, F_basis, grid_limit, target_cp, k
            )
        else:
            I = I_fixed

        dynamics_obj.step(
            local_particles,
            F_basis,
            forces_obj,
            dt,
            i * dt,
            grid_limit,
            I,
        )

        if record_potential and (i % pot_stride == 0):
            U_total = th.einsum("bc,cxy->bxy", I, U_basis)
            if csv_writer is not None:
                csv_writer.writerow(U_total[0].detach().cpu().flatten().tolist())
            else:
                pot_hist.append(U_total.detach().to(hist_device))
        if i % stride == 0:
            current_hist.append(I.detach().to(hist_device))
            if record_positions:
                pos_hist.append(local_particles.position.detach().to(hist_device))

        pos_ctl = local_particles.position
        if pos_ctl.ndim == 3:
            pos_ctl = pos_ctl[:, 0, :]
        dist = th.norm(pos_ctl - target, dim=1)
        if (dist < 1e-6).all():
            print(f"Final distance: {dist} at time:{i*dt}")
            break

    if csv_file is not None:
        csv_file.close()

    return {"pos": pos_hist, "pot": pot_hist, "curr": current_hist, "mode": mode}


class SimulationRunner:
    def __init__(self, run_fn):
        self.run_fn = run_fn

    def run(self, mode):
        return self.run_fn(mode)

    def run_both(self):
        return [self.run_fn("open"), self.run_fn("closed")]
