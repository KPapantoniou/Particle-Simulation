import numpy as np
import torch as th
import time
from visualization.animate import Animate
from run_sim import run_simulation, SimulationRunner
from core.particle import Particle
from core.field import Field
from core.forces import Forces
from core.dynamics import Dynamics
from visualization.field_visual import (
    grid_visualizer,
    visualize_quiver,
    visualize_3d_surface,
    visualize_force_flow,
    visualize_force_magnitude,
    visualize_overlay,
    contrabillity_condition,
)
from visualization.plot_particles import (
    plot_particle_paths,
    plot_particle_velocity,
    plot_coil_currents,
    plot_position_error,
)
import sys
import os


device = 'cuda' if th.cuda.is_available() else 'cpu'
batch_size = int(os.environ.get("BATCH_SIZE", "1"))
batch_seed_env = os.environ.get("BATCH_SEED")
k_gain_jitter = float(os.environ.get("K_GAIN_JITTER", "0.0"))
damping_jitter = float(os.environ.get("DAMPING_JITTER", "0.0"))
start_margin = float(os.environ.get("START_MARGIN", "0.9"))
target_margin = float(os.environ.get("TARGET_MARGIN", "0.9"))
output_dir = os.environ.get("OUTPUT_DIR", "results")
record_potential = os.environ.get("SAVE_POTENTIAL", "0") == "1"
potential_stride = int(os.environ.get("POTENTIAL_STRIDE", "1"))
history_device = os.environ.get("HISTORY_DEVICE", "cpu")
history_stride = int(os.environ.get("HISTORY_STRIDE", "1"))
save_pot_csv = os.environ.get("SAVE_POT_CSV", "0") == "1"
pot_csv_dir = os.environ.get("POT_CSV_DIR", "pot_csv")
show_animation = os.environ.get("SHOW_ANIMATION") == "1"
show_plot = os.environ.get("SHOW_PLOT") == "1"

modes = ["uniform","time_varying", "grid","coil"]
mode = ""
if len(sys.argv) > 1 and sys.argv[1] in modes:
    mode = sys.argv[1]

rho_iron = 7800  # kg/m^3
r = 3e-6
mass = rho_iron * (4/3) * th.pi * r**3

Ms = 1.7e6
volume = (4/3) * th.pi * r**3
m = Ms * volume   # ≈ 1e-10 A·m²
VISCOSITY = 0.001
damping_coeff = float(6*th.pi *VISCOSITY*5e-6)
relaxation_time = mass/damping_coeff

Nx,Ny = 257,257
physical_width = 3e-3
dx = physical_width / Nx
grid_limit = physical_width / 2
N_coils =100

x= -(grid_limit-5e-4)
y= -(grid_limit -5e-4)

x2 = x
y2 = -y

x3 = -x
y3 = -y

x4 = -x
y4 = y

k=1.75

def _sample_xy(grid_limit, margin, batch_size, device):
    span = grid_limit * margin
    xy = (th.rand((batch_size, 2), device=device) * 2 - 1) * span
    z = th.zeros((batch_size, 1), device=device)
    return th.cat([xy, z], dim=1)


def _load_pot_csv(path, nx, ny):
    data = np.loadtxt(path, delimiter=",", comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    frames = [th.tensor(row.reshape(nx, ny), device=device, dtype=th.float32) for row in data]
    return frames

def circular_coil(radius=1e-3, n_points=2, z=0.0, device=None):
    if device is None:
        device = th.device("cpu")
    theta = th.linspace(0, 2*th.pi, n_points, device=device)
    x = radius * th.cos(theta)
    y = radius * th.sin(theta)
    z = th.full_like(x, z)

    return th.stack([x, y, z], axis=1)

mag_moment_base = th.tensor([0, 0, m], device=device, dtype=th.float32).unsqueeze(0)

B_field = Field( Nx,
                Ny,
                radius=1e-3,
                x=x,
                y=y,
                z_distance=5e-5,
                magnetic_moment=m,
                N_coils=N_coils,
                mode=mode,
                B0=[0.0,0,0.001],
                omega = 20.0,
                coil_points=circular_coil(radius=0.001, n_points=10, device=device),
                I=1.0,
                device=device,
                physical_width=physical_width,
                dx=dx,
                grid_limit=grid_limit,
                )

B_field2 =Field(Nx,
                Ny,
                radius=1e-3,
                x=x2,
                y=y2,
                z_distance=5e-5,
                magnetic_moment=m,
                N_coils=N_coils,
                mode=mode,
                B0=[0.0,0,0.001],
                omega = 20.0,
                coil_points=circular_coil(radius=0.001, n_points=10, device=device),
                I=1.0,
                device=device,
                physical_width=physical_width,
                dx=dx,
                grid_limit=grid_limit,
                ) 

B_field3 =Field(Nx,
                Ny,
                radius=1e-3,
                x=x3,
                y=y3,
                z_distance=5e-5,
                magnetic_moment=m,
                N_coils=N_coils,
                mode=mode,
                B0=[0.0,0,0.001],
                omega = 20.0,
                coil_points=circular_coil(radius=0.001, n_points=10, device=device),
                I=1.0,
                device=device,
                physical_width=physical_width,
                dx=dx,
                grid_limit=grid_limit,
                ) 

B_field4 =Field(Nx,
                Ny,
                radius=1e-3,
                x=x4,
                y=y4,
                z_distance=5e-5,
                magnetic_moment=m,
                N_coils=N_coils,
                mode=mode,
                B0=[0.0,0,0.001],
                omega = 20.0,
                coil_points=circular_coil(radius=0.001, n_points=10, device=device),
                I=1.0,
                device=device,
                physical_width=physical_width,
                dx=dx,
                grid_limit=grid_limit,
                ) 

grid1 = B_field.generate_grid()
B_grid = grid1[0]
U_grid = grid1[1]

grid2 = B_field2.generate_grid()
B_grid_2 =grid2[0]
U_grid_2 = grid2[1]

grid3 = B_field3.generate_grid()
B_grid_3 = grid3[0]
U_grid_3 = grid3[1]

grid4 = B_field4.generate_grid()
B_grid_4 = grid4[0]
U_grid_4  = grid4[1]

B_tot = B_grid + B_grid_2 + B_grid_3 + B_grid_4
U_tot = U_grid + U_grid_2 + U_grid_3 + U_grid_4

forces_base = Forces(damping=damping_coeff, dipole_interactions=False, device=device)
# F_grid=forces_base.magnetic_force(None, U_tot, dx)
#Checking for contrabillity condition
F1 = forces_base.magnetic_force(None, U_grid, dx)
F2 = forces_base.magnetic_force(None, U_grid_2, dx)
F3 = forces_base.magnetic_force(None, U_grid_3, dx)
F4 = forces_base.magnetic_force(None, U_grid_4, dx)
# Normalize forces so they have length 1
# F1_unit = F1 / (th.norm(F1, dim=-1, keepdim=True) + 1e-30)
# F2_unit = F2 / (th.norm(F2, dim=-1, keepdim=True) + 1e-30)
# F3_unit = F3 / (th.norm(F3, dim=-1, keepdim=True) + 1e-30)
# F4_unit = F4 / (th.norm(F3, dim=-1, keepdim=True) + 1e-30)

# det_control = (F1_unit[:,:,0] * F2_unit[:,:,1]) - (F1_unit[:,:,1] * F2_unit[:,:,0])
# det_control = (F1[:,:,0] * F2[:,:,1]) - (F1[:,:,1] * F2[:,:,0])


dt = 1e-3
t_max = 50
steps = int(t_max/dt)
# 
F = [F1,F2,F3,F4]
F_basis = th.stack(F, dim=0).to(device=device)
U_basis = th.stack([U_grid, U_grid_2, U_grid_3, U_grid_4], dim=0).to(device=device)

if batch_seed_env is not None:
    th.manual_seed(int(batch_seed_env))

damping_scale = 1.0 + float((th.rand(1) * 2 - 1) * damping_jitter)
k_scale = 1.0 + float((th.rand(1) * 2 - 1) * k_gain_jitter)
damping_run = damping_coeff * damping_scale
k_run = k * k_scale

def _build_particle(start_pos_tensor):
    batch = start_pos_tensor.shape[0]
    return Particle(
        position=start_pos_tensor,
        velocity=th.zeros((batch, 3), device=device, dtype=th.float32),
        magnetic_moment=mag_moment_base.repeat(batch, 1),
        mass=mass,
        radius=5e-6,
        device=device,
    )

target = _sample_xy(grid_limit, target_margin, batch_size, device)
start_pos = _sample_xy(grid_limit, start_margin, batch_size, device)

particles = _build_particle(start_pos)

forces = Forces(damping=damping_run, dipole_interactions=False, device=device)
dynamics = Dynamics(gamma=damping_run, method="euler", damping=damping_run, device=device)

print(f"  Time step: {dt*1e6:.0f} μs")
print(f"  Duration: {t_max*1e3:.0f} ms")
print(f"  Total steps: {steps:,}")
print(f"  Mode: {mode or 'uniform'}")
print(f"  Device: {device}")
print(f"  Batch size: {batch_size}")
print(f"  Grid: {Nx}x{Ny}")
print(f"  k gain: {k_run:.6f}")
print(f"  Damping: {damping_run:.6e}")
print(f"  Start margin: {start_margin:.3f}")
print(f"  Target margin: {target_margin:.3f}")
if batch_seed_env is not None:
    print(f"  Batch seed: {batch_seed_env}")
print()

def run_mode(mode, particles, target):
    batch_size_run = particles.position.shape[0]
    pot_csv_path = None
    if save_pot_csv:
        os.makedirs(pot_csv_dir, exist_ok=True)
        pot_csv_path = os.path.join(pot_csv_dir, f"pot_{mode}.csv")
    return run_simulation(
        mode,
        particles,
        F_basis,
        U_basis,
        target,
        grid_limit,
        dt,
        k_run,
        dynamics,
        forces,
        steps,
        batch_size=batch_size_run,
        record_potential=record_potential or save_pot_csv,
        potential_stride=potential_stride,
        history_device=history_device,
        history_stride=history_stride,
        pot_csv_path=pot_csv_path,
        record_positions=True,
    )

validate_batch = os.environ.get("VALIDATE_BATCH", "1") == "1"

def _select_batch_positions(pos_hist, batch_index=0):
    positions = th.stack(pos_hist, dim=0)
    if positions.ndim == 3:
        return positions[:, batch_index, :].unsqueeze(1)
    return positions[:, batch_index, :, :]

def _select_batch_currents(curr_hist, batch_index=0):
    if not curr_hist:
        return th.empty((0,), device=device)
    currents = th.stack(curr_hist, dim=0)
    return currents[:, batch_index, :]

def _validate_batch_result(mode_label, batched_result, start_ref, target_ref):
    single_particle = _build_particle(start_ref[:1].clone())
    single_target = target_ref[:1].clone()
    dynamics_single = Dynamics(gamma=damping_run, method="euler", damping=damping_run, device=device)
    forces_single = Forces(damping=damping_run, dipole_interactions=False, device=device)
    single_result = run_simulation(
        mode_label,
        single_particle,
        F_basis,
        U_basis,
        single_target,
        grid_limit,
        dt,
        k_run,
        dynamics_single,
        forces_single,
        steps,
        batch_size=1,
        record_positions=True,
        history_device=history_device,
        history_stride=history_stride,
    )
    batched_positions = _select_batch_positions(batched_result["pos"], 0)
    single_positions = _select_batch_positions(single_result["pos"], 0)
    min_len = min(batched_positions.shape[0], single_positions.shape[0])
    if not th.allclose(
        batched_positions[:min_len],
        single_positions[:min_len],
        rtol=1e-5,
        atol=1e-6,
    ):
        raise RuntimeError(f"Batch validation failed for positions ({mode_label}).")
    if batched_result["curr"]:
        batched_currents = _select_batch_currents(batched_result["curr"], 0)
        single_currents = _select_batch_currents(single_result["curr"], 0)
        min_len = min(batched_currents.shape[0], single_currents.shape[0])
        if not th.allclose(
            batched_currents[:min_len],
            single_currents[:min_len],
            rtol=1e-5,
            atol=1e-6,
        ):
            raise RuntimeError(f"Batch validation failed for currents ({mode_label}).")

# # Magnetic plots
# grid_visualizer(B_tot, grid_limit=grid_limit)
# visualize_quiver(B_tot, grid_limit=grid_limit)
if show_plot or show_animation:
    visualize_3d_surface(B_tot, "3D Field Intensity Surface", grid_limit=grid_limit)
    visualize_3d_surface(-B_tot, "3D U Intensity Surface", grid_limit=grid_limit)

results = []
target_open = None
target_closed = None
start_open = None
start_closed = None
if batch_size == 1:
    target_closed = target
    start_closed = start_pos
    results = [run_mode("closed", particles, target)]
else:
    target_open = target[:1]
    start_open = start_pos[:1]
    target_closed = target[1:]
    start_closed = start_pos[1:]

    particles_open = _build_particle(start_open)
    particles_closed = _build_particle(start_closed)

    results = [
        run_mode("open", particles_open, target_open),
        run_mode("closed", particles_closed, target_closed),
    ]

for result in results:
    mode_label = result["mode"]
    if validate_batch and batch_size > 1 and mode_label == "closed":
        _validate_batch_result(mode_label, result, start_closed, target_closed)

    if batch_size > 1:
        os.makedirs(output_dir, exist_ok=True)
        batched_positions = th.stack(result["pos"], dim=0)
        batched_currents = (
            th.stack(result["curr"], dim=0)
            if result["curr"]
            else th.empty((0,), device=device)
        )
        th.save(
            {"pos": batched_positions, "curr": batched_currents},
            os.path.join(output_dir, f"trajectories_{mode_label}.pt"),
        )

    positions_over_time = th.stack(result["pos"], dim=0)
    current = th.stack(result["curr"]) if result["curr"] else th.empty((0,), device=device)

    pot_history = []
    if show_animation:
        if save_pot_csv:
            pot_history = _load_pot_csv(os.path.join(pot_csv_dir, f"pot_{mode_label}.csv"), Nx, Ny)
        else:
            pot_history = list(result["pot"])

    batch_targets = target_open if mode_label == "open" else target_closed
    batch_count = batch_targets.shape[0]

    for batch_idx in range(batch_count):
        if positions_over_time.ndim == 3:
            positions_i = positions_over_time[:, batch_idx, :].unsqueeze(1)
        else:
            positions_i = positions_over_time[:, batch_idx, :, :]

        current_i = th.empty((0,), device=device)
        if current.numel() > 0:
            current_i = current[:, batch_idx, :]

        pot_history_i = []
        if show_animation and pot_history:
            if save_pot_csv:
                pot_history_i = pot_history if batch_idx == 0 else []
            else:
                pot_history_i = [frame[batch_idx] for frame in pot_history]

        if pot_history_i:
            animate = Animate(
                pot_history_i,
                grid_limit,
                positions_i,
                title=f"3D Potential Field Surface ({mode_label})",
                zlabel="Potential [J]",
            )
            if show_animation:
                animate.show()

        global_batch_idx = 1 if mode_label == "open" else batch_idx + 2
        if show_plot or show_animation:
            plot_particle_paths(
                positions_i,
                labels=["p1"],
                grid_limit=grid_limit,
                target=batch_targets[batch_idx],
                title=f"Particle Trajectory ({mode_label})",
            )
            plot_position_error(
                positions_i,
                target=batch_targets[batch_idx],
                dt=dt,
                labels=["p1"],
            )
            if current_i.numel() > 0:
                plot_coil_currents(current_i, dt)
        else:
            os.makedirs(output_dir, exist_ok=True)
            plot_particle_paths(
                positions_i,
                labels=["p1"],
                grid_limit=grid_limit,
                target=batch_targets[batch_idx],
                title=f"Particle Trajectory ({mode_label})",
                save_path=os.path.join(
                    output_dir, f"batch_{global_batch_idx}_trajectory_{mode_label}.png"
                ),
            )
            plot_position_error(
                positions_i,
                target=batch_targets[batch_idx],
                dt=dt,
                labels=["p1"],
                save_path=os.path.join(
                    output_dir, f"batch_{global_batch_idx}_error_{mode_label}.png"
                ),
            )
            if current_i.numel() > 0:
                plot_coil_currents(
                    current_i,
                    dt,
                    save_path=os.path.join(
                        output_dir, f"batch_{global_batch_idx}_currents_{mode_label}.png"
                    ),
                )
