from __future__ import annotations

from dataclasses import replace
from typing import Callable

import torch as th

from physics.forces import compute_force, compute_potential
from .state import SimulationState


ControllerFn = Callable[[SimulationState], th.Tensor]
StepFn = Callable[
    [SimulationState, th.Tensor, th.Tensor, float, float, float, float, float, float],
    tuple[SimulationState, dict[str, th.Tensor]],
]


def _current_update_exact(curr: th.Tensor, voltage: th.Tensor, dt: float, resistance: float, inductance: float) -> th.Tensor:
    alpha = th.exp(
        th.tensor(
            -resistance * dt / inductance,
            dtype=curr.dtype,
            device=curr.device,
        )
    )
    i_inf = voltage / resistance
    return i_inf + (curr - i_inf) * alpha


def _advance_linear_drag(
    pos: th.Tensor,
    vel: th.Tensor,
    force: th.Tensor,
    dt: float,
    damping: float,
    mass: float,
) -> tuple[th.Tensor, th.Tensor]:
    if damping <= 0.0:
        acc = force / mass
        vel_next = vel + acc * dt
        pos_next = pos + vel_next * dt
        return pos_next, vel_next

    c = damping / mass
    exp_term = th.exp(th.tensor(-c * dt, dtype=vel.dtype, device=vel.device))
    v_inf = force / damping
    vel_next = v_inf + (vel - v_inf) * exp_term
    pos_next = pos + v_inf * dt + (vel - v_inf) * ((1.0 - exp_term) / c)
    return pos_next, vel_next

def _euler_step(
    state: SimulationState,
    voltage: th.Tensor,
    f_basis: th.Tensor,
    grid_limit: float,
    dt: float,
    damping: float,
    mass: float,
    resistance: float,
    inductance: float,
 ) -> tuple[SimulationState, dict[str, th.Tensor]]:
    curr_next = _current_update_exact(state.curr, voltage, dt, resistance, inductance)
    force = compute_force(state.pos, curr_next, f_basis, grid_limit)
    pos_next, vel_next = _advance_linear_drag(
        state.pos,
        state.vel,
        force,
        dt,
        damping,
        mass,
    )
    return replace(state, pos=pos_next, vel=vel_next, curr=curr_next), {"force": force, "vel": vel_next}

def _rk2_mid_step(
    state: SimulationState,
    voltage: th.Tensor,
    f_basis: th.Tensor,
    grid_limit: float,
    dt: float,
    damping: float,
    mass: float,
    resistance: float,
    inductance: float,
 ) -> tuple[SimulationState, dict[str, th.Tensor]]:
    curr_next = _current_update_exact(state.curr, voltage, dt, resistance, inductance)
    curr_mid = 0.5 * (state.curr + curr_next)
    half_force = compute_force(state.pos, curr_mid, f_basis, grid_limit)
    x_half, v_half = _advance_linear_drag(state.pos, state.vel, half_force, 0.5 * dt, damping, mass)
    full_force = compute_force(x_half, curr_mid, f_basis, grid_limit)
    pos_next, vel_next = _advance_linear_drag(state.pos, state.vel, full_force, dt, damping, mass)
    return replace(state, pos=pos_next, vel=vel_next, curr=curr_next), {"force": full_force, "vel": vel_next}

_STEP_FUNCTIONS: dict[str, StepFn] = {
    "euler": _euler_step,
    "rk2":_rk2_mid_step,
}


def _resolve_open_loop_voltage(
    open_loop_voltage: th.Tensor | list[float] | list[list[float]] | None,
    batch_size: int,
    n_coils: int,
    state_dtype: th.dtype,
    state_device: th.device,
) -> th.Tensor | None:
    if open_loop_voltage is None:
        return None
    v = th.as_tensor(open_loop_voltage, dtype=state_dtype, device=state_device)
    if v.ndim == 1:
        if v.numel() != n_coils:
            raise ValueError(f"experiment.open_loop_voltage must have {n_coils} values, got {v.numel()}.")
        return v.unsqueeze(0).repeat(batch_size, 1)
    if v.ndim == 2:
        if v.shape[0] != batch_size:
            raise ValueError(
                f"experiment.open_loop_voltage batch dimension must match batch_size ({batch_size}), got {v.shape[0]}."
            )
        if v.shape[1] != n_coils:
            raise ValueError(f"experiment.open_loop_voltage must have {n_coils} columns, got {v.shape[1]}.")
        return v
    raise ValueError("experiment.open_loop_voltage must be shape (4,) or (B,4).")


def integrate(
    config: dict,
    state: SimulationState,
    f_basis: th.Tensor,
    u_basis: th.Tensor,
    grid_limit: float,
    controller: ControllerFn,
    damping: float,
) -> dict:
    numerics = config["numerics"]
    model = config["model"]
    experiment = config["experiment"]

    dt = float(numerics.get("dt", 1e-3))
    t_max = float(numerics.get("t_max", 20.0))
    steps = int(numerics.get("steps", max(1, round(t_max / dt))))
    integrator = str(numerics.get("integrator", "euler")).lower()
    step_fn = _STEP_FUNCTIONS.get(integrator)
    if step_fn is None:
        supported = ", ".join(sorted(_STEP_FUNCTIONS))
        raise NotImplementedError(f"Unsupported integrator '{integrator}'. Currently implemented: {supported}.")
    mode = str(experiment.get("mode", "closed"))
    record_positions = bool(numerics.get("record_positions", True))
    record_potential = bool(numerics.get("record_potential", True))
    history_stride = max(1, int(numerics.get("history_stride", 1)))
    potential_stride = max(1, int(numerics.get("potential_stride", 1)))
    stop_tolerance = float(experiment.get("stop_tolerance", 1e-6))
    history_device = th.device(numerics.get("history_device", "cpu"))
    mass = float(model.get("particle_mass", 1.0))
    resistance = float(model.get("coil_resistance", 1.0))
    inductance = float(model.get("coil_inductance", 1.0))
    viscosity = float(model.get("viscosity", 1e-3))
    hydrodynamic_radius = float(model.get("hydrodynamic_radius", model.get("particle_radius", 3e-6)))
    fluid_density = float(model.get("fluid_density", 1000.0))
    characteristic_length = 2.0 * hydrodynamic_radius
    if mass <= 0.0:
        raise ValueError("model.particle_mass must be > 0")
    if resistance <= 0.0:
        raise ValueError("model.coil_resistance must be > 0")
    if inductance <= 0.0:
        raise ValueError("model.coil_inductance must be > 0")
    initial_pos = state.pos.clone()

    pos_hist = []
    curr_hist = []
    voltage_hist = []
    pot_hist = []

    re_history: list[float] = []
    force_magnitude_history: list[float] = []
    max_force_magnitude = 0.0
    max_reynolds_number = 0.0
    max_position_norm = 0.0
    nan_inf_detected = False

    fixed_voltage = None
    if mode == "open":
        fixed_voltage = _resolve_open_loop_voltage(
            experiment.get("open_loop_voltage"),
            batch_size=state.pos.shape[0],
            n_coils=state.curr.shape[1],
            state_dtype=state.curr.dtype,
            state_device=state.curr.device,
        )
        if fixed_voltage is None:
            fixed_voltage = controller(state)

    executed_steps = 0
    for step_i in range(steps):
        voltage_cmd = controller(state) if mode == "closed" else fixed_voltage
        state, diag = step_fn(
            state,
            voltage_cmd,
            f_basis,
            grid_limit,
            dt,
            damping,
            mass,
            resistance,
            inductance,
        )
        force = diag["force"]
        vel_next = diag["vel"]

        force_mag = force.norm(dim=-1)
        force_mag_device = force_mag.detach()
        mean_force = float(force_mag_device.mean().item())
        force_magnitude_history.append(mean_force)
        max_force_magnitude = max(max_force_magnitude, float(force_mag_device.max().item()))
        if not th.isfinite(force).all():
            nan_inf_detected = True

        vel_norm = vel_next.norm(dim=-1)
        reynolds = fluid_density * vel_norm * characteristic_length / viscosity
        reynolds_values = reynolds.detach()
        mean_re = float(reynolds_values.mean().item())
        re_history.append(mean_re)
        max_reynolds_number = max(max_reynolds_number, float(reynolds_values.max().item()))

        position_norm = state.pos.norm(dim=-1)
        max_position_norm = max(max_position_norm, float(position_norm.detach().max().item()))
        if not th.isfinite(state.pos).all():
            nan_inf_detected = True

        if step_i % history_stride == 0:
            curr_hist.append(state.curr.detach().to(history_device))
            voltage_hist.append(voltage_cmd.detach().to(history_device))
            if record_positions:
                pos_hist.append(state.pos.detach().to(history_device))
        if record_potential and (step_i % potential_stride == 0):
            pot_hist.append(compute_potential(state.curr, u_basis).detach().to(history_device))

        dist = th.norm(state.pos - state.target, dim=1)
        executed_steps = step_i + 1
        if bool((dist < stop_tolerance).all()):
            break

    final_target_error = float(th.norm(state.pos - state.target, dim=1).max().item())
    return {
        "mode": mode,
        "dt": dt,
        "steps_executed": executed_steps,
        "grid_limit": grid_limit,
        "start_pos": initial_pos.detach().to(history_device),
        "target": state.target.detach().to(history_device),
        "pos": pos_hist,
        "curr": curr_hist,
        "voltage": voltage_hist,
        "pot": pot_hist,
        "re_history": re_history,
        "force_magnitude_history": force_magnitude_history,
        "max_force_magnitude": max_force_magnitude,
        "max_reynolds_number": max_reynolds_number,
        "max_position_norm": max_position_norm,
        "nan_inf_detected": nan_inf_detected,
        "final_target_error": final_target_error,
    }
