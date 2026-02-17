from __future__ import annotations

from dataclasses import replace
from typing import Callable

import torch as th

from physics.forces import compute_force, compute_potential
from .state import SimulationState


ControllerFn = Callable[[th.Tensor, th.Tensor], th.Tensor]


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
    experiment = config["experiment"]

    dt = float(numerics.get("dt", 1e-3))
    t_max = float(numerics.get("t_max", 20.0))
    steps = int(numerics.get("steps", max(1, round(t_max / dt))))
    integrator = str(numerics.get("integrator", "euler")).lower()
    if integrator != "euler":
        raise NotImplementedError(f"Unsupported integrator '{integrator}'. Currently implemented: 'euler'.")
    mode = str(experiment.get("mode", "closed"))
    record_positions = bool(numerics.get("record_positions", True))
    record_potential = bool(numerics.get("record_potential", True))
    history_stride = max(1, int(numerics.get("history_stride", 1)))
    potential_stride = max(1, int(numerics.get("potential_stride", 1)))
    stop_tolerance = float(experiment.get("stop_tolerance", 1e-6))
    history_device = th.device(numerics.get("history_device", "cpu"))
    initial_pos = state.pos.clone()

    pos_hist = []
    curr_hist = []
    pot_hist = []

    fixed_currents = None
    if mode == "open":
        fixed_currents = controller(state.pos, state.target)

    executed_steps = 0
    for step_i in range(steps):
        currents = controller(state.pos, state.target) if mode == "closed" else fixed_currents
        force = compute_force(state.pos, currents, f_basis, grid_limit)
        vel = force / damping
        pos = state.pos + vel * dt
        state = replace(state, pos=pos, vel=vel)

        if step_i % history_stride == 0:
            curr_hist.append(currents.detach().to(history_device))
            if record_positions:
                pos_hist.append(state.pos.detach().to(history_device))
        if record_potential and (step_i % potential_stride == 0):
            pot_hist.append(compute_potential(currents, u_basis).detach().to(history_device))

        dist = th.norm(state.pos - state.target, dim=1)
        executed_steps = step_i + 1
        if bool((dist < stop_tolerance).all()):
            break

    return {
        "mode": mode,
        "dt": dt,
        "steps_executed": executed_steps,
        "grid_limit": grid_limit,
        "start_pos": initial_pos.detach().to(history_device),
        "target": state.target.detach().to(history_device),
        "pos": pos_hist,
        "curr": curr_hist,
        "pot": pot_hist,
    }
