from __future__ import annotations

import torch as th

from numerics.state import SimulationState
from physics.forces import sample_force_basis


def compute_desired_voltages(
    pos: th.Tensor,
    target: th.Tensor,
    f_basis: th.Tensor,
    grid_limit: float,
    k: float,
    gamma: float,
    resistance: float,
    voltage_limit: float,
) -> th.Tensor:
    if resistance <= 0.0:
        raise ValueError("resistance must be > 0 for voltage control.")
    sampled = sample_force_basis(pos, f_basis, grid_limit)
    g = sampled.permute(0, 2, 1)
    g_pinv = th.linalg.pinv(g)
    e = -k * (pos[:, :2] - target[:, :2])
    desired_currents = th.bmm(g_pinv, (gamma * e).unsqueeze(-1)).squeeze(-1)
    desired_voltages = resistance * desired_currents
    return th.clamp(desired_voltages, -voltage_limit, voltage_limit)


def compute_control_voltages(
    state: SimulationState,
    f_basis: th.Tensor,
    grid_limit: float,
    k: float,
    gamma: float,
    resistance: float,
    voltage_limit: float,
) -> th.Tensor:
    return compute_desired_voltages(
        state.pos,
        state.target,
        f_basis=f_basis,
        grid_limit=grid_limit,
        k=k,
        gamma=gamma,
        resistance=resistance,
        voltage_limit=voltage_limit,
    )
