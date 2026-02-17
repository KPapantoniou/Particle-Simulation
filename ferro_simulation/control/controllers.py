from __future__ import annotations

import torch as th

from physics.forces import sample_force_basis


def compute_control_currents(
    pos: th.Tensor,
    target: th.Tensor,
    f_basis: th.Tensor,
    grid_limit: float,
    k: float,
    gamma: float,
    current_limit: float,
) -> th.Tensor:
    sampled = sample_force_basis(pos, f_basis, grid_limit)
    g = sampled.permute(0, 2, 1)
    g_pinv = th.linalg.pinv(g)
    e = -k * (pos[:, :2] - target[:, :2])
    currents = th.bmm(g_pinv, (gamma * e).unsqueeze(-1)).squeeze(-1)
    return th.clamp(currents, -current_limit, current_limit)
