from __future__ import annotations

import torch as th


def sample_force_basis(pos: th.Tensor, f_basis: th.Tensor, grid_limit: float) -> th.Tensor:
    nx, ny = f_basis.shape[1], f_basis.shape[2]
    idx_x = ((pos[:, 0] + grid_limit) / (2 * grid_limit) * (nx - 1)).long().clamp(0, nx - 1)
    idx_y = ((pos[:, 1] + grid_limit) / (2 * grid_limit) * (ny - 1)).long().clamp(0, ny - 1)
    return f_basis[:, idx_x, idx_y].permute(1, 0, 2)


def compute_force(pos: th.Tensor, currents: th.Tensor, f_basis: th.Tensor, grid_limit: float) -> th.Tensor:
    sampled = sample_force_basis(pos, f_basis, grid_limit)
    force_xy = (currents.unsqueeze(-1) * sampled).sum(dim=1)
    return th.cat([force_xy, th.zeros((pos.shape[0], 1), device=pos.device, dtype=pos.dtype)], dim=1)


def compute_potential(currents: th.Tensor, u_basis: th.Tensor) -> th.Tensor:
    return th.einsum("bc,cxy->bxy", currents, u_basis)
