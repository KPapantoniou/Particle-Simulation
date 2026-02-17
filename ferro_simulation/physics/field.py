from __future__ import annotations

from typing import Any

import torch as th

from .constants import MU0
from .material import magnetic_moment


def build_coil_centers(grid_limit: float, offset_margin: float) -> list[tuple[float, float]]:
    offset = grid_limit - offset_margin
    return [
        (-offset, -offset),
        (-offset, offset),
        (offset, offset),
        (offset, -offset),
    ]


def build_field_bases(config: dict[str, Any], device: th.device) -> tuple[th.Tensor, th.Tensor, float]:
    model = config["model"]
    numerics = config["numerics"]
    
    nx = int(numerics.get("nx", 129))
    ny = int(numerics.get("ny", 129))
    physical_width = float(model.get("physical_width", 3e-3))
    grid_limit = physical_width / 2.0
    dx = physical_width / nx
    coil_radius = float(model.get("coil_radius", 1e-3))
    coil_z_distance = float(model.get("coil_z_distance", 1e-3))
    current_per_coil = float(model.get("coil_current", 1.0))
    m = magnetic_moment(config)

    if "coil_positions" in model:
        centers = [tuple(map(float, p[:2])) for p in model["coil_positions"]]
    else:
        coil_offset_margin = float(model.get("coil_offset_margin", 5e-4))
        centers = build_coil_centers(grid_limit, coil_offset_margin)

    x_idx = th.arange(nx, device=device, dtype=th.float32).view(nx, 1)
    y_idx = th.arange(ny, device=device, dtype=th.float32).view(1, ny)
    x_m = (x_idx - nx // 2) * dx
    y_m = (y_idx - ny // 2) * dx

    u_basis = []
    f_basis = []
    for cx, cy in centers:
        dist_sq = (x_m - cx) ** 2 + (y_m - cy) ** 2
        denom = (coil_radius**2 + coil_z_distance**2 + dist_sq) ** 1.5
        bz = (MU0 * current_per_coil * coil_radius**2) / (2.0 * denom)
        u = -m * bz

        fx = th.empty_like(u)
        fy = th.empty_like(u)
        fx[1:-1, :] = -(u[2:, :] - u[:-2, :]) / (2 * dx)
        fy[:, 1:-1] = -(u[:, 2:] - u[:, :-2]) / (2 * dx)
        fx[0, :] = -(u[1, :] - u[0, :]) / dx
        fx[-1, :] = -(u[-1, :] - u[-2, :]) / dx
        fy[:, 0] = -(u[:, 1] - u[:, 0]) / dx
        fy[:, -1] = -(u[:, -1] - u[:, -2]) / dx

        u_basis.append(u.to(dtype=th.float32))
        f_basis.append(th.stack([fx, fy], dim=-1).to(dtype=th.float32))

    return th.stack(f_basis, dim=0), th.stack(u_basis, dim=0), grid_limit
