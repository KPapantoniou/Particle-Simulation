from __future__ import annotations

from dataclasses import dataclass

import torch as th


@dataclass(frozen=True)
class SimulationState:
    pos: th.Tensor
    vel: th.Tensor
    target: th.Tensor


def sample_xy(grid_limit: float, margin: float, batch_size: int, device: th.device) -> th.Tensor:
    span = grid_limit * margin
    xy = (th.rand((batch_size, 2), device=device) * 2 - 1) * span
    z = th.zeros((batch_size, 1), device=device)
    return th.cat([xy, z], dim=1)


def _resolve_tensor(
    value: th.Tensor | list[float] | list[list[float]] | None,
    batch_size: int,
    grid_limit: float,
    margin: float,
    device: th.device,
) -> th.Tensor:
    if value is None:
        return sample_xy(grid_limit, margin, batch_size, device)
    tensor = th.as_tensor(value, dtype=th.float32, device=device)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0).repeat(batch_size, 1)
    return tensor


def initialize_state(config: dict, batch_size: int, grid_limit: float, device: th.device) -> SimulationState:
    experiment = config["experiment"]
    start_margin = float(experiment.get("start_margin", 0.9))
    target_margin = float(experiment.get("target_margin", 0.9))

    start_spec = experiment.get("start", "random")
    if isinstance(start_spec, str) and start_spec == "random":
        start_pos = sample_xy(grid_limit, start_margin, batch_size, device)
    else:
        start_pos = _resolve_tensor(start_spec, batch_size, grid_limit, start_margin, device)

    target_spec = experiment.get("target", "random")
    if isinstance(target_spec, str) and target_spec == "random":
        target = sample_xy(grid_limit, target_margin, batch_size, device)
    else:
        target = _resolve_tensor(target_spec, batch_size, grid_limit, target_margin, device)

    vel = th.zeros_like(start_pos)
    return SimulationState(pos=start_pos, vel=vel, target=target)
