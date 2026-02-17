from __future__ import annotations

from copy import deepcopy

import torch as th

from experiments.configs import base_config

def _resolve_device(arg_device: str) -> str:
    if arg_device == "auto":
        return "cuda" if th.cuda.is_available() else "cpu"
    return arg_device
    
def build_config(
    *,
    device: str,
    batch_size: int,
    mode: str,
    dt: float,
    t_max: float,
    history_device: str,
    history_stride: int,
    potential_stride: int,
    record_potential: bool,
    k: float,
    gamma: float,
    current_limit: float,
    start_margin: float,
    target_margin: float,
    stop_tolerance: float,
    seed: int | None = None,
    k_gain_jitter: float = 0.0,
    damping_jitter: float = 0.0,
    k_scale: float | None = None,
    damping_scale: float | None = None,
) -> dict:
    cfg = deepcopy(base_config())

    if damping_scale is None:
        damping_scale = 1.0 + float((th.rand(1) * 2 - 1) * damping_jitter)
    if k_scale is None:
        k_scale = 1.0 + float((th.rand(1) * 2 - 1) * k_gain_jitter)

    cfg["numerics"]["device"] = device
    cfg["numerics"]["dt"] = float(dt)
    cfg["numerics"]["t_max"] = float(t_max)
    cfg["numerics"]["history_device"] = history_device
    cfg["numerics"]["history_stride"] = int(history_stride)
    cfg["numerics"]["potential_stride"] = int(potential_stride)
    cfg["numerics"]["record_potential"] = bool(record_potential)
    cfg["numerics"]["record_positions"] = True

    cfg["model"]["hydrodynamic_radius"] = float(cfg["model"]["hydrodynamic_radius"]) * damping_scale

    cfg["experiment"]["mode"] = mode
    cfg["experiment"]["batch_size"] = int(batch_size)
    cfg["experiment"]["k"] = float(k) * k_scale
    cfg["experiment"]["gamma"] = float(gamma)
    cfg["experiment"]["current_limit"] = float(current_limit)
    cfg["experiment"]["start_margin"] = float(start_margin)
    cfg["experiment"]["target_margin"] = float(target_margin)
    cfg["experiment"]["stop_tolerance"] = float(stop_tolerance)
    if seed is not None:
        cfg["experiment"]["seed"] = int(seed)
    return cfg


def _sample_xy(grid_limit: float, margin: float, batch_size: int, device: th.device) -> th.Tensor:
    span = grid_limit * margin
    xy = (th.rand((batch_size, 2), device=device) * 2 - 1) * span
    z = th.zeros((batch_size, 1), device=device)
    return th.cat([xy, z], dim=1)


def build_run_configs(args) -> list[dict]:
    device = _resolve_device(args.device)
    batch_size = int(args.batch_size)
    seed = args.batch_seed
    common = dict(        
        device=device,
        dt=args.dt,
        t_max=args.t_max,
        history_device=args.history_device,
        history_stride=args.history_stride,
        potential_stride=args.potential_stride,
        record_potential=args.save_potential,
        k=args.k,
        gamma=args.gamma,
        current_limit=args.current_limit,
        start_margin=args.start_margin,
        target_margin=args.target_margin,
        stop_tolerance=args.stop_tolerance,
        seed=seed,
        k_gain_jitter=args.k_gain_jitter,
        damping_jitter=args.damping_jitter,
    )
    jitter = dict(
        k_scale=1.0 + float((th.rand(1) * 2 - 1) * args.k_gain_jitter),
        damping_scale=1.0 + float((th.rand(1) * 2 - 1) * args.damping_jitter),
    )

    mode = getattr(args, "mode", "closed")
    if mode == "closed":
        return [build_config(batch_size=batch_size, mode="closed", **common, **jitter)]
    if mode == "open":
        return [build_config(batch_size=batch_size, mode="open", **common, **jitter)]

    # mode == "both": run full-batch open and closed with identical start/target.
    device_t = th.device(device)
    if seed is not None:
        th.manual_seed(int(seed))
    grid_limit = float(base_config()["model"]["physical_width"]) / 2.0
    start_all = _sample_xy(
        grid_limit,
        float(args.start_margin),
        batch_size,
        device_t,
    )
    target_all = _sample_xy(
        grid_limit,
        float(args.target_margin),
        batch_size,
        device_t,
    )

    cfg_open = build_config(batch_size=batch_size, mode="open", **common, **jitter)
    cfg_open["experiment"]["start"] = start_all.tolist()
    cfg_open["experiment"]["target"] = target_all.tolist()

    cfg_closed = build_config(batch_size=batch_size, mode="closed", **common, **jitter)
    cfg_closed["experiment"]["start"] = start_all.tolist()
    cfg_closed["experiment"]["target"] = target_all.tolist()

    return [cfg_open, cfg_closed]
