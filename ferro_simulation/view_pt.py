#!/usr/bin/env python3
import argparse
import os
from typing import Any, Dict, Optional

os.environ.setdefault("SHOW_PLOT", "1")

import numpy as np
import torch as th

from visualization.plot_particles import (
    plot_particle_paths,
    plot_coil_currents,
    plot_position_error,
)


def _describe_tensor(name: str, value: th.Tensor) -> None:
    desc = f"{name}: tensor shape={tuple(value.shape)} dtype={value.dtype} device={value.device}"
    print(desc)
    if value.numel() == 0:
        return
    with th.no_grad():
        v = value.float() if value.dtype.is_floating_point else value
        print(f"  min={v.min().item()} max={v.max().item()}")


def _describe_value(name: str, value: Any) -> None:
    if isinstance(value, th.Tensor):
        _describe_tensor(name, value)
    elif isinstance(value, (list, tuple)) and value and isinstance(value[0], th.Tensor):
        print(f"{name}: list of {len(value)} tensors")
        _describe_tensor(f"{name}[0]", value[0])
    elif isinstance(value, dict):
        print(f"{name}: dict with keys {list(value.keys())}")
    else:
        print(f"{name}: {type(value).__name__}")


def describe_loaded(obj: Any) -> None:
    if isinstance(obj, dict):
        for key in obj:
            _describe_value(str(key), obj[key])
    else:
        _describe_value("data", obj)


def _parse_target(arg: Optional[str]) -> Optional[np.ndarray]:
    if not arg:
        return None
    parts = [p for p in arg.split(",") if p.strip()]
    if len(parts) < 2:
        raise ValueError("target must be at least 2 values, e.g. 0.0,0.0")
    return np.array([float(p) for p in parts], dtype=np.float32)


def _first_batch(x: th.Tensor) -> th.Tensor:
    if x.ndim < 2:
        return x
    return x[:, 0, ...]


def main() -> None:
    parser = argparse.ArgumentParser(description="View plots from a PyTorch .pt file.")
    parser.add_argument("path", help="Path to the .pt file")
    parser.add_argument("--dt", type=float, default=None, help="Timestep in seconds for time axis.")
    parser.add_argument(
        "--target",
        type=str,
        default=None,
        help="Target position as comma-separated values, e.g. 0.0,0.0 or 0.0,0.0,0.0",
    )
    parser.add_argument("--no-error", action="store_true", help="Skip position error plot.")
    parser.add_argument("--no-curr", action="store_true", help="Skip current plot.")
    args = parser.parse_args()

    path = args.path
    if not os.path.isfile(path):
        raise SystemExit(f"File not found: {path}")

    data: Dict[str, Any] = th.load(path, map_location="cpu")
    describe_loaded(data)

    pos = data.get("pos")
    if isinstance(pos, th.Tensor):
        positions_over_time = _first_batch(pos).unsqueeze(1) if pos.ndim == 3 else pos
        plot_particle_paths(positions_over_time, labels=["p1"])
        target = _parse_target(args.target)
        if target is not None and not args.no_error:
            plot_position_error(positions_over_time, target=target, dt=args.dt, labels=["p1"])

    curr = data.get("curr")
    if isinstance(curr, th.Tensor) and not args.no_curr:
        currents_over_time = _first_batch(curr)
        plot_coil_currents(currents_over_time, dt=args.dt or 1.0)


if __name__ == "__main__":
    main()
