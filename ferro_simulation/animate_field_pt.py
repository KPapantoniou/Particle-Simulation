#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import torch as th

from visualization.animate import Animate


def _as_time_tensor(data: dict[str, Any], key: str) -> th.Tensor | None:
    tensor_key = f"{key}_tensor"
    value = data.get(tensor_key, data.get(key))
    if isinstance(value, th.Tensor):
        return value
    if isinstance(value, list) and value and isinstance(value[0], th.Tensor):
        return th.stack(value, dim=0)
    return None


def _sanitize_path(path: th.Tensor) -> th.Tensor:
    out = path.clone()
    if out.ndim != 2 or out.shape[1] < 2:
        return out
    finite_xy = th.isfinite(out[:, :2]).all(dim=1)
    if bool(finite_xy.all()):
        return out

    fallback = th.zeros((out.shape[1],), dtype=out.dtype, device=out.device)
    for i in range(out.shape[0]):
        if bool(finite_xy[i]):
            fallback = out[i]
        else:
            out[i] = fallback
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Create MP4 animation of potential field evolution from simulation .pt files.")
    parser.add_argument("path", help="Path to a .pt result file.")
    parser.add_argument("--batch", type=int, default=0, help="Batch index to animate.")
    parser.add_argument("--fps", type=int, default=20, help="Output video FPS.")
    parser.add_argument("--out", type=str, default=None, help="Output mp4 path. Default: <input_stem>_potential_animation.mp4")
    parser.add_argument("--title", type=str, default="Potential Field Evolution", help="Animation title.")
    parser.add_argument("--zlabel", type=str, default="Potential", help="Z-axis label.")
    parser.add_argument("--show", action="store_true", help="Show interactive animation window after saving.")
    args = parser.parse_args()

    if args.show:
        os.environ["SHOW_ANIMATION"] = "1"

    p = Path(args.path)
    if not p.is_file():
        raise FileNotFoundError(f"Missing input file: {p}")

    data: dict[str, Any] = th.load(p, map_location="cpu")
    pot_t = _as_time_tensor(data, "pot")
    pos_t = _as_time_tensor(data, "pos")
    if not isinstance(pot_t, th.Tensor):
        raise ValueError("No potential history found. Re-run simulation with --save-potential.")
    if not isinstance(pos_t, th.Tensor):
        raise ValueError("No position history found in result file.")
    if pot_t.ndim != 4:
        raise ValueError(f"Expected pot tensor shape (T,B,Nx,Ny), got {tuple(pot_t.shape)}")
    if pos_t.ndim != 3:
        raise ValueError(f"Expected pos tensor shape (T,B,D), got {tuple(pos_t.shape)}")
    if args.batch < 0 or args.batch >= pot_t.shape[1]:
        raise IndexError(f"batch index {args.batch} out of range for B={pot_t.shape[1]}")

    field_frames = th.nan_to_num(pot_t[:, args.batch], nan=0.0, posinf=0.0, neginf=0.0)
    path = _sanitize_path(th.nan_to_num(pos_t[:, args.batch], nan=0.0, posinf=0.0, neginf=0.0))
    grid_limit = float(data.get("grid_limit", 1.0))

    out_path = Path(args.out) if args.out else p.with_name(f"{p.stem}_potential_animation.mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    animator = Animate(
        field_frames=field_frames,
        grid_limit=grid_limit,
        path=path,
        title=args.title,
        zlabel=args.zlabel,
    )
    tmp_out = out_path.with_name(f"{out_path.stem}.tmp{out_path.suffix}")
    try:
        animator.save(str(tmp_out), fps=args.fps)
        os.replace(tmp_out, out_path)
    finally:
        if tmp_out.exists():
            tmp_out.unlink()
    print(f"Saved animation to: {out_path}")


if __name__ == "__main__":
    main()
