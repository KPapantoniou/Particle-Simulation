#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch as th


def _as_time_tensor(data: dict[str, Any], key: str) -> th.Tensor | None:
    tensor_key = f"{key}_tensor"
    value = data.get(tensor_key, data.get(key))
    if isinstance(value, th.Tensor):
        return value
    if isinstance(value, list) and value and isinstance(value[0], th.Tensor):
        return th.stack(value, dim=0)
    return None


def _pick_frame_index(frame: int, total: int) -> int:
    idx = frame if frame >= 0 else total + frame
    if idx < 0 or idx >= total:
        raise IndexError(f"frame index {frame} out of range for T={total}")
    return idx


def _plot_heatmap(field_2d: th.Tensor, grid_limit: float | None, title: str, save_path: Path | None) -> None:
    z = field_2d.detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(8, 6))
    if grid_limit is not None:
        extent = [-grid_limit, grid_limit, -grid_limit, grid_limit]
        im = ax.imshow(z, origin="lower", extent=extent, cmap="viridis")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
    else:
        im = ax.imshow(z, origin="lower", cmap="viridis")
        ax.set_xlabel("x index")
        ax.set_ylabel("y index")
    fig.colorbar(im, ax=ax, label="Potential")
    ax.set_title(title)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


def _plot_surface(field_2d: th.Tensor, grid_limit: float | None, title: str, save_path: Path | None) -> None:
    z = field_2d.detach().cpu().numpy()
    nx, ny = z.shape
    if grid_limit is None:
        x = th.arange(nx).numpy()
        y = th.arange(ny).numpy()
    else:
        x = th.linspace(-grid_limit, grid_limit, nx).numpy()
        y = th.linspace(-grid_limit, grid_limit, ny).numpy()
    xx, yy = th.meshgrid(th.as_tensor(x), th.as_tensor(y), indexing="ij")

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(xx.numpy(), yy.numpy(), z, cmap="magma", edgecolor="none")
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label="Potential")
    ax.set_title(title)
    ax.set_xlabel("x [m]" if grid_limit is not None else "x index")
    ax.set_ylabel("y [m]" if grid_limit is not None else "y index")
    ax.set_zlabel("Potential")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot 2D/3D electromagnetic potential fields from simulation .pt files.")
    parser.add_argument("path", nargs="+", help="Path(s) to .pt result files.")
    parser.add_argument("--batch", type=int, default=0, help="Batch index to visualize.")
    parser.add_argument("--frame", type=int, default=-1, help="Time frame index (default -1 means last frame).")
    parser.add_argument("--no-heatmap", action="store_true", help="Skip 2D heatmap output.")
    parser.add_argument("--no-surface", action="store_true", help="Skip 3D surface output.")
    parser.add_argument("--save-dir", type=str, default=None, help="Directory to save PNG files.")
    args = parser.parse_args()

    save_dir = Path(args.save_dir) if args.save_dir else None
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)

    for file_path in args.path:
        p = Path(file_path)
        if not p.is_file():
            print(f"Skipping missing file: {p}")
            continue
        data: dict[str, Any] = th.load(p, map_location="cpu")
        pot_t = _as_time_tensor(data, "pot")
        if not isinstance(pot_t, th.Tensor):
            print(f"Skipping {p.name}: no potential history found. Re-run simulation with --save-potential.")
            continue
        if pot_t.ndim != 4:
            print(f"Skipping {p.name}: expected pot tensor shape (T,B,Nx,Ny), got {tuple(pot_t.shape)}")
            continue
        if args.batch < 0 or args.batch >= pot_t.shape[1]:
            print(f"Skipping {p.name}: batch index {args.batch} out of range for B={pot_t.shape[1]}")
            continue

        frame_idx = _pick_frame_index(args.frame, pot_t.shape[0])
        field = pot_t[frame_idx, args.batch]
        grid_limit = float(data["grid_limit"]) if "grid_limit" in data else None
        stem = p.stem
        title_suffix = f"{stem} | batch={args.batch} frame={frame_idx}"

        if not args.no_heatmap:
            heatmap_path = save_dir / f"{stem}_potential_heatmap_b{args.batch:03d}_t{frame_idx:05d}.png" if save_dir else None
            _plot_heatmap(field, grid_limit, f"Potential Heatmap | {title_suffix}", heatmap_path)
        if not args.no_surface:
            surface_path = save_dir / f"{stem}_potential_surface_b{args.batch:03d}_t{frame_idx:05d}.png" if save_dir else None
            _plot_surface(field, grid_limit, f"Potential Surface | {title_suffix}", surface_path)

    if save_dir is not None:
        print(f"Saved field plots to: {save_dir}")


if __name__ == "__main__":
    main()
