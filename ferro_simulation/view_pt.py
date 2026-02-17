#!/usr/bin/env python3
import argparse
import os
from typing import Any, Dict, Optional

os.environ["SHOW_PLOT"] = "1"

import matplotlib.pyplot as plt
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


def _select_batch(x: th.Tensor, batch: Optional[int]) -> th.Tensor:
    if batch is None or x.ndim < 2:
        return x
    if batch < 0 or batch >= x.shape[1]:
        raise IndexError(f"batch index {batch} out of range for B={x.shape[1]}")
    return x[:, batch, ...]


def _as_time_tensor(data: Dict[str, Any], key: str) -> Optional[th.Tensor]:
    tensor_key = f"{key}_tensor"
    if isinstance(data.get(tensor_key), th.Tensor):
        return data[tensor_key]
    value = data.get(key)
    if isinstance(value, th.Tensor):
        return value
    if isinstance(value, list) and value and isinstance(value[0], th.Tensor):
        return th.stack(value, dim=0)
    return None


def _target_tensor_for_batches(data: Dict[str, Any], batch_count: int) -> Optional[th.Tensor]:
    target = data.get("target")
    if not isinstance(target, th.Tensor):
        return None
    t = target.detach().cpu()
    if t.ndim == 1:
        return t.unsqueeze(0).repeat(batch_count, 1)
    if t.ndim == 2:
        if t.shape[0] == 1 and batch_count > 1:
            return t.repeat(batch_count, 1)
        return t
    if t.ndim >= 3:
        return t[0]
    return None


def _print_batch_metrics(pos_t: th.Tensor, target_t: Optional[th.Tensor], dt: float) -> None:
    if target_t is None or pos_t.ndim != 3:
        return
    dims = min(pos_t.shape[2], target_t.shape[1])
    errors = th.norm(pos_t[:, :, :dims] - target_t[None, :, :dims], dim=2)  # (T,B)
    final_error = errors[-1]  # (B,)
    print(f"Mean final error: {final_error.mean().item():.6e}")
    print(f"Worst case: {final_error.max().item():.6e}")
    print(f"Std: {final_error.std(unbiased=False).item():.6e}")

    if pos_t.shape[0] > 1:
        vel = th.diff(pos_t[:, :, :dims], dim=0) / dt  # (T-1,B,D)
        vel_dev = th.norm(vel - vel.mean(dim=1, keepdim=True), dim=2).max(dim=0).values
        print(f"Max velocity deviation (mean over batch): {vel_dev.mean().item():.6e}")

    tol = 1e-6
    conv_steps = []
    for b in range(errors.shape[1]):
        hit = th.nonzero(errors[:, b] < tol, as_tuple=False)
        conv_steps.append(int(hit[0].item()) if hit.numel() > 0 else -1)
    valid = [s for s in conv_steps if s >= 0]
    if valid:
        print(f"Mean time to convergence: {np.mean(valid) * dt:.6e} s")
    else:
        print("Mean time to convergence: not reached")


def _plot_ensemble_envelope(pos_t: th.Tensor, dt: float, save_path: Optional[str]) -> None:
    if pos_t.ndim != 3 or pos_t.shape[2] < 2:
        return
    mean_traj = pos_t.mean(dim=1)  # (T,D)
    std_traj = pos_t.std(dim=1, unbiased=False)  # (T,D)
    steps = pos_t.shape[0]
    t = np.linspace(0, (steps - 1) * dt, steps) if dt is not None else np.arange(steps)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for dim, ax, lbl in [(0, axes[0], "x [m]"), (1, axes[1], "y [m]")]:
        m = mean_traj[:, dim].detach().cpu().numpy()
        s = std_traj[:, dim].detach().cpu().numpy()
        ax.plot(t, m, color="tab:blue", label=f"mean {lbl[0]}")
        ax.fill_between(t, m - s, m + s, color="tab:blue", alpha=0.2, label=f"{lbl[0]} ± 1σ")
        ax.set_ylabel(lbl)
        ax.grid(True, alpha=0.3)
        ax.legend()
    axes[1].set_xlabel("Time [s]" if dt is not None else "Step")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="View plots from a PyTorch .pt file.")
    parser.add_argument("path", nargs="+", help="Path(s) to .pt file(s)")
    parser.add_argument("--dt", type=float, default=None, help="Timestep in seconds for time axis.")
    parser.add_argument("--batch", type=int, default=None, help="Select one batch index (0-based) for debugging.")
    parser.add_argument("--no-error", action="store_true", help="Skip position error plot.")
    parser.add_argument("--no-curr", action="store_true", help="Skip current plot.")
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Optional directory to save generated plots as PNG files.",
    )
    args = parser.parse_args()

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    for path in args.path:
        if not os.path.isfile(path):
            print(f"Skipping missing file: {path}")
            continue

        data: Dict[str, Any] = th.load(path, map_location="cpu")
        describe_loaded(data)

        stem = os.path.splitext(os.path.basename(path))[0]
        dt = args.dt if args.dt is not None else float(data.get("dt", 1.0))
        pos = _as_time_tensor(data, "pos")
        if isinstance(pos, th.Tensor):
            pos_all = pos if pos.ndim == 3 else pos.unsqueeze(1)
            target_batches = _target_tensor_for_batches(data, pos_all.shape[1])

            if args.batch is None:
                _print_batch_metrics(pos_all, target_batches, dt)
                env_save = os.path.join(args.save_dir, f"{stem}_ensemble_envelope.png") if args.save_dir else None
                _plot_ensemble_envelope(pos_all, dt, env_save)

            pos_sel = _select_batch(pos_all, args.batch)
            positions_over_time = pos_sel if pos_sel.ndim == 3 else pos_sel.unsqueeze(1)
            target = None
            if target_batches is not None:
                target_sel = _select_batch(target_batches.unsqueeze(0), args.batch).squeeze(0)
                target = target_sel.numpy() if target_sel.ndim == 1 else target_sel[0].numpy()
            path_save = os.path.join(args.save_dir, f"{stem}_trajectory.png") if args.save_dir else None
            plot_particle_paths(positions_over_time, labels=["p1"], target=target, save_path=path_save)
            if target is not None and not args.no_error:
                err_save = os.path.join(args.save_dir, f"{stem}_position_error.png") if args.save_dir else None
                plot_position_error(
                    positions_over_time,
                    target=target,
                    dt=dt,
                    labels=["p1"],
                    save_path=err_save,
                )

        curr = _as_time_tensor(data, "curr")
        if isinstance(curr, th.Tensor) and not args.no_curr:
            curr_all = curr if curr.ndim == 3 else curr.unsqueeze(1)
            curr_sel = _select_batch(curr_all, args.batch)
            currents_over_time = curr_sel if curr_sel.ndim == 2 else curr_sel.squeeze(1)
            curr_save = os.path.join(args.save_dir, f"{stem}_coil_currents.png") if args.save_dir else None
            plot_coil_currents(currents_over_time, dt=dt, save_path=curr_save)

    if args.save_dir:
        print(f"Saved plots to: {args.save_dir}")


if __name__ == "__main__":
    main()
