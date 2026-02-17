import numpy as np
import matplotlib
import os

_SHOW_PLOT = os.environ.get("SHOW_PLOT") == "1"
_SHOW_ANIMATION = os.environ.get("SHOW_ANIMATION") == "1"
if not (_SHOW_PLOT or _SHOW_ANIMATION):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _should_show_plot():
    return _SHOW_PLOT


def _to_numpy(array_like):
    if hasattr(array_like, "detach"):
        return array_like.detach().cpu().numpy()
    return np.asarray(array_like)


def plot_particle_paths(
    positions_over_time,
    labels=None,
    grid_limit=None,
    physical_width=None,
    target=None,
    title=None,
    save_path=None,
):
    data = _to_numpy(positions_over_time)
    if data.ndim != 3 or data.shape[2] < 2:
        raise ValueError("positions_over_time must be shaped (T, N, 2+) for plotting.")

    n_particles = data.shape[1]
    if labels is None:
        labels = [f"p{i + 1}" for i in range(n_particles)]

    fig, ax = plt.subplots(figsize=(7, 7))
    for i in range(n_particles):
        label = labels[i] if i < len(labels) else f"p{i + 1}"

        (line,) = ax.plot(data[:, i, 0], data[:, i, 1], label=label)

        # Start marker
        ax.scatter(
            data[0, i, 0],
            data[0, i, 1],
            marker="x",
            color=line.get_color(),
            s=100,
            zorder=5,
            label="start" if i == 0 else None,
        )

        # Final marker
        ax.scatter(
            data[-1, i, 0],
            data[-1, i, 1],
            marker="o",
            facecolors="none",
            edgecolors=line.get_color(),
            s=90,
            linewidths=2,
            zorder=6,
            label="final" if i == 0 else None,
        )

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    if target is not None:
        target_np = _to_numpy(target)
        if target_np.ndim == 1:
            target_np = target_np[None, :]
        ax.scatter(
            target_np[:, 0],
            target_np[:, 1],
            marker="*",
            s=160,
            color="black",
            zorder=6,
            label="target",
        )
    if title:
        ax.set_title(title)
    ax.legend()
    if grid_limit is not None:
        ax.set_xlim(-grid_limit, grid_limit)
        ax.set_ylim(-grid_limit, grid_limit)
    elif physical_width is not None:
        half = physical_width / 2
        ax.set_xlim(-half, half)
        ax.set_ylim(-half, half)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    if _should_show_plot():
        plt.show()
    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    return fig

def plot_particle_velocity(
    velocity_over_time,
    labels=None,
    grid_limit=None,
    physical_width=None,
    save_path=None,
):
    data = _to_numpy(velocity_over_time)
    n_particles = data.shape[1]
    if labels is None:
        labels = [f"p{i + 1}" for i in range(n_particles)]
    fig, ax = plt.subplots(figsize=(7, 7))
    for i in range(n_particles):
        label = labels[i] if i < len(labels) else f"p{i + 1}"
        ax.plot(data[:, i, 0], data[:, i, 1], label=label)

    ax.set_xlabel("ux [m/2]")
    ax.set_ylabel("uy [m/2]")
    ax.legend()
    if grid_limit is not None:
        ax.set_xlim(-grid_limit, grid_limit)
        ax.set_ylim(-grid_limit, grid_limit)
    elif physical_width is not None:
        half = physical_width / 2
        ax.set_xlim(-half, half)
        ax.set_ylim(-half, half)
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    if _should_show_plot():
        plt.show()
    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    return fig

def plot_coil_currents(currents_over_time, dt, labels=None, save_path=None):

    data = _to_numpy(currents_over_time)
    steps, n_coils = data.shape
    
    # Create time axis in seconds
    time_axis = np.linspace(0, steps * dt, steps)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    if labels is None:
        labels = [f"Coil {i+1}" for i in range(n_coils)]
        
    for i in range(n_coils):
        ax.plot(time_axis, data[:, i], label=labels[i], linewidth=2)

    ax.set_title("Coil Control Currents")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Current [A]")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Optional: Add a horizontal line at 0 for reference
    ax.axhline(0, color="black", lw=1, ls="--")

    fig.tight_layout()
    if _should_show_plot():
        plt.show()
    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    return fig

def plot_position_error(
    positions_over_time,
    target,
    dt=None,
    labels=None,
    save_path=None,
):
    data = _to_numpy(positions_over_time)
    if data.ndim != 3:
        raise ValueError("positions_over_time must be shaped (T, N, 2+) for plotting.")
    target_np = _to_numpy(target)
    if target_np.ndim == 1:
        target_np = target_np[None, :]
    if target_np.shape[0] == 1 and data.shape[1] > 1:
        target_np = np.repeat(target_np, data.shape[1], axis=0)
    if target_np.shape[0] != data.shape[1]:
        raise ValueError("target must have shape (N, D) or (D,) matching positions_over_time.")

    dims = min(data.shape[2], target_np.shape[1])
    error = data[:, :, :dims] - target_np[None, :, :dims]
    error_norm = np.linalg.norm(error, axis=2)

    steps = error_norm.shape[0]
    if dt is None:
        time_axis = np.arange(steps)
        xlabel = "Step"
    else:
        time_axis = np.linspace(0, (steps - 1) * dt, steps)
        xlabel = "Time [s]"

    fig, ax = plt.subplots(figsize=(10, 5))
    if labels is None:
        labels = [f"p{i + 1}" for i in range(error_norm.shape[1])]
    for i in range(error_norm.shape[1]):
        label = labels[i] if i < len(labels) else f"p{i + 1}"
        ax.plot(time_axis, error_norm[:, i], label=label, linewidth=2)

    ax.set_title("Position Error Magnitude")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("||position - target|| [m]")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    if _should_show_plot():
        plt.show()
    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    return fig
