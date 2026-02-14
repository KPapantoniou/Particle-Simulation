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


def grid_visualizer(B_grid, grid_limit=None, physical_width=None, save_path=None):
    Bz_values = _to_numpy(B_grid)[:, :, 2]
    nx, ny = Bz_values.shape
    extent = None
    if grid_limit is not None:
        extent = [-grid_limit, grid_limit, -grid_limit, grid_limit]
    elif physical_width is not None:
        half = physical_width / 2
        extent = [-half, half, -half, half]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(
        Bz_values,
        extent=extent if extent is not None else [0, nx, 0, ny],
        origin="lower",
        cmap="viridis",
    )
    fig.colorbar(im, ax=ax, label="Magnetic Grid")
    ax.set_title("2D Heatmap of magnetic field")
    ax.set_xlabel("X Grid Index")
    ax.set_ylabel("Y Grid Index")
    fig.tight_layout()
    if _should_show_plot():
        plt.show()
    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    return fig


def visualize_quiver(B_grid, grid_limit=None, physical_width=None, save_path=None):
    B_np = _to_numpy(B_grid)
    nx, ny, _ = B_np.shape
    extent = None
    if grid_limit is not None:
        extent = (-grid_limit, grid_limit, -grid_limit, grid_limit)
    elif physical_width is not None:
        half = physical_width / 2
        extent = (-half, half, -half, half)

    if extent is None:
        i = np.arange(nx, dtype=float)
        j = np.arange(ny, dtype=float)
    else:
        i = np.linspace(extent[0], extent[1], nx)
        j = np.linspace(extent[2], extent[3], ny)
    X, Y = np.meshgrid(i, j, indexing="ij")

    U = B_np[:, :, 0]
    V = B_np[:, :, 2]

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.quiver(X, Y, U, V, color="teal")
    ax.set_title("Magnetic Field Vectors ($B_x$ vs $B_z$)")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    fig.tight_layout()
    if _should_show_plot():
        plt.show()
    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    return fig


def visualize_3d_surface(B_grid, title, grid_limit=None, physical_width=None, save_path=None):
    B_np = _to_numpy(B_grid)
    nx, ny, _ = B_np.shape
    extent = None
    if grid_limit is not None:
        extent = (-grid_limit, grid_limit, -grid_limit, grid_limit)
    elif physical_width is not None:
        half = physical_width / 2
        extent = (-half, half, -half, half)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    if extent is None:
        X, Y = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    else:
        x = np.linspace(extent[0], extent[1], nx)
        y = np.linspace(extent[2], extent[3], ny)
        X, Y = np.meshgrid(x, y, indexing="ij")
    Z = B_np[:, :, 2]

    surf = ax.plot_surface(X, Y, Z, cmap="magma", edgecolor="none")
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    ax.set_title(title)
    if _should_show_plot():
        plt.show()
    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    return fig


def visualize_force_flow(F_grid, grid_limit=None, physical_width=None, save_path=None):
    F_np = _to_numpy(F_grid)
    Fx = F_np[:, :, 0].squeeze()
    Fy = F_np[:, :, 1].squeeze()

    nx, ny = Fx.shape
    if grid_limit is not None:
        x = np.linspace(-grid_limit, grid_limit, nx)
        y = np.linspace(-grid_limit, grid_limit, ny)
    elif physical_width is not None:
        half = physical_width / 2
        x = np.linspace(-half, half, nx)
        y = np.linspace(-half, half, ny)
    else:
        x = np.arange(nx)
        y = np.arange(ny)

    color_array = np.sqrt(Fx**2 + Fy**2)
    fig, ax = plt.subplots(figsize=(10, 8))
    stream = ax.streamplot(y, x, Fy, Fx, color=color_array, cmap="autumn")
    fig.colorbar(stream.lines, ax=ax, label="Force Magnitude")
    ax.set_title("Force Field Pathways ($F = -\\nabla U$)")
    ax.set_xlabel("Y-axis (columns)")
    ax.set_ylabel("X-axis (rows)")
    fig.tight_layout()
    if _should_show_plot():
        plt.show()
    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    return fig


def visualize_force_magnitude(F_grid, grid_limit=None, physical_width=None, save_path=None):
    F_np = _to_numpy(F_grid)
    F_mag = np.linalg.norm(F_np, axis=2)
    extent = None
    if grid_limit is not None:
        extent = [-grid_limit, grid_limit, -grid_limit, grid_limit]
    elif physical_width is not None:
        half = physical_width / 2
        extent = [-half, half, -half, half]

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(
        F_mag,
        extent=extent if extent is not None else None,
        origin="lower",
        cmap="viridis",
    )
    fig.colorbar(im, ax=ax, label="Force Magnitude (Newtons)")
    ax.set_title("Map of Force Intensity")
    fig.tight_layout()
    if _should_show_plot():
        plt.show()
    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    return fig


def visualize_overlay(B_grid, F_grid, grid_limit=None, physical_width=None, save_path=None):
    B_np = _to_numpy(B_grid)
    F_np = _to_numpy(F_grid)

    bz = B_np[:, :, 2]
    fx = F_np[:, :, 0]
    fy = F_np[:, :, 1]

    fig, ax = plt.subplots(figsize=(10, 8))
    extent = None
    if grid_limit is not None:
        extent = [-grid_limit, grid_limit, -grid_limit, grid_limit]
    elif physical_width is not None:
        half = physical_width / 2
        extent = [-half, half, -half, half]

    ax.imshow(
        bz,
        origin="lower",
        alpha=0.6,
        cmap="Blues",
        extent=extent if extent is not None else None,
    )

    skip = 8
    idx_x, idx_y = np.meshgrid(
        np.arange(0, bz.shape[1], skip), np.arange(0, bz.shape[0], skip)
    )
    ax.quiver(
        idx_x,
        idx_y,
        fy[::skip, ::skip],
        fx[::skip, ::skip],
        color="red",
        pivot="mid",
        scale=None,
    )

    ax.set_title("Force Vectors (Red) over Magnetic Field (Blue)")
    fig.tight_layout()
    if _should_show_plot():
        plt.show()
    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    return fig

def contrabillity_condition(det_control, grid_limit=None, save_path=None):
    det = _to_numpy(det_control)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(det, extent=[-grid_limit, grid_limit, -grid_limit, grid_limit])
    fig.colorbar(im, ax=ax, label="Controllability (Determinant magnitude)")
    ax.set_title("Workspace Controllability Map")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    fig.tight_layout()
    if _should_show_plot():
        plt.show()
    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    return fig
