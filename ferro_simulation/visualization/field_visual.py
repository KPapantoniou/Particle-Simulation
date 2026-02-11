import numpy as np
import matplotlib.pyplot as plt



def _to_numpy(array_like):
    if hasattr(array_like, "detach"):
        return array_like.detach().cpu().numpy()
    return np.asarray(array_like)


def grid_visualizer(B_grid, grid_limit=None, physical_width=None):
    Bz_values = _to_numpy(B_grid)[:, :, 2]
    nx, ny = Bz_values.shape
    extent = None
    if grid_limit is not None:
        extent = [-grid_limit, grid_limit, -grid_limit, grid_limit]
    elif physical_width is not None:
        half = physical_width / 2
        extent = [-half, half, -half, half]

    plt.figure(figsize=(8, 6))
    plt.imshow(
        Bz_values,
        extent=extent if extent is not None else [0, nx, 0, ny],
        origin="lower",
        cmap="viridis",
    )
    plt.colorbar(label="Magnetic Grid")
    plt.title("2D Heatmap of magnetic field")
    plt.xlabel("X Grid Index")
    plt.ylabel("Y Grid Index")
    plt.show()


def visualize_quiver(B_grid, grid_limit=None, physical_width=None):
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

    plt.figure(figsize=(8, 8))
    plt.quiver(X, Y, U, V, color="teal")
    plt.title("Magnetic Field Vectors ($B_x$ vs $B_z$)")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.show()


def visualize_3d_surface(B_grid, title, grid_limit=None, physical_width=None):
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
    plt.show()


def visualize_force_flow(F_grid, grid_limit=None, physical_width=None):
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
    plt.figure(figsize=(10, 8))
    plt.streamplot(y, x, Fy, Fx, color=color_array, cmap="autumn")
    plt.colorbar(label="Force Magnitude")
    plt.title("Force Field Pathways ($F = -\\nabla U$)")
    plt.xlabel("Y-axis (columns)")
    plt.ylabel("X-axis (rows)")
    plt.show()


def visualize_force_magnitude(F_grid, grid_limit=None, physical_width=None):
    F_np = _to_numpy(F_grid)
    F_mag = np.linalg.norm(F_np, axis=2)
    extent = None
    if grid_limit is not None:
        extent = [-grid_limit, grid_limit, -grid_limit, grid_limit]
    elif physical_width is not None:
        half = physical_width / 2
        extent = [-half, half, -half, half]

    plt.figure(figsize=(8, 6))
    plt.imshow(
        F_mag,
        extent=extent if extent is not None else None,
        origin="lower",
        cmap="viridis",
    )
    plt.colorbar(label="Force Magnitude (Newtons)")
    plt.title("Map of Force Intensity")
    plt.show()


def visualize_overlay(B_grid, F_grid, grid_limit=None, physical_width=None):
    B_np = _to_numpy(B_grid)
    F_np = _to_numpy(F_grid)

    bz = B_np[:, :, 2]
    fx = F_np[:, :, 0]
    fy = F_np[:, :, 1]

    plt.figure(figsize=(10, 8))
    extent = None
    if grid_limit is not None:
        extent = [-grid_limit, grid_limit, -grid_limit, grid_limit]
    elif physical_width is not None:
        half = physical_width / 2
        extent = [-half, half, -half, half]

    plt.imshow(
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
    plt.quiver(
        idx_x,
        idx_y,
        fy[::skip, ::skip],
        fx[::skip, ::skip],
        color="red",
        pivot="mid",
        scale=None,
    )

    plt.title("Force Vectors (Red) over Magnetic Field (Blue)")
    plt.show()

def contrabillity_condition(det_control, grid_limit=None):
    det = _to_numpy(det_control)
    plt.imshow(det, extent=[-grid_limit, grid_limit, -grid_limit, grid_limit])
    plt.colorbar(label='Controllability (Determinant magnitude)')
    plt.title("Workspace Controllability Map")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.show()
