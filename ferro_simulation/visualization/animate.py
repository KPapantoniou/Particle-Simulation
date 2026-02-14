import matplotlib
import os

_SHOW_PLOT = os.environ.get("SHOW_PLOT") == "1"
_SHOW_ANIMATION = os.environ.get("SHOW_ANIMATION") == "1"
if not (_SHOW_PLOT or _SHOW_ANIMATION):
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation


class Animate:
    def __init__(
        self, field_frames, grid_limit, path, title="3D Field Surface", zlabel="Field Value"
    ):
        self.field_frames = field_frames
        self.grid_limit = grid_limit
        self.path = path
        self.title = title
        self.zlabel = zlabel

        self.field_count = len(field_frames)
        self.path_count = len(path)
        self.frames = max(self.field_count, self.path_count)
        if self.frames == 0:
            raise ValueError("field_frames and path cannot both be empty.")

        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.cbar = None

        self.ani = FuncAnimation(self.fig, self.animate_field_changes, self.frames, interval=50)

    def save(self, path, fps=20):
        writer = FFMpegWriter(fps=fps, codec="libx264")
        self.ani.save(path, writer=writer)
        if _SHOW_ANIMATION:
            plt.show()

    def show(self):
        if _SHOW_ANIMATION:
            plt.show()

    def _to_numpy(self, array_like):
        if hasattr(array_like, "detach"):
            return array_like.detach().cpu().numpy()
        return np.asarray(array_like)

    def _frame_index(self, frame_idx, sequence_len):
        if sequence_len <= 1 or self.frames <= 1:
            return 0
        scaled = frame_idx * (sequence_len - 1) / (self.frames - 1)
        return int(round(scaled))

    def _extract_scalar_field(self, frame_np):
        if frame_np.ndim == 2:
            return frame_np
        if frame_np.ndim == 3 and frame_np.shape[-1] == 2:
            return np.linalg.norm(frame_np, axis=-1)
        if frame_np.ndim == 3 and frame_np.shape[-1] == 3:
            return np.linalg.norm(frame_np, axis=-1)
        raise ValueError(f"Unsupported frame shape {frame_np.shape}. Expected (Nx,Ny), (Nx,Ny,2), or (Nx,Ny,3).")

    def animate_field_changes(self, i):
        field_idx = self._frame_index(i, self.field_count)
        path_idx = self._frame_index(i, self.path_count)

        frame_np = self._to_numpy(self.field_frames[field_idx])
        Z = self._extract_scalar_field(frame_np)
        nx, ny = Z.shape

        x = np.linspace(-self.grid_limit, self.grid_limit, nx)
        y = np.linspace(-self.grid_limit, self.grid_limit, ny)
        X, Y = np.meshgrid(x, y, indexing="ij")

        self.ax.clear()
        self.ax.set_xlim(-self.grid_limit, self.grid_limit)
        self.ax.set_ylim(-self.grid_limit, self.grid_limit)
        surf = self.ax.plot_surface(X, Y, Z, cmap="magma", edgecolor="none")
        if self.cbar is None:
            self.cbar = self.fig.colorbar(surf, ax=self.ax, shrink=0.5, aspect=5)
            self.cbar.set_label(self.zlabel)
        else:
            self.cbar.update_normal(surf)

        current_pos = self._to_numpy(self.path[path_idx]).flatten()
        px, py = current_pos[0], current_pos[1]
        idx_x = int(((px + self.grid_limit) / (2 * self.grid_limit)) * (nx - 1))
        idx_y = int(((py + self.grid_limit) / (2 * self.grid_limit)) * (ny - 1))
        idx_x = max(0, min(nx - 1, idx_x))
        idx_y = max(0, min(ny - 1, idx_y))

        z_span = Z.max() - Z.min()
        eps = 0.1 * z_span if z_span > 0 else 1e-12
        pz = Z[idx_x, idx_y] + eps

        self.ax.scatter(
            [px],
            [py],
            [pz],
            color="#39FF14",
            s=150,
            edgecolors="white",
            linewidth=1.5,
            label="Particle",
            zorder=5,
        )
        self.ax.set_title(f"{self.title} - Step {i}\nParticle Position: ({px:.5f}, {py:.5f})")
        self.ax.set_xlabel("x [m]")
        self.ax.set_ylabel("y [m]")
        self.ax.set_zlabel(self.zlabel)
        return (surf,)
