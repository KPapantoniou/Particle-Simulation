import numpy as np
import matplotlib.pyplot as plt


def _to_numpy(array_like):
    if hasattr(array_like, "detach"):
        return array_like.detach().cpu().numpy()
    return np.asarray(array_like)


def plot_particle_paths(positions_over_time, labels=None, grid_limit=None, physical_width=None):
    data = _to_numpy(positions_over_time)
    if data.ndim != 3 or data.shape[2] < 2:
        raise ValueError("positions_over_time must be shaped (T, N, 2+) for plotting.")

    n_particles = data.shape[1]
    if labels is None:
        labels = [f"p{i + 1}" for i in range(n_particles)]

    for i in range(n_particles):
        label = labels[i] if i < len(labels) else f"p{i + 1}"
        
        # 1. Plot the path line
        line, = plt.plot(data[:, i, 0], data[:, i, 1], label=label)
        
        # 2. Add an 'X' at the very first position (time index 0)
        # Use the same color as the line for clarity
        plt.scatter(data[0, i, 0], data[0, i, 1], 
                    marker='x', 
                    color=line.get_color(), 
                    s=100,      # size of the cross
                    zorder=5)   # ensure it stays on top of the line

    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.legend()
    if grid_limit is not None:
        plt.xlim(-grid_limit, grid_limit)
        plt.ylim(-grid_limit, grid_limit)
    elif physical_width is not None:
        half = physical_width / 2
        plt.xlim(-half, half)
        plt.ylim(-half, half)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.show()

def plot_particle_velocity(velocity_over_time, labels=None, grid_limit=None, physical_width=None):
    data = _to_numpy(velocity_over_time)
    n_particles = data.shape[1]
    if labels is None:
        labels = [f"p{i + 1}" for i in range(n_particles)]
    for i in range(n_particles):
        label = labels[i] if i < len(labels) else f"p{i + 1}"
        plt.plot(data[:, i, 0], data[:, i, 1], label=label)

    plt.xlabel("ux [m/2]")
    plt.ylabel("uy [m/2]")
    plt.legend()
    if grid_limit is not None:
        plt.xlim(-grid_limit, grid_limit)
        plt.ylim(-grid_limit, grid_limit)
    elif physical_width is not None:
        half = physical_width / 2
        plt.xlim(-half, half)
        plt.ylim(-half, half)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.show()

def plot_coil_currents(currents_over_time, dt, labels=None):

    data = _to_numpy(currents_over_time)
    steps, n_coils = data.shape
    
    # Create time axis in seconds
    time_axis = np.linspace(0, steps * dt, steps)
    
    plt.figure(figsize=(10, 5))
    
    if labels is None:
        labels = [f"Coil {i+1}" for i in range(n_coils)]
        
    for i in range(n_coils):
        plt.plot(time_axis, data[:, i], label=labels[i], linewidth=2)
    
    plt.title("Coil Control Currents")
    plt.xlabel("Time [s]")
    plt.ylabel("Current [A]")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Optional: Add a horizontal line at 0 for reference
    plt.axhline(0, color='black', lw=1, ls='--')
    
    plt.show()