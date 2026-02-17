# Particle-Sim

This project simulates the motion of a ferromagnetic particle on a 2D grid using a force field derived from a magnetic potential. It supports batched runs so multiple particles can be simulated in parallel on CPU or GPU.

## Physics overview

- Magnetic potential: The field is represented on a grid. Each coil produces a scalar potential U(x, y), and the total potential can be formed by a linear combination of basis fields.
- Force: The particle experiences a force from the potential gradient:
  - F = -grad(U)
  - The code computes this using finite differences on the grid (see `ferro_simulation/core/forces.py`).
- Dynamics: The particle state updates in discrete time steps using Euler integration:
  - v = F / damping
  - x = x + v * dt
  - See `ferro_simulation/core/dynamics.py`.

## Control modes

- Closed loop: At each step, the controller computes coil currents that drive the particle toward a target.
  - The controller builds a local control matrix G from the precomputed force grids and uses a pseudo-inverse to solve for currents.
- Open loop: The controller computes currents once and reuses them during the run.

Both modes are handled in `ferro_simulation/run_sim.py`.

## Batched simulation

The simulation is vectorized so multiple particles are updated in a single run. A batch means:

- `positions` is shaped like (B, 3) or (B, N, 3)
- `currents` is shaped like (B, C)
- The force lookup uses the batch dimension, so all particles are updated in one step.

In `ferro_simulation/test_core.py` the logic is:

- If `BATCH_SIZE=1`, a single closed-loop run is executed.
- If `BATCH_SIZE=n` (n > 1):
  - The first particle (index 0) is run in open-loop mode.
  - The remaining particles (indices 1..n-1) are run in closed-loop mode.
  - This is done as two vectorized runs (one for open, one for closed), not sequential single-particle runs.

This keeps the computation batched and avoids running n separate simulations.

## Running 5 simulations in one run

Use environment variables to control the run:

```bash
BATCH_SIZE=5 SHOW_PLOT=1 SHOW_ANIMATION=0 \
  /home/kpapantoniou/MyProjects/Particle-Sim/particle_sim/bin/python \
  /home/kpapantoniou/MyProjects/Particle-Sim/ferro_simulation/test_core.py
```

Behavior with `BATCH_SIZE=5`:
- Particle 1 runs in open loop.
- Particles 2-5 run in closed loop.
- Each step updates the full batch on the selected device (CPU/GPU).

## Outputs

- Trajectories are saved as `.pt` files in `results/` when batch size > 1.
- If `SHOW_PLOT=1`, plots are displayed.
- If `SHOW_PLOT=0`, plots are saved to `results/` with names like:
  - `batch_1_trajectory_open.png`
  - `batch_2_trajectory_closed.png`
  - `batch_2_error_closed.png`
  - `batch_2_currents_closed.png`

To view `.pt` files and regenerate plots:

```bash
python /home/kpapantoniou/MyProjects/Particle-Sim/ferro_simulation/view_pt.py \
  results/trajectories_closed.pt --dt 0.001 --target 0.0,0.0
```
