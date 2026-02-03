import numpy as np
import torch as th
import matplotlib.pyplot as plt
from core.particle import Particle
from core.field import Field
from core.forces import Forces
from core.dynamics import Dynamics
import sys


device = 'cuda' if th.cuda.is_available() else 'cpu'

def circular_coil(radius=1e-3, n_points=2, z=0.0):
    theta = th.linspace(0, 2*th.pi, n_points)
    x = radius * th.cos(theta)
    y = radius * th.sin(theta)
    z = th.full_like(x, z)

    return th.stack([x, y, z], axis=1)

modes = ["uniform","time_varying","coil"]
mode = ""
if sys.argv[1] in modes:
    mode = sys.argv[1]

rho_iron = 7800  # kg/m^3
r = 3e-6
mass = rho_iron * (4/3) * th.pi * r**3

Ms = 1.7e6
volume = (4/3) * th.pi * r**3
m = Ms * volume   # ≈ 1e-10 A·m²
VISCOSITY = 0.001
damping_coeff = 6*th.pi *VISCOSITY*5e-6
relaxation_time = mass/damping_coeff

p1 = Particle(position=[0.0, 0, 0.005],
              velocity=[0, 0, 0],
              magnetic_moment=[0, 0, m],
              mass=mass,
              radius=5e-6,
              device = device
              )

p2 = Particle(position=[0.01 + 2*3e-6, 0, 0], #~6μm apart
              velocity=[0, 0, 0],
              magnetic_moment=[0, 0, 0.9*m],
              mass=mass,
              radius=5e-6,
              device=device
              )

particles = [p1]

B_field = Field(mode=mode,
                 B0=[0.0,0,0.001],
                 omega = 20.0,
                 coil_points=circular_coil(radius=0.001, n_points=10),
                 I=50,
                 device=device
                )

forces = Forces(damping=damping_coeff, dipole_interactions=False,device=device)
dynamics = Dynamics(gamma=5e3, method ="euler")
print(f"Damping coefficient: {damping_coeff:.2e} N·s/m")
print(f"Reynolds number: ~{(mass*50e-6)/(damping_coeff*1e-3):.2f} (low Re = Stokes flow)")
# print(mass,"/",damping_coeff)
print()
dt = 1e-5
t_max = 0.3
steps = int(t_max/dt)


print("Simulation Parameters:")
print(f"  Time step: {dt*1e6:.0f} μs")
print(f"  Duration: {t_max*1e3:.0f} ms")
print(f"  Total steps: {steps:,}")
print()

positions = {0: [], 1: []}
mag_moments = {0: [], 1: []}

positions_over_time = []

for step in range(steps):
    # 1. Compute forces
    # F_list = forces.compute_forces(particles, B_field)
    dynamics.step(particles, forces, B_field, dt)

    positions_over_time.append([p.position.cpu().numpy() for p in particles])

# -------------------------------
# 6. Plot trajectories
# -------------------------------


positions_over_time = th.tensor(np.array(positions_over_time))
print(positions_over_time[:,0,0], positions_over_time[:,0,1])
plt.plot(positions_over_time[:,0,0], positions_over_time[:,0,1], label='p1')
# plt.plot(positions_over_time[:,1,0], positions_over_time[:,1,1], label='p2')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.legend()
plt.show()

