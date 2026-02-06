import numpy as np
import torch as th
from core.particle import Particle
from core.field import Field
from core.forces import Forces
from core.dynamics import Dynamics
from visualization.field_visual import (
    grid_visualizer,
    visualize_quiver,
    visualize_3d_surface,
    visualize_force_flow,
    visualize_force_magnitude,
    visualize_overlay,
    contrabillity_condition,
)
from visualization.plot_particles import (
    plot_particle_paths,
    plot_particle_velocity,
    plot_coil_currents,
)                                          
import sys


device = 'cuda' if th.cuda.is_available() else 'cpu'

modes = ["uniform","time_varying", "grid","coil"]
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
damping_coeff = float(6*th.pi *VISCOSITY*5e-6)
relaxation_time = mass/damping_coeff

Nx,Ny = 129,129
physical_width = 4e-3
dx = physical_width / Nx
grid_limit = physical_width / 2
N_coils =100

x= -(grid_limit-5e-4)
y= -(grid_limit -5e-4)

x2 = x
y2 = -y

x3 = -x
y3 = -y

x4 = -x
y4 = y

target = [-0.0005,-0.001]
k=2

def circular_coil(radius=1e-3, n_points=2, z=0.0):
    theta = th.linspace(0, 2*th.pi, n_points)
    x = radius * th.cos(theta)
    y = radius * th.sin(theta)
    z = th.full_like(x, z)

    return th.stack([x, y, z], axis=1)

p1 = Particle(position=[0.001, 0.001, 0.0],
              velocity=[0, 0, 0],
              magnetic_moment=[0, 0, m],
              mass=mass,
              radius=5e-6,
              device = device
              )

p2 = Particle(position=[grid_limit, 0, 0], #~6μm apart
              velocity=[0, 0, 0],
              magnetic_moment=[0, 0, 0.9*m],
              mass=mass,
              radius=5e-6,
              device=device
              )

particles = [p1]

B_field = Field( Nx,
                Ny,
                radius=1e-3,
                x=x,
                y=y,
                z_distance=5e-5,
                magnetic_moment=m,
                N_coils=N_coils,
                mode=mode,
                B0=[0.0,0,0.001],
                omega = 20.0,
                coil_points=circular_coil(radius=0.001, n_points=10),
                I=1,
                device=device,
                physical_width=physical_width,
                dx=dx,
                grid_limit=grid_limit,
                )

B_field2 =Field(Nx,
                Ny,
                radius=1e-3,
                x=x2,
                y=y2,
                z_distance=5e-5,
                magnetic_moment=m,
                N_coils=N_coils,
                mode=mode,
                B0=[0.0,0,0.001],
                omega = 20.0,
                coil_points=circular_coil(radius=0.001, n_points=10),
                I=1,
                device=device,
                physical_width=physical_width,
                dx=dx,
                grid_limit=grid_limit,
                ) 

B_field3 =Field(Nx,
                Ny,
                radius=1e-3,
                x=x3,
                y=y3,
                z_distance=5e-5,
                magnetic_moment=m,
                N_coils=N_coils,
                mode=mode,
                B0=[0.0,0,0.001],
                omega = 20.0,
                coil_points=circular_coil(radius=0.001, n_points=10),
                I=1,
                device=device,
                physical_width=physical_width,
                dx=dx,
                grid_limit=grid_limit,
                ) 

B_field4 =Field(Nx,
                Ny,
                radius=1e-3,
                x=x4,
                y=y4,
                z_distance=5e-5,
                magnetic_moment=m,
                N_coils=N_coils,
                mode=mode,
                B0=[0.0,0,0.001],
                omega = 20.0,
                coil_points=circular_coil(radius=0.001, n_points=10),
                I=1,
                device=device,
                physical_width=physical_width,
                dx=dx,
                grid_limit=grid_limit,
                ) 

grid1 = B_field.generate_grid()
B_grid = grid1[0]
U_grid = grid1[1]

grid2 = B_field2.generate_grid()
B_grid_2 =grid2[0]
U_grid_2 = grid2[1]

grid3 = B_field3.generate_grid()
B_grid_3 = grid3[0]
U_grid_3 = grid3[1]

grid4 = B_field4.generate_grid()
B_grid_4 = grid4[0]
U_grid_4  = grid4[1]

B_tot = B_grid + B_grid_2 + B_grid_3 + B_grid_4
U_tot = U_grid + U_grid_2 + U_grid_3 + U_grid_4



forces = Forces(damping=damping_coeff, dipole_interactions=False,device=device)

F_grid=forces.magnetic_force(p1,U_tot,dx)
#Checking for contrabillity condition
F1 = forces.magnetic_force(p1, U_grid, dx)    
F2 = forces.magnetic_force(p1, U_grid_2, dx)
F3 = forces.magnetic_force(p1, U_grid_3, dx)
F4 = forces.magnetic_force(p1, U_grid_4, dx)
# Normalize forces so they have length 1
# F1_unit = F1 / (th.norm(F1, dim=-1, keepdim=True) + 1e-30)
# F2_unit = F2 / (th.norm(F2, dim=-1, keepdim=True) + 1e-30)
# F3_unit = F3 / (th.norm(F3, dim=-1, keepdim=True) + 1e-30)
# F4_unit = F4 / (th.norm(F3, dim=-1, keepdim=True) + 1e-30)

# det_control = (F1_unit[:,:,0] * F2_unit[:,:,1]) - (F1_unit[:,:,1] * F2_unit[:,:,0])
# det_control = (F1[:,:,0] * F2[:,:,1]) - (F1[:,:,1] * F2[:,:,0])

G_real = th.stack([F1, F2, F3, F4], dim=-1)


s = th.linalg.svdvals(G_real)

det_control = s[:, :, 1] 

contrabillity_condition(det_control, grid_limit=grid_limit)


dt = 1e-3
t_max = 5
steps = int(t_max/dt)
# 
dynamics = Dynamics(gamma=damping_coeff, method ="euler")

print(f"  Time step: {dt*1e6:.0f} μs")
print(f"  Duration: {t_max*1e3:.0f} ms")
print(f"  Total steps: {steps:,}")
print()

positions = {0: [], 1: []}
mag_moments = {0: [], 1: []}

positions_over_time = []
positions_over_time.append([p.position.cpu().numpy() for p in particles])

velocities_over_time = []
velocities_over_time.append([p.velocity.cpu().numpy() for p in particles])

I = 0
currents_over_time = []
# currents_over_time.append([I])
F = [F1,F2,F3,F4]
for step in range(steps):
    I = dynamics.close_loop_control(p1,F, grid_limit ,target,k)
    currents_over_time.append(I.detach().cpu())
    dynamics.step(particles, F_grid, forces, dt, steps, grid_limit,I)
    positions_over_time.append([p.position.cpu().numpy() for p in particles])
    velocities_over_time.append([p.velocity.cpu().numpy() for p in particles])
    # print(I)
    # currents_over_time.append(i for i in I)
current = th.stack(currents_over_time)
positions_over_time = th.tensor(np.array(positions_over_time))
print(positions_over_time[:,0,0], positions_over_time[:,0,1])
# print(f"Final Position (meters): {particles[0].position[:2]}")

# # Magnetic plots
grid_visualizer(B_tot, grid_limit=grid_limit)
visualize_quiver(B_tot, grid_limit=grid_limit)
visualize_3d_surface(B_tot, grid_limit=grid_limit)
# #Force plots
# visualize_force_flow(F_grid, grid_limit=grid_limit)
# visualize_force_magnitude(F_grid, grid_limit=grid_limit)
# visualize_overlay(B_grid, F_grid, grid_limit=grid_limit)
# contrabillity_condition(det_control,grid_limit=grid_limit)
#particle plots
plot_particle_paths(positions_over_time, labels=["p1"], grid_limit=grid_limit)
plot_coil_currents(current,dt)
# plot_particle_velocity(velocities_over_time,labels=["p1"], grid_limit=grid_limit)