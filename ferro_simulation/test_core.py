import numpy as np
import torch as th
import matplotlib.pyplot as plt
from core.particle import Particle
from core.field import Field
from core.forces import Forces
from core.dynamics import Dynamics
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
damping_coeff = 6*th.pi *VISCOSITY*5e-6
relaxation_time = mass/damping_coeff

Nx,Ny = 128,128
physical_width = 2e-2
dx = physical_width/Nx
N_coils =100


def circular_coil(radius=1e-3, n_points=2, z=0.0):
    theta = th.linspace(0, 2*th.pi, n_points)
    x = radius * th.cos(theta)
    y = radius * th.sin(theta)
    z = th.full_like(x, z)

    return th.stack([x, y, z], axis=1)

def grid_visualizer(B_grid):
    Bz_values = B_grid[:,:,2].numpy()

    plt.figure(figsize=(8,6))
    plt.imshow(Bz_values, extent=[0,Nx,0,Ny], origin="lower", cmap='viridis')
    plt.colorbar(label="Magnetic Grid")
    plt.title("2D Heatmap of magnetic field")
    plt.xlabel("X Grid Index")
    plt.ylabel("Y Grid Index")
    plt.show()

def visualize_quiver(B_grid):
    Nx, Ny, _ = B_grid.shape

    i = th.arange(Nx).float()
    j = th.arange(Ny).float()
    X, Y = th.meshgrid(i, j, indexing='ij')

    U = B_grid[:, :, 0].detach().cpu().float().numpy() 
    V = B_grid[:, :, 2].detach().cpu().float().numpy() 
    
    plt.figure(figsize=(8, 8))
    # 3. Use the numpy arrays for the coordinates as well
    plt.quiver(X.numpy(), Y.numpy(), U, V, color='teal')
    
    plt.title("Magnetic Field Vectors ($B_x$ vs $B_z$)")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.show()

def visualize_3d_surface(B_grid):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    X, Y = th.meshgrid(th.arange(Nx), th.arange(Ny), indexing='ij')
    Z = B_grid[:, :, 2].numpy()

    surf = ax.plot_surface(X.numpy(), Y.numpy(), Z, cmap='magma', edgecolor='none')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    ax.set_title("3D Field Intensity Surface")
    plt.show()

def visualize_force_flow(F_grid):

    if hasattr(F_grid, 'detach'):
        F_grid = F_grid.detach().cpu().numpy()

    
    Fx = F_grid[:, :, 0].squeeze() 
    Fy = F_grid[:, :, 1].squeeze()
    # print(Fx.shape)
    # shape = Fx.shape
    nx, ny = Fx.shape[0], Fx.shape[1]
    x = np.arange(nx)
    y = np.arange(ny)

    # print(x.shape, y.shape)
    color_array = np.sqrt(Fx**2 + Fy**2)
    # print(color_array.shape)
    plt.figure(figsize=(10, 8))
    
   
    plt.streamplot(y, x, Fy, Fx, color=color_array, cmap='autumn')
    
    plt.colorbar(label='Force Magnitude')
    plt.title("Force Field Pathways ($F = -\\nabla U$)")
    plt.xlabel("Y-axis (columns)")
    plt.ylabel("X-axis (rows)")
    plt.show()

def visualize_force_magnitude(F_grid):
    F_mag = th.norm(F_grid, dim=2).detach().cpu().numpy()

    plt.figure(figsize=(8,6))
    plt.imshow(F_mag, origin='lower', cmap='viridis')
    plt.colorbar(label='Force Magnitude (Newtons)')
    plt.title("Map of Force Intensity")
    plt.show()

def visualize_overlay(B_grid, F_grid):
    bz = B_grid[:, :, 2].detach().cpu().numpy()
    fx = F_grid[:, :, 0].detach().cpu().numpy()
    fy = F_grid[:, :, 1].detach().cpu().numpy()
    
    plt.figure(figsize=(10, 8))
    plt.imshow(bz, origin='lower', alpha=0.6, cmap='Blues')
    
    skip = 8
    idx_x, idx_y = np.meshgrid(np.arange(0, bz.shape[1],skip),
                               np.arange(0, bz.shape[0],skip))
    plt.quiver(idx_x,idx_y, fy[::skip, ::skip], fx[::skip, ::skip],color='red', pivot='mid', scale=None)
    
    plt.title("Force Vectors (Red) over Magnetic Field (Blue)")
    plt.show()

p1 = Particle(position=[0.000045, 0.000045, 0.0],
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

B_field = Field( Nx,
                Ny,
                radius=1e-3,
                x=0,
                y=0,
                z_distance=5e-5,
                magnetic_moment=m,
                N_coils=N_coils,
                mode=mode,
                B0=[0.0,0,0.001],
                omega = 20.0,
                coil_points=circular_coil(radius=0.001, n_points=10),
                I=5,
                device=device
                )

B_grid = B_field.generate_grid()[0]
U_grid = B_field.generate_grid()[1]



# print(B_grid.detach().cpu().tolist())
forces = Forces(damping=damping_coeff, dipole_interactions=False,device=device)
# grid_visualizer(B_grid)
# visualize_quiver(B_grid)
# visualize_3d_surface(B_grid)
# U_scalar = U_grid.sum(dim=-1)

# F_grid = forces.magnetic_force(p1,U_scalar,dx=dx)
F_grid=forces.magnetic_force(p1,U_grid,dx)
# print((F_grid[:,:,0]))
# print(F_grid.shape)
visualize_force_flow(F_grid)
visualize_force_magnitude(F_grid)
visualize_overlay(B_grid,F_grid)

dt = 1e-5
t_max = 0.3
steps = int(t_max/dt)
# 
dynamics = Dynamics(gamma=5e3, method ="euler")



# print(f"Damping coefficient: {damping_coeff:.2e} N·s/m")
# print(f"Reynolds number: ~{(mass*50e-6)/(damping_coeff*1e-3):.2f} (low Re = Stokes flow)")
# # print(mass,"/",damping_coeff)
# print()



# print("Simulation Parameters:")
print(f"  Time step: {dt*1e6:.0f} μs")
print(f"  Duration: {t_max*1e3:.0f} ms")
print(f"  Total steps: {steps:,}")
print()

positions = {0: [], 1: []}
mag_moments = {0: [], 1: []}

positions_over_time = []
positions_over_time.append([p.position.cpu().numpy() for p in particles])
for step in range(steps):
#     # 1. Compute forces
#     # F_list = forces.compute_forces(particles, B_field)
#     dynamics.step(particles, forces, B_field, dt)
    dynamics.step(particles,F_grid,forces,dt,steps)
    positions_over_time.append([p.position.cpu().numpy() for p in particles])

# # -------------------------------
# # 6. Plot trajectories
# # -------------------------------


positions_over_time = th.tensor(np.array(positions_over_time))
print(positions_over_time[:,0,0], positions_over_time[:,0,1])
print(f"Final Position (meters): {particles[0].position[:2]}")
plt.plot(positions_over_time[:,0,0], positions_over_time[:,0,1], label='p1')
# plt.plot(positions_over_time[:,1,0], positions_over_time[:,1,1], label='p2')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.legend()
plt.show()

