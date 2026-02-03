"""""
field.py
---------

Computes magnetic fields at particle positions.

Features:
---------
- Uniform fields
- Coil-based fields (Biot-Savart approximation)
- Time-dependent fields for dynamic control

Equations:
----------
1. External field: B_ext(t)
2. Time-varying: B(r, t) = B0 * sin(ω t)
3. Field from a coil: B(r) = (μ0 I / 4π) integral (dl × (r - r')) / |r - r'|^3

"""

import torch as th

MU0 = 4 * th.pi * 1e-7

class Field:
    def __init__(self, Nx, Ny,radius, x, y, xk, yk, z_distance, mode="uniform", B0=th.tensor([0,0,1.0]), omega = 1.0, coil_points=None, I=1.0, device= 'cuda'):
        self.mode = mode
        self.B0 = th.tensor(B0 if B0 is not None else [0.0,0.0,1.0], dtype=th.float32, device=device)
        self.omega = th.tensor(omega,dtype=th.float32,device=device)
        if coil_points is not None:
            # self.coil_points = th.tensor(coil_points, dtype=th.float32, device=device)
            self.coil_points = coil_points.detach().clone().to(device)
        self.I = I
        self.device = device
        self.Nx = Nx
        self.Ny = Ny
        self.radius = radius
        self.z_distance = z_distance
        self.x =x 
        self.y = y
        self.xk = xk
        self.yk=yk

        

    def evaluate(self, position, t=0.0):

        # r = th.tensor(position, dtype=th.float32, device=self.device)
        r=position
        
        if self.mode == "uniform":
            #constant field
            return self.B0
        elif self.mode == "time_varying":
            B =self.B0 * th.sin(self.omega*t)
            print(B)
            return B
        elif self.mode =="coil":
            if self.coil_points is None or len(self.coil_points)<2:
                raise ValueError("coil_points must be provided with at least 2 points")
            
            # B_tot = th.zeros(3, device=self.device)
            # for i in range(len(self.coil_points)-1):
            #     dl = self.coil_points[i+1] - self.coil_points[i]
            #     r_vec = r - self.coil_points[i]
            #     r_mag = th.norm(r_vec)
            #     r_mag = th.clamp(r_mag,min=1e-6)
            #     if r_mag ==0:
            #         continue
            #     dB = MU0 * self.I / (4 * th.pi) * th.linalg.cross(dl,r_vec) / r_mag**3
            #     B_tot += dB
            r_vec = r.unsqueeze(0) - self.coil_points[:-1]
            dl = self.coil_points[1:] - self.coil_points[:-1]
            r_mag = th.norm(r_vec, dim=1, keepdim=True)**3
            dB=MU0*self.I/(4*th.pi)*th.linalg.cross(dl, r_vec) / (r_mag )
            B_tot = dB.sum(dim=0)
            return B_tot
        raise ValueError(f"Unknown mode {self.mode}. Chose 'uniform' , 'time_varying' or 'coil'")

    def generate_grid(self):
        B_grid = th.zero(self.Nx,self.Ny,3)
        U_energy = B_grid
        Bz = 0
        B = th.tensor([0,0,Bz])
        for j in range(self.Ny-1):
            for i in range(self.Nx-1):
                Bz = (MU0*self.I*(self.radius**2))/2*(self.radius**2+self.z_distance**2+(self))**(3/2)
                
            B_grid[j] = B 