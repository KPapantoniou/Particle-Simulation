from __future__ import annotations
import torch as th
from torch import Tensor

MU0 = 4.0 * 3.141592653589793 * 1e-7
PI  = 3.141592653589793



def _ellipk(m: Tensor) -> Tensor:
    m  = th.clamp(m, 0.0, 1.0 - 1e-7)
    m1 = 1.0 - m
    a  = m.new_tensor([1.38629436112, 0.09666344259, 0.03590092383,
                       0.03742563713, 0.01451196212])
    b  = m.new_tensor([0.5,           0.12498593597, 0.06880248576,
                       0.03328355346, 0.00441787012])
    A  = a[0] + m1*(a[1] + m1*(a[2] + m1*(a[3] + m1*a[4])))
    B  = b[0] + m1*(b[1] + m1*(b[2] + m1*(b[3] + m1*b[4])))
    return A - B * th.log(m1)


def _ellipe(m: Tensor) -> Tensor:
    m  = th.clamp(m, 0.0, 1.0 - 1e-7)
    m1 = 1.0 - m
    a  = m.new_tensor([1.0,           0.44325141463, 0.06260601220,
                       0.04757383546, 0.01736506451])
    b  = m.new_tensor([0.0,           0.24998368310, 0.09200180037,
                       0.04069697526, 0.00526449639])
    A  = a[0] + m1*(a[1] + m1*(a[2] + m1*(a[3] + m1*a[4])))
    B  =        m1*(b[1] + m1*(b[2] + m1*(b[3] + m1*b[4])))
    return A - B * th.log(m1)



def _bz_from_r(r: Tensor, z: Tensor, R: float, mu0: float = MU0) -> Tensor:

    alpha2 = th.clamp(r**2 + R**2 + z**2 - 2*r*R, min=1e-20)
    beta2  = r**2 + R**2 + z**2 + 2*r*R
    beta   = th.sqrt(th.clamp(beta2, min=1e-30))
    k2     = th.clamp(1.0 - alpha2 / beta2, min=0.0, max=1.0 - 1e-7)

    K  = _ellipk(k2)
    E  = _ellipe(k2)
    C  = mu0 / (2.0 * PI)

    Bz = (C / (alpha2 * beta)) * (
        (R**2 - r**2 - z**2) * E + alpha2 * K
    )
    return Bz



def _coil_force_on_batch(
    pos_xy: Tensor,           
    currents_i: Tensor,    
    cx: float,
    cy: float,
    coil_radius: float,
    coil_z_distance: float,
    particle_moment: float,
) -> Tensor:                 
    
    xy    = pos_xy.detach().requires_grad_(True)     
    x_rel = xy[:, 0] - cx
    y_rel = xy[:, 1] - cy
    r     = th.sqrt(th.clamp(x_rel**2 + y_rel**2, min=1e-20))
    z     = th.full_like(r, -coil_z_distance)

    Bz    = _bz_from_r(r, z, coil_radius)         
    Bz_I  = (Bz * currents_i).sum()                
    grad  = th.autograd.grad(Bz_I, xy)[0]          

    return (particle_moment * grad).detach()         



def sample_force_biot_savart(
    pos: Tensor,                             
    currents: Tensor,                         
    coil_centers: list[tuple[float, float]],
    coil_radius: float,
    coil_z_distance: float,
    particle_moment: float,
) -> Tensor:                                 
    force  = th.zeros(pos.shape[0], 3, device=pos.device, dtype=pos.dtype)
    pos_xy = pos[:, :2]

    for coil_idx, (cx, cy) in enumerate(coil_centers):
        force[:, :2] += _coil_force_on_batch(
            pos_xy,
            currents[:, coil_idx],
            cx, cy,
            coil_radius,
            coil_z_distance,
            particle_moment,
        )

    return force