from __future__ import annotations
import math
import torch as th


MU0 = 4 * math.pi * 1e-7
KB  = 1.380649e-23

def particle_volume(config: dict) -> float:
    model = config["model"]
    particle_radius = float(model.get("particle_radius", 3e-6))
    return float((4.0 / 3.0) * th.pi * particle_radius**3)


def magnetic_moment(config: dict) -> float:
    model = config["model"]
    ms = float(model.get("Ms", 1.7e6))
    return ms * particle_volume(config)

def langevin(x: th.Tensor) -> th.Tensor:
    out = th.empty_like(x)
    small = x.abs() < 1e-3
    xs = x[small]
    out[small] = xs /3.0 - xs ** 3/45.0
    xl = x[~small]
    out[~small] = 1.0 / th.tanh(xl) - 1.0/xl
    return out

def magnetization(H_magnitude: th.Tensor, config: dict) -> th.Tensor:
    model = config["model"]
    ms = float(model.get("Ms", 1.7e6))
    m_sat = magnetic_moment(config)
    T = float(model.get("temperature",300.0))
    alpha = (MU0*m_sat*H_magnitude)/(KB*T)
    return ms * langevin(alpha)

def effective_moment(H_magnitude: th.Tensor, config: dict) -> th.Tensor:
    V = particle_volume(config)
    return magnetization(H_magnitude,config) *V

def default_damping(config: dict) -> float:
    model = config["model"]
    viscosity = float(model.get("viscosity", 1e-3))
    hydrodynamic_radius = float(model.get("hydrodynamic_radius", model.get("particle_radius", 3e-6)))
    return float(6 * th.pi * viscosity * hydrodynamic_radius)
