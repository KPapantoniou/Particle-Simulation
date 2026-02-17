from __future__ import annotations

import torch as th


def particle_volume(config: dict) -> float:
    model = config["model"]
    particle_radius = float(model.get("particle_radius", 3e-6))
    return float((4.0 / 3.0) * th.pi * particle_radius**3)


def magnetic_moment(config: dict) -> float:
    model = config["model"]
    ms = float(model.get("Ms", 1.7e6))
    return ms * particle_volume(config)


def default_damping(config: dict) -> float:
    model = config["model"]
    viscosity = float(model.get("viscosity", 1e-3))
    hydrodynamic_radius = float(model.get("hydrodynamic_radius", model.get("particle_radius", 3e-6)))
    return float(6 * th.pi * viscosity * hydrodynamic_radius)
