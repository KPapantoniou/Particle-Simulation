"""
Core Physics Module for Ferromagnetic Particle Simulation

This module implements the foundational physics for simulating
ferromagnetic particle assemblies under external and inter-particle
magnetic fields. It provides classes and functions for:

- Particle representation (position, velocity, magnetic moment)
- Electromagnetic field computation (external coils, uniform fields)
- Force and torque calculation (dipole-dipole interactions, damping)
- Particle dynamics integration (translational and rotational motion)

Mathematical References:
- Torque: τ = m × B
- Force: dipole-dipole F_ij
- Translational motion: F = m * a
- Rotational motion: dm/dt = γ (m × B)
"""

from .particle import Particle
from .field import Field
from .forces import Forces
from .dynamics import Dynamics

__all__ = [
    "Particle",
    "Field",
    "Forces",
    "Dynamics"
]
