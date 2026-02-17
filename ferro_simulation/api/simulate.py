from __future__ import annotations

from functools import partial

import torch as th

from control.controllers import compute_control_currents
from numerics.integrator import integrate
from numerics.state import initialize_state
from physics.field import build_field_bases
from physics.material import default_damping


_DERIVED_KEYS = {
    "dx",
    "grid_limit",
    "volume",
    "particle_volume",
    "magnetic_moment",
    "drag_coeff",
    "damping",
}

_UNIVERSAL_CONSTANT_KEYS = {"MU0", "mu0", "KB", "kb", "k_b"}


def _validate_config(config: dict) -> None:
    required_sections = ("model", "numerics", "experiment")
    missing = [k for k in required_sections if k not in config]
    if missing:
        raise ValueError(f"Missing config sections: {missing}. Expected model/numerics/experiment.")
    for section in required_sections:
        if not isinstance(config[section], dict):
            raise TypeError(f"config['{section}'] must be a dict.")
        bad = _DERIVED_KEYS.intersection(config[section].keys())
        if bad:
            raise ValueError(f"Do not pass derived quantities in config['{section}']: {sorted(bad)}")
        universal = _UNIVERSAL_CONSTANT_KEYS.intersection(config[section].keys())
        if universal:
            raise ValueError(f"Universal constants are hardcoded in physics and cannot be configured: {sorted(universal)}")
    bad_top_level = _DERIVED_KEYS.intersection(config.keys())
    if bad_top_level:
        raise ValueError(f"Do not pass derived quantities at top-level config: {sorted(bad_top_level)}")
    universal_top = _UNIVERSAL_CONSTANT_KEYS.intersection(config.keys())
    if universal_top:
        raise ValueError(f"Universal constants are hardcoded in physics and cannot be configured: {sorted(universal_top)}")


def simulate(config: dict) -> dict:
    cfg = dict(config)
    _validate_config(cfg)

    numerics = cfg["numerics"]
    experiment = cfg["experiment"]

    device = th.device(numerics.get("device", "cuda" if th.cuda.is_available() else "cpu"))
    numerics["device"] = str(device)
    batch_size = int(experiment.get("batch_size", 1))

    if experiment.get("seed") is not None:
        th.manual_seed(int(experiment["seed"]))

    damping = default_damping(cfg)
    f_basis, u_basis, grid_limit = build_field_bases(cfg, device)
    state = initialize_state(cfg, batch_size, grid_limit, device)
    start_pos = state.pos.clone()

    controller = partial(
        compute_control_currents,
        f_basis=f_basis,
        grid_limit=grid_limit,
        k=float(experiment.get("k", 1.75)),
        gamma=float(experiment.get("gamma", 1.0)),
        current_limit=float(experiment.get("current_limit", 2.0)),
    )
    result = integrate(cfg, state, f_basis, u_basis, grid_limit, controller, damping=damping)
    result["start_pos"] = start_pos.detach().to(th.device(numerics.get("history_device", "cpu")))
    return result
