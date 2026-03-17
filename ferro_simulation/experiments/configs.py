from __future__ import annotations


def base_config() -> dict:
    return {
        "model": {
            "coil_radius": 1e-3,
            "coil_z_distance": 1e-3,
            "coil_offset_margin": 5e-4,
            "coil_resistance": 3,
            "coil_inductance": 5e-7,
            "particle_radius": 3e-6,
            "particle_mass": 8e-13,
            "Ms": 1.7e6,
            "viscosity": 1e-3,
            "fluid_density": 1000.0,
            "hydrodynamic_radius": 5e-6,
            "physical_width": 3e-3,
        },
        "numerics": {
            "nx": 129,
            "ny": 129,
            "dt": 1e-3,
            "t_max": 20.0,
            "device": "cuda",
            "history_device": "cpu",
            "history_stride": 1,
            "potential_stride": 1,
            "record_positions": True,
            "record_potential": True,
        },
        "experiment": {
            "mode": "closed",
            "batch_size": 1,
            "k": 1.75,
            "gamma": 1.0,
            "voltage_limit": 2.0,
            "open_loop_voltage": None,
            "start": "random",
            "target": [0.0, 0.0, 0.0],
        },
    }


def quick_test_config() -> dict:
    cfg = base_config()
    cfg["numerics"]["device"] = "cpu"
    cfg["numerics"]["t_max"] = 0.02
    cfg["numerics"]["record_potential"] = False
    cfg["experiment"]["batch_size"] = 2
    return cfg
