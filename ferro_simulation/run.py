from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path

import torch as th

from api import simulate
from experiments.builders import build_run_configs
from sim_io.results import save_results_pt


def _parse_open_loop_voltage(value: str) -> list[float]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "--open-loop-voltage must provide exactly 4 comma-separated values (one per coil)."
        )
    try:
        return [float(p) for p in parts]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--open-loop-voltage must contain numeric values."
        ) from exc


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Thin launcher: args -> configs -> simulate -> save.")
    parser.add_argument("--device", default=os.environ.get("DEVICE", "auto"))
    parser.add_argument(
        "--mode",
        choices=["closed", "open", "both"],
        default=os.environ.get("MODE", "both"),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.environ.get("BATCH_SIZE", "1")),
        help="Simulation batch size (number of particles in one run).",
    )
    parser.add_argument("--batch-seed", type=int, default=int(os.environ["BATCH_SEED"]) if "BATCH_SEED" in os.environ else None)

    parser.add_argument("--dt", type=float, default=float(os.environ.get("DT", "1e-3")))
    parser.add_argument("--t-max", type=float, default=float(os.environ.get("T_MAX", "20.0")))
    parser.add_argument("--history-device", default=os.environ.get("HISTORY_DEVICE", "cpu"))
    parser.add_argument("--history-stride", type=int, default=int(os.environ.get("HISTORY_STRIDE", "1")))
    parser.add_argument("--potential-stride", type=int, default=int(os.environ.get("POTENTIAL_STRIDE", "1")))
    parser.add_argument("--save-potential", action="store_true", default=os.environ.get("SAVE_POTENTIAL", "0") == "1")

    parser.add_argument("--k", type=float, default=float(os.environ.get("K", "1.75")))
    parser.add_argument("--gamma", type=float, default=float(os.environ.get("GAMMA", "1.0")))
    parser.add_argument("--voltage-limit", type=float, default=float(os.environ.get("VOLTAGE_LIMIT", "2.0")))
    parser.add_argument(
        "--open-loop-voltage",
        type=_parse_open_loop_voltage,
        default=_parse_open_loop_voltage(os.environ["OPEN_LOOP_VOLTAGE"])
        if os.environ.get("OPEN_LOOP_VOLTAGE", "").strip()
        else None,
        help="Optional constant 4-coil voltage command for open-loop mode, e.g. '0.5,0,0,-0.5'.",
    )
    parser.add_argument("--k-gain-jitter", type=float, default=float(os.environ.get("K_GAIN_JITTER", "0.0")))
    parser.add_argument("--damping-jitter", type=float, default=float(os.environ.get("DAMPING_JITTER", "0.0")))
    parser.add_argument("--start-margin", type=float, default=float(os.environ.get("START_MARGIN", "0.9")))
    parser.add_argument("--target-margin", type=float, default=float(os.environ.get("TARGET_MARGIN", "0.9")))
    parser.add_argument("--stop-tolerance", type=float, default=float(os.environ.get("STOP_TOLERANCE", "1e-6")))
    parser.add_argument(
        "--dt-sweep",
        type=float,
        nargs="+",
        help="Optional list of timestep values to iterate for stability diagnostics.",
    )
    parser.add_argument("--output-dir", default=os.environ.get("OUTPUT_DIR", "results"))
    return parser


def _dt_label(dt: float) -> str:
    return f"{dt:.0e}".replace("+", "")


def _build_configs_for_dt(args: argparse.Namespace, dt_value: float) -> list[dict]:
    args_copy = copy.deepcopy(args)
    args_copy.dt = dt_value
    return build_run_configs(args_copy)


def _dump_dt_summary(out_dir: Path, summary: list[dict]) -> None:
    if not summary:
        return
    lines = [
        "dt,mode,steps_executed,final_target_error,max_force,max_re,max_position_norm,nan_inf_detected",
    ]
    for entry in summary:
        lines.append(
            ",".join(
                [
                    f"{entry['dt']:.0e}",
                    entry["mode"],
                    str(entry["steps_executed"]),
                    f"{entry['final_target_error']:.6g}",
                    f"{entry['max_force']:.6g}",
                    f"{entry['max_re']:.6g}",
                    f"{entry['max_position_norm']:.6g}",
                    "1" if entry["nan_inf_detected"] else "0",
                ]
            )
        )
    summary_path = out_dir / "dt_sweep_summary.csv"
    summary_path.write_text("\n".join(lines))
    print(f"dt sweep summary written to {summary_path}")


def _run_dt_sweep(args: argparse.Namespace, out_dir: Path) -> None:
    summary = []
    for dt_value in args.dt_sweep:
        configs = _build_configs_for_dt(args, dt_value)
        for cfg in configs:
            result = simulate(cfg)
            print(f"Max Re: {result['max_reynolds_number']}")
            prefix = f"trajectories_{result['mode']}_dt{_dt_label(dt_value)}"
            save_results_pt(result, out_dir / f"{prefix}.pt")
            summary.append(
                {
                    "dt": dt_value,
                    "mode": result["mode"],
                    "steps_executed": result["steps_executed"],
                    "final_target_error": result["final_target_error"],
                    "max_force": result["max_force_magnitude"],
                    "max_re": result["max_reynolds_number"],
                    "max_position_norm": result["max_position_norm"],
                    "nan_inf_detected": result["nan_inf_detected"],
                }
            )
    _dump_dt_summary(out_dir, summary)





def main(argv: list[str] | None = None):
    args = _parser().parse_args(argv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dt_sweep:
        _run_dt_sweep(args, out_dir)
        return

    configs = build_run_configs(args)
    results = [simulate(cfg) for cfg in configs]
   
    for result in results:
        prefix = f"trajectories_{result['mode']}"
        save_results_pt(result, out_dir / f"{prefix}.pt")


if __name__ == "__main__":
    main()
