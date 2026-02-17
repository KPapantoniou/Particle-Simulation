from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch as th

from api import simulate
from experiments.builders import build_run_configs
from sim_io.results import save_results_pt


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
    parser.add_argument("--current-limit", type=float, default=float(os.environ.get("CURRENT_LIMIT", "2.0")))
    parser.add_argument("--k-gain-jitter", type=float, default=float(os.environ.get("K_GAIN_JITTER", "0.0")))
    parser.add_argument("--damping-jitter", type=float, default=float(os.environ.get("DAMPING_JITTER", "0.0")))
    parser.add_argument("--start-margin", type=float, default=float(os.environ.get("START_MARGIN", "0.9")))
    parser.add_argument("--target-margin", type=float, default=float(os.environ.get("TARGET_MARGIN", "0.9")))
    parser.add_argument("--stop-tolerance", type=float, default=float(os.environ.get("STOP_TOLERANCE", "1e-6")))
    parser.add_argument("--output-dir", default=os.environ.get("OUTPUT_DIR", "results"))
    return parser





def main(argv: list[str] | None = None):
    args = _parser().parse_args(argv)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    configs = build_run_configs(args)
    results = [simulate(cfg) for cfg in configs]
    for result in results:
        prefix = f"trajectories_{result['mode']}"
        save_results_pt(result, out_dir / f"{prefix}.pt")


if __name__ == "__main__":
    main()
