from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import _bootstrap  # noqa: F401

from src.configuration import load_config
from src.experiment import resolve_project_root, run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train one controlled neural architecture with isolated run artifacts."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Evaluate the already-inspected test split only after configuration freeze.",
    )
    parser.add_argument("--experiment-group", default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    args = parser.parse_args()

    root = resolve_project_root(args.project_root or args.config.resolve().parent)
    config = load_config(args.config)
    if args.experiment_group:
        config = deepcopy(config)
        config["experiment_group"] = args.experiment_group
    seeds = args.seeds or config["training"]["seeds"]

    for seed in seeds:
        run_directory = run_experiment(
            config=config,
            project_root=root,
            seed=seed,
            device_name=args.device,
            evaluate_test=args.evaluate_test,
            output_root=args.output_root,
        )
        print(f"Completed run: {run_directory}")


if __name__ == "__main__":
    main()

