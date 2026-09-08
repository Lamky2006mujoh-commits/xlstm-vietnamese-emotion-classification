from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd

from src.constants import CANONICAL_TASK_ID, EXPECTED_SPLIT_SIZES
from src.evaluation import metrics_from_prediction_file


METRIC_KEYS = ("accuracy", "macro_f1", "weighted_f1")


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild one isolated experiment summary.")
    parser.add_argument("--experiment-dir", type=Path, required=True)
    args = parser.parse_args()
    experiment_dir = args.experiment_dir.resolve()
    manifests = sorted(experiment_dir.glob("*/run_manifest.json"))
    if not manifests:
        raise SystemExit(f"No run manifests found under {experiment_dir}")

    rows = []
    identities = set()
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "complete" or not manifest.get("test_evaluated"):
            continue
        if manifest.get("task_id") != CANONICAL_TASK_ID:
            raise ValueError(
                f"Wrong task in {manifest_path}: {manifest.get('task_id')!r}"
            )
        identity = (manifest["model_name"], int(manifest["seed"]))
        if identity in identities:
            raise ValueError(f"Duplicate model/seed pair: {identity}")
        identities.add(identity)

        prediction_path = manifest_path.parent / manifest["artifacts"]["test_predictions"]
        rebuilt = metrics_from_prediction_file(prediction_path)
        if rebuilt["samples"] != EXPECTED_SPLIT_SIZES["test"]:
            raise ValueError(
                f"{prediction_path} has {rebuilt['samples']} rows; expected {EXPECTED_SPLIT_SIZES['test']}"
            )
        stored = manifest["test_metrics"]
        for metric in METRIC_KEYS:
            if abs(float(rebuilt[metric]) - float(stored[metric])) > 1e-12:
                raise ValueError(
                    f"Metric mismatch for {manifest_path.parent.name}/{metric}: "
                    f"rebuilt={rebuilt[metric]}, stored={stored[metric]}"
                )
        rows.append(
            {
                "task_id": manifest["task_id"],
                "model": manifest["model_name"],
                "backend": manifest["model_backend"],
                "seed": int(manifest["seed"]),
                "parameter_count": int(manifest["parameter_count"]),
                "training_seconds": float(manifest["training_seconds"]),
                "peak_gpu_memory_bytes": manifest["peak_gpu_memory_bytes"],
                **{metric: float(rebuilt[metric]) for metric in METRIC_KEYS},
            }
        )

    if not rows:
        raise SystemExit("No completed test-evaluated runs were found.")
    run_frame = pd.DataFrame(rows).sort_values(["model", "seed"])
    run_frame.to_csv(experiment_dir / "aggregate_runs.csv", index=False, encoding="utf-8-sig")
    summary = run_frame.groupby(["task_id", "model", "backend"]).agg(
        seeds=("seed", "count"),
        parameter_count=("parameter_count", "first"),
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
        weighted_f1_mean=("weighted_f1", "mean"),
        weighted_f1_std=("weighted_f1", "std"),
        training_seconds_mean=("training_seconds", "mean"),
        peak_gpu_memory_bytes_max=("peak_gpu_memory_bytes", "max"),
    )
    summary.to_csv(experiment_dir / "stability_summary.csv", encoding="utf-8-sig")
    print(run_frame.to_string(index=False))
    print("\nStability summary:\n")
    print(summary.to_string())


if __name__ == "__main__":
    main()

