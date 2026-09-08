"""End-to-end run orchestration with isolated artifacts and manifests."""

from __future__ import annotations

import json
import os
import platform
import shutil
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from time import perf_counter
from typing import Any

import pandas as pd
import torch
import yaml

from .configuration import serializable_config
from .constants import LABELS
from .data import PreparedData, load_prepared_data, make_loader, save_vocabulary
from .evaluation import classification_metrics, save_json, save_predictions
from .models import build_model, trainable_parameter_count
from .training import load_model_checkpoint, predict, set_seed, train_with_early_stopping


def resolve_project_root(start: str | Path | None = None) -> Path:
    candidate = Path(start or Path.cwd()).resolve()
    if candidate.is_file():
        candidate = candidate.parent
    for directory in (candidate, *candidate.parents):
        if (directory / "configs").is_dir() and (directory / "data" / "raw" / "uit_vsmec").is_dir():
            return directory
    raise FileNotFoundError(
        "Could not locate the project root containing configs/ and data/raw/uit_vsmec/."
    )


def select_device(requested: str) -> torch.device:
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu or cuda.")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was explicitly requested but is unavailable.")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def environment_manifest(device: torch.device) -> dict[str, Any]:
    packages = {}
    for package_name in (
        "numpy",
        "pandas",
        "scikit-learn",
        "PyYAML",
        "torch",
        "tqdm",
    ):
        try:
            packages[package_name] = metadata.version(package_name)
        except metadata.PackageNotFoundError:
            packages[package_name] = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "device": str(device),
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "nvcc_available": shutil.which("nvcc") is not None,
        "conda_available": shutil.which("conda") is not None,
        "native_mlstm_requires_nvcc": False,
        "native_mlstm_requires_conda": False,
    }


def _write_yaml(payload: dict[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(payload, stream, allow_unicode=True, sort_keys=False)


def _prepare_run_directory(
    project_root: Path,
    config: dict[str, Any],
    seed: int,
    output_root: str | Path | None,
) -> Path:
    model_name = str(config["model"]["type"])
    base = Path(output_root).resolve() if output_root else project_root / "results" / config["experiment_group"]
    run_directory = base / f"{model_name}_seed{seed}"
    if run_directory.exists() and any(run_directory.iterdir()):
        raise FileExistsError(
            f"Run directory already contains artifacts: {run_directory}. "
            "Use a new experiment group instead of overwriting evidence."
        )
    run_directory.mkdir(parents=True, exist_ok=True)
    return run_directory


def _build_loaders(data: PreparedData, config: dict[str, Any], seed: int):
    training = config["training"]
    common = {
        "vocabulary": data.vocabulary,
        "max_len": data.max_len,
        "batch_size": int(training["batch_size"]),
        "seed": seed,
        "num_workers": int(training["num_workers"]),
    }
    return {
        "train": make_loader(frame=data.train, shuffle=True, **common),
        "validation": make_loader(frame=data.validation, shuffle=False, **common),
        "test": make_loader(frame=data.test, shuffle=False, **common),
    }


def run_experiment(
    config: dict[str, Any],
    project_root: str | Path,
    seed: int,
    device_name: str = "auto",
    evaluate_test: bool = False,
    output_root: str | Path | None = None,
) -> Path:
    root = resolve_project_root(project_root)
    device = select_device(device_name)
    set_seed(seed)
    started_at = datetime.now(timezone.utc)
    run_directory = _prepare_run_directory(root, config, seed, output_root)

    data_config = config["data"]
    raw_dir = Path(data_config["raw_dir"])
    if not raw_dir.is_absolute():
        raw_dir = root / raw_dir
    data = load_prepared_data(
        raw_dir=raw_dir,
        preprocessing=str(data_config["preprocessing"]),
        max_len=int(data_config["max_len"]),
        min_token_frequency=int(data_config["min_token_frequency"]),
    )
    loaders = _build_loaders(data, config, seed)

    model = build_model(config, vocab_size=len(data.vocabulary), num_classes=len(LABELS))
    parameter_count = trainable_parameter_count(model)
    checkpoint_path = run_directory / "checkpoint_best.pt"

    resolved = serializable_config(config)
    resolved["training"]["active_seed"] = seed
    _write_yaml(resolved, run_directory / "config_resolved.yaml")
    save_json(data.manifest, run_directory / "dataset_manifest.json")
    save_json(environment_manifest(device), run_directory / "environment.json")
    save_vocabulary(data.vocabulary, run_directory / "vocabulary.json")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    history, training_summary = train_with_early_stopping(
        model=model,
        train_loader=loaders["train"],
        validation_loader=loaders["validation"],
        device=device,
        checkpoint_path=checkpoint_path,
        seed=seed,
        epochs=int(config["training"]["epochs"]),
        patience=int(config["training"]["patience"]),
        learning_rate=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
        gradient_clip_norm=float(config["training"]["gradient_clip_norm"]),
    )
    history.to_csv(run_directory / "history.csv", index=False, encoding="utf-8-sig")

    reloaded_model = build_model(config, vocab_size=len(data.vocabulary), num_classes=len(LABELS))
    checkpoint_payload = load_model_checkpoint(reloaded_model, checkpoint_path, device)
    validation_output = predict(reloaded_model, loaders["validation"], device)
    validation_metrics = classification_metrics(validation_output.y_true, validation_output.y_pred)
    save_predictions(
        run_directory / "validation_predictions.csv",
        validation_output.texts,
        validation_output.y_true,
        validation_output.y_pred,
    )
    save_json(validation_metrics, run_directory / "metrics_validation.json")

    test_metrics = None
    if evaluate_test:
        test_output = predict(reloaded_model, loaders["test"], device)
        test_metrics = classification_metrics(test_output.y_true, test_output.y_pred)
        save_predictions(
            run_directory / "test_predictions.csv",
            test_output.texts,
            test_output.y_true,
            test_output.y_pred,
        )
        save_json(test_metrics, run_directory / "metrics_test.json")

    finished_at = datetime.now(timezone.utc)
    run_manifest = {
        "run_id": run_directory.name,
        "experiment_group_id": config["experiment_group"],
        "task_id": config["task_id"],
        "model_name": config["model"]["type"],
        "model_backend": getattr(reloaded_model, "backend_name", "pytorch_builtin"),
        "seed": seed,
        "status": "complete",
        "test_evaluated": bool(evaluate_test),
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": finished_at.isoformat(),
        "parameter_count": parameter_count,
        "best_epoch": int(checkpoint_payload["epoch"]),
        "best_validation_weighted_f1": float(checkpoint_payload["validation_weighted_f1"]),
        "training_seconds": float(training_summary["training_seconds"]),
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
        "artifacts": {
            "checkpoint": "checkpoint_best.pt",
            "config": "config_resolved.yaml",
            "dataset_manifest": "dataset_manifest.json",
            "environment": "environment.json",
            "history": "history.csv",
            "validation_predictions": "validation_predictions.csv",
            "test_predictions": "test_predictions.csv" if evaluate_test else None,
        },
        "validation_metrics": {
            key: validation_metrics[key] for key in ("samples", "accuracy", "macro_f1", "weighted_f1")
        },
        "test_metrics": (
            {key: test_metrics[key] for key in ("samples", "accuracy", "macro_f1", "weighted_f1")}
            if test_metrics
            else None
        ),
    }
    save_json(run_manifest, run_directory / "run_manifest.json")
    return run_directory
