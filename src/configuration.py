"""Load and validate experiment YAML without silent notebook overrides."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from .constants import CANONICAL_TASK_ID, DEFAULT_SEEDS


class ConfigurationError(ValueError):
    """Raised when an experiment configuration violates the project contract."""


DEFAULTS: dict[str, Any] = {
    "task_id": CANONICAL_TASK_ID,
    "experiment_group": "final_v1",
    "data": {
        "dataset": "UIT-VSMEC",
        "raw_dir": "data/raw/uit_vsmec",
        "preprocessing": "light",
        "max_len": 80,
        "min_token_frequency": 2,
    },
    "training": {
        "batch_size": 64,
        "epochs": 20,
        "patience": 4,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "gradient_clip_norm": 1.0,
        "selection_metric": "weighted_f1",
        "num_workers": 0,
        "seeds": list(DEFAULT_SEEDS),
    },
    "model": {
        "type": "lstm",
        "embedding_dim": 128,
        "hidden_dim": 64,
        "dropout": 0.3,
    },
}

ALLOWED_TOP_LEVEL_KEYS = {"task_id", "experiment_group", "data", "training", "model", "classical"}
ALLOWED_MODEL_TYPES = {"lstm", "bilstm", "cnn", "xlstm_native", "classical_svm"}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    if not config_path.is_file():
        raise ConfigurationError(f"Configuration file does not exist: {config_path}")

    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}

    if not isinstance(raw, dict):
        raise ConfigurationError("The YAML root must be a mapping.")

    unknown = sorted(set(raw) - ALLOWED_TOP_LEVEL_KEYS)
    if unknown:
        raise ConfigurationError(f"Unknown top-level configuration keys: {unknown}")

    config = _deep_merge(DEFAULTS, raw)
    validate_config(config)
    config["_config_path"] = str(config_path)
    return config


def validate_config(config: dict[str, Any]) -> None:
    if config["task_id"] != CANONICAL_TASK_ID:
        raise ConfigurationError(
            f"This pipeline implements {CANONICAL_TASK_ID!r}; got {config['task_id']!r}."
        )

    data = config["data"]
    training = config["training"]
    model = config["model"]

    if data["preprocessing"] not in {"raw", "light"}:
        raise ConfigurationError("preprocessing must be 'raw' or 'light'.")
    if int(data["max_len"]) <= 0:
        raise ConfigurationError("max_len must be positive.")
    if int(data["min_token_frequency"]) <= 0:
        raise ConfigurationError("min_token_frequency must be positive.")

    model_type = str(model["type"])
    if model_type not in ALLOWED_MODEL_TYPES:
        raise ConfigurationError(
            f"model.type must be one of {sorted(ALLOWED_MODEL_TYPES)}; got {model_type!r}."
        )

    for key in ("batch_size", "epochs", "patience"):
        if int(training[key]) <= 0:
            raise ConfigurationError(f"training.{key} must be positive.")
    for key in ("learning_rate", "gradient_clip_norm"):
        if float(training[key]) <= 0:
            raise ConfigurationError(f"training.{key} must be positive.")
    if float(training["weight_decay"]) < 0:
        raise ConfigurationError("training.weight_decay cannot be negative.")
    if training["selection_metric"] != "weighted_f1":
        raise ConfigurationError("The frozen protocol selects checkpoints by weighted_f1.")

    seeds = training["seeds"]
    if not isinstance(seeds, list) or not seeds or any(not isinstance(seed, int) for seed in seeds):
        raise ConfigurationError("training.seeds must be a non-empty list of integers.")
    if len(set(seeds)) != len(seeds):
        raise ConfigurationError("training.seeds contains duplicates.")

    embedding_dim = int(model["embedding_dim"])
    if embedding_dim <= 0:
        raise ConfigurationError("model.embedding_dim must be positive.")
    dropout = float(model["dropout"])
    if not 0 <= dropout < 1:
        raise ConfigurationError("model.dropout must be in [0, 1).")

    if model_type in {"lstm", "bilstm"} and int(model.get("hidden_dim", 0)) <= 0:
        raise ConfigurationError("LSTM models require a positive model.hidden_dim.")
    if model_type == "cnn":
        kernels = model.get("kernel_sizes", [3, 5, 7])
        if not kernels or any(int(kernel) <= 0 or int(kernel) % 2 == 0 for kernel in kernels):
            raise ConfigurationError("CNN kernel_sizes must be positive odd integers.")
    if model_type == "xlstm_native":
        num_heads = int(model.get("num_heads", 0))
        num_blocks = int(model.get("num_blocks", 0))
        block_size = int(model.get("qkv_proj_blocksize", 0))
        proj_factor = float(model.get("proj_factor", 0))
        if num_heads <= 0 or num_blocks <= 0 or block_size <= 0 or proj_factor <= 0:
            raise ConfigurationError(
                "xlstm_native requires positive num_heads, num_blocks, qkv_proj_blocksize and proj_factor."
            )
        if embedding_dim % num_heads != 0:
            raise ConfigurationError("xLSTM embedding_dim must be divisible by num_heads.")
        if model.get("backend") != "native_pytorch_mlstm":
            raise ConfigurationError("xlstm_native backend must be native_pytorch_mlstm.")
        if model.get("pooling") != "last_valid":
            raise ConfigurationError("The current frozen xLSTM interface requires last_valid pooling.")


def serializable_config(config: dict[str, Any]) -> dict[str, Any]:
    """Remove internal runtime-only fields before saving a resolved YAML."""
    return {key: deepcopy(value) for key, value in config.items() if not key.startswith("_")}
