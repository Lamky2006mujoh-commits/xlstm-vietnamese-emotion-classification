"""Metric computation and prediction-file reconstruction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

from .constants import LABELS


def classification_metrics(y_true: Iterable[str], y_pred: Iterable[str]) -> dict[str, object]:
    true_values = list(y_true)
    predicted_values = list(y_pred)
    if len(true_values) != len(predicted_values):
        raise ValueError("y_true and y_pred lengths differ.")
    if not true_values:
        raise ValueError("Cannot evaluate an empty prediction set.")
    unknown = (set(true_values) | set(predicted_values)) - set(LABELS)
    if unknown:
        raise ValueError(f"Predictions contain unknown labels: {sorted(unknown)}")

    return {
        "samples": len(true_values),
        "accuracy": float(accuracy_score(true_values, predicted_values)),
        "macro_f1": float(
            f1_score(true_values, predicted_values, labels=list(LABELS), average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(
                true_values,
                predicted_values,
                labels=list(LABELS),
                average="weighted",
                zero_division=0,
            )
        ),
        "per_class": classification_report(
            true_values,
            predicted_values,
            labels=list(LABELS),
            output_dict=True,
            zero_division=0,
        ),
    }


def save_predictions(
    path: str | Path,
    texts: Iterable[str],
    y_true: Iterable[str],
    y_pred: Iterable[str],
) -> None:
    frame = pd.DataFrame(
        {
            "text": list(texts),
            "true_label": list(y_true),
            "pred_label": list(y_pred),
        }
    )
    if frame.empty:
        raise ValueError("Refusing to save an empty prediction file.")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def metrics_from_prediction_file(path: str | Path) -> dict[str, object]:
    prediction_path = Path(path)
    frame = pd.read_csv(prediction_path)
    required = {"text", "true_label", "pred_label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{prediction_path} is missing columns: {sorted(missing)}")
    return classification_metrics(frame["true_label"], frame["pred_label"])


def save_json(payload: dict[str, object], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)

