"""Project-wide constants that define the canonical task contract."""

from __future__ import annotations

LABELS = (
    "Anger",
    "Disgust",
    "Enjoyment",
    "Fear",
    "Other",
    "Sadness",
    "Surprise",
)

LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}
ID_TO_LABEL = {index: label for label, index in LABEL_TO_ID.items()}

EXPECTED_SPLIT_SIZES = {
    "train": 5_548,
    "validation": 686,
    "test": 693,
}

CANONICAL_TASK_ID = "uit_vsmec_7label"
DEFAULT_SEEDS = (42, 123, 2026, 7, 999)

