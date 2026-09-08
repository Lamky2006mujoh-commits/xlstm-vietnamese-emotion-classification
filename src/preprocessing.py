"""Auditable text preprocessing for the canonical UIT-VSMEC experiment."""

from __future__ import annotations

import pandas as pd

NORMALIZE_MAP = {
    "dc": "được",
    "dk": "được",
    "duoc": "được",
    "đc": "được",
    "ng": "người",
    "ngừi": "người",
    "trc": "trước",
    "trk": "trước",
    "cg": "cũng",
    "cug": "cũng",
    "cũg": "cũng",
    "mk": "mình",
    "mik": "mình",
    "mh": "mình",
    "ko": "không",
    "k": "không",
    "kh": "không",
    "hok": "không",
    "hong": "không",
    "j": "gì",
    "z": "vậy",
    "v": "vậy",
}


def normalize_social_text(text: str, mode: str = "light") -> str:
    text = " ".join(text.strip().split())
    if mode == "raw":
        return text
    if mode != "light":
        raise ValueError(f"Unsupported preprocessing mode: {mode!r}")

    tokens = text.lower().split()
    return " ".join(NORMALIZE_MAP.get(token, token) for token in tokens)


def normalize_split(
    frame: pd.DataFrame,
    text_column: str = "Sentence",
    label_column: str = "Emotion",
    mode: str = "light",
) -> pd.DataFrame:
    missing_columns = {text_column, label_column} - set(frame.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    output = frame[[text_column, label_column]].copy()
    output = output.dropna(subset=[text_column, label_column])
    output.columns = ["text", "label"]
    output["text"] = output["text"].astype(str).map(
        lambda value: normalize_social_text(value, mode=mode)
    )
    output["label"] = output["label"].astype(str)
    output = output[output["text"].str.len() > 0]
    return output.reset_index(drop=True)

