"""Data contracts, vocabulary construction and PyTorch loaders."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from .constants import EXPECTED_SPLIT_SIZES, LABELS, LABEL_TO_ID
from .preprocessing import normalize_split


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tokenize(text: str) -> list[str]:
    return str(text).split()


def build_vocabulary(texts: Iterable[str], min_frequency: int = 2) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for text in texts:
        counter.update(tokenize(text))

    vocabulary = {"<pad>": 0, "<unk>": 1}
    retained = sorted(
        ((token, frequency) for token, frequency in counter.items() if frequency >= min_frequency),
        key=lambda item: (-item[1], item[0]),
    )
    for token, _ in retained:
        vocabulary[token] = len(vocabulary)
    return vocabulary


def encode_text(text: str, vocabulary: dict[str, int], max_len: int) -> list[int]:
    token_ids = [vocabulary.get(token, vocabulary["<unk>"]) for token in tokenize(text)]
    token_ids = token_ids[:max_len]
    return token_ids or [vocabulary["<unk>"]]


def _split_integrity(frame: pd.DataFrame, split: str) -> dict[str, object]:
    conflicting = frame.groupby("Sentence")["Emotion"].nunique()
    lengths = frame["Sentence"].astype(str).str.split().str.len()
    return {
        "split": split,
        "rows": int(len(frame)),
        "missing": {key: int(value) for key, value in frame.isna().sum().items()},
        "exact_duplicate_rows": int(frame.duplicated().sum()),
        "duplicate_text_rows": int(frame.duplicated("Sentence").sum()),
        "conflicting_text_count": int((conflicting > 1).sum()),
        "conflicting_texts": sorted(conflicting[conflicting > 1].index.astype(str).tolist()),
        "max_whitespace_tokens": int(lengths.max()),
    }


def validate_raw_dataset(raw_dir: str | Path, strict_sizes: bool = True) -> dict[str, object]:
    raw_path = Path(raw_dir).resolve()
    frames: dict[str, pd.DataFrame] = {}
    manifest: dict[str, object] = {"raw_dir": str(raw_path), "splits": {}, "overlaps": {}}

    for split in ("train", "validation", "test"):
        file_path = raw_path / f"{split}.csv"
        if not file_path.is_file():
            raise FileNotFoundError(f"Missing dataset split: {file_path}")
        frame = pd.read_csv(file_path)
        if list(frame.columns) != ["Sentence", "Emotion"]:
            raise ValueError(
                f"{split}.csv columns must be ['Sentence', 'Emotion']; got {list(frame.columns)}"
            )
        if frame[["Sentence", "Emotion"]].isna().any().any():
            raise ValueError(f"{split}.csv contains missing text or labels.")
        unknown_labels = sorted(set(frame["Emotion"].astype(str)) - set(LABELS))
        missing_labels = sorted(set(LABELS) - set(frame["Emotion"].astype(str)))
        if unknown_labels or missing_labels:
            raise ValueError(
                f"{split}.csv label mismatch; unknown={unknown_labels}, missing={missing_labels}"
            )
        if strict_sizes and len(frame) != EXPECTED_SPLIT_SIZES[split]:
            raise ValueError(
                f"{split}.csv expected {EXPECTED_SPLIT_SIZES[split]} rows; got {len(frame)}"
            )
        frames[split] = frame
        details = _split_integrity(frame, split)
        details["sha256"] = sha256_file(file_path)
        details["label_counts"] = {
            label: int(count)
            for label, count in frame["Emotion"].value_counts().sort_index().items()
        }
        manifest["splits"][split] = details

    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        shared = sorted(
            set(frames[left]["Sentence"].astype(str))
            & set(frames[right]["Sentence"].astype(str))
        )
        manifest["overlaps"][f"{left}__{right}"] = {
            "count": len(shared),
            "texts": shared,
        }

    return manifest


@dataclass
class PreparedData:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    vocabulary: dict[str, int]
    max_len: int
    manifest: dict[str, object]


def load_prepared_data(
    raw_dir: str | Path,
    preprocessing: str,
    max_len: int,
    min_token_frequency: int,
) -> PreparedData:
    manifest = validate_raw_dataset(raw_dir)
    raw_path = Path(raw_dir).resolve()
    frames = {
        split: normalize_split(
            pd.read_csv(raw_path / f"{split}.csv"),
            mode=preprocessing,
        )
        for split in ("train", "validation", "test")
    }
    vocabulary = build_vocabulary(frames["train"]["text"], min_token_frequency)

    manifest["preprocessing"] = preprocessing
    manifest["max_len"] = int(max_len)
    manifest["min_token_frequency"] = int(min_token_frequency)
    manifest["vocabulary_size"] = len(vocabulary)
    manifest["truncated_rows"] = {
        split: int((frame["text"].str.split().str.len() > max_len).sum())
        for split, frame in frames.items()
    }

    return PreparedData(
        train=frames["train"],
        validation=frames["validation"],
        test=frames["test"],
        vocabulary=vocabulary,
        max_len=max_len,
        manifest=manifest,
    )


class EmotionDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, vocabulary: dict[str, int], max_len: int):
        self.texts = frame["text"].tolist()
        self.labels = [LABEL_TO_ID[label] for label in frame["label"]]
        self.vocabulary = vocabulary
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        ids = encode_text(self.texts[index], self.vocabulary, self.max_len)
        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(self.labels[index], dtype=torch.long),
            self.texts[index],
        )


def collate_batch(batch):
    input_ids, labels, texts = zip(*batch)
    lengths = torch.tensor([len(sequence) for sequence in input_ids], dtype=torch.long)
    padded = pad_sequence(input_ids, batch_first=True, padding_value=0)
    return padded, lengths, torch.stack(labels), list(texts)


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader(
    frame: pd.DataFrame,
    vocabulary: dict[str, int],
    max_len: int,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int = 0,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        EmotionDataset(frame, vocabulary, max_len),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_batch,
        worker_init_fn=_seed_worker if num_workers else None,
        generator=generator,
        pin_memory=torch.cuda.is_available(),
    )


def save_vocabulary(vocabulary: dict[str, int], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as stream:
        json.dump(vocabulary, stream, ensure_ascii=False, indent=2, sort_keys=True)

