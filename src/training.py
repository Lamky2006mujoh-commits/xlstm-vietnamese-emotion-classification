"""Shared deterministic training and checkpoint-selection helpers."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader

from .constants import ID_TO_LABEL, LABELS


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = not deterministic


@dataclass
class PredictionOutput:
    texts: list[str]
    y_true: list[str]
    y_pred: list[str]


def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> PredictionOutput:
    model.eval()
    texts: list[str] = []
    true_ids: list[int] = []
    predicted_ids: list[int] = []
    with torch.no_grad():
        for input_ids, lengths, labels, batch_texts in loader:
            input_ids = input_ids.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            logits = model(input_ids, lengths)
            predictions = logits.argmax(dim=-1)
            texts.extend(batch_texts)
            true_ids.extend(labels.tolist())
            predicted_ids.extend(predictions.cpu().tolist())
    return PredictionOutput(
        texts=texts,
        y_true=[ID_TO_LABEL[value] for value in true_ids],
        y_pred=[ID_TO_LABEL[value] for value in predicted_ids],
    )


def _validation_scores(output: PredictionOutput) -> tuple[float, float, float]:
    accuracy = accuracy_score(output.y_true, output.y_pred)
    macro_f1 = f1_score(
        output.y_true,
        output.y_pred,
        labels=list(LABELS),
        average="macro",
        zero_division=0,
    )
    weighted_f1 = f1_score(
        output.y_true,
        output.y_pred,
        labels=list(LABELS),
        average="weighted",
        zero_division=0,
    )
    return float(accuracy), float(macro_f1), float(weighted_f1)


def train_with_early_stopping(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
    checkpoint_path: str | Path,
    seed: int,
    epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
    gradient_clip_norm: float,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    checkpoint = Path(checkpoint_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    best_weighted_f1 = -1.0
    best_epoch = 0
    bad_epochs = 0
    history: list[dict[str, float | int]] = []
    started = perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        examples = 0
        for input_ids, lengths, labels, _ in train_loader:
            input_ids = input_ids.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids, lengths)
            if not torch.isfinite(logits).all():
                raise FloatingPointError("Model produced NaN or Inf logits.")
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("Training loss became NaN or Inf.")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), gradient_clip_norm
            )
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError("Gradient norm became NaN or Inf.")
            optimizer.step()

            batch_size = labels.shape[0]
            total_loss += float(loss.item()) * batch_size
            examples += batch_size

        validation_output = predict(model, validation_loader, device)
        validation_accuracy, validation_macro_f1, validation_weighted_f1 = _validation_scores(
            validation_output
        )
        row = {
            "epoch": epoch,
            "train_loss": total_loss / examples,
            "validation_accuracy": validation_accuracy,
            "validation_macro_f1": validation_macro_f1,
            "validation_weighted_f1": validation_weighted_f1,
        }
        history.append(row)
        print(
            f"epoch={epoch:02d} train_loss={row['train_loss']:.4f} "
            f"val_accuracy={validation_accuracy:.4f} "
            f"val_macro_f1={validation_macro_f1:.4f} "
            f"val_weighted_f1={validation_weighted_f1:.4f}",
            flush=True,
        )

        if validation_weighted_f1 > best_weighted_f1:
            best_weighted_f1 = validation_weighted_f1
            best_epoch = epoch
            bad_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "seed": seed,
                    "validation_weighted_f1": validation_weighted_f1,
                },
                checkpoint,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    if not checkpoint.is_file():
        raise RuntimeError("No best checkpoint was saved.")

    summary: dict[str, float | int] = {
        "best_epoch": best_epoch,
        "best_validation_weighted_f1": best_weighted_f1,
        "epochs_completed": len(history),
        "training_seconds": perf_counter() - started,
    }
    return pd.DataFrame(history), summary


def load_model_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    device: torch.device,
) -> dict:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    model.eval()
    return payload

