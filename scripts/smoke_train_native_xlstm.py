from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd
import torch

from src.constants import LABELS
from src.data import load_prepared_data, make_loader
from src.experiment import resolve_project_root
from src.models import NativeXLSTMClassifier
from src.training import set_seed


def accuracy_on_loader(model, loader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for input_ids, lengths, labels, _ in loader:
            logits = model(input_ids.to(device), lengths.to(device))
            predictions = logits.argmax(dim=-1).cpu()
            correct += int((predictions == labels).sum())
            total += labels.shape[0]
    return correct / total


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Overfit a tiny balanced subset to validate the native xLSTM training path."
    )
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--target-accuracy", type=float, default=0.90)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    root = resolve_project_root(args.project_root)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable.")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    set_seed(42)
    data = load_prepared_data(
        root / "data" / "raw" / "uit_vsmec",
        preprocessing="light",
        max_len=80,
        min_token_frequency=2,
    )
    tiny = pd.concat(
        [
            group.sample(n=min(16, len(group)), random_state=42)
            for _, group in data.train.groupby("label", sort=True)
        ],
        ignore_index=True,
    )
    loader = make_loader(
        tiny,
        vocabulary=data.vocabulary,
        max_len=data.max_len,
        batch_size=32,
        shuffle=True,
        seed=42,
        num_workers=0,
    )
    evaluation_loader = make_loader(
        tiny,
        vocabulary=data.vocabulary,
        max_len=data.max_len,
        batch_size=32,
        shuffle=False,
        seed=42,
        num_workers=0,
    )

    model_kwargs = {
        "vocab_size": len(data.vocabulary),
        "num_classes": len(LABELS),
        "embedding_dim": 64,
        "dropout": 0.0,
        "context_length": 80,
        "num_blocks": 1,
        "num_heads": 4,
        "proj_factor": 2.0,
        "qkv_proj_blocksize": 4,
        "conv1d_kernel_size": 4,
    }
    model = NativeXLSTMClassifier(**model_kwargs).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.0)
    criterion = torch.nn.CrossEntropyLoss()

    best_accuracy = 0.0
    completed_epoch = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for input_ids, lengths, labels, _ in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids.to(device), lengths.to(device))
            loss = criterion(logits, labels.to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError("Tiny-overfit loss became non-finite.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        best_accuracy = max(best_accuracy, accuracy_on_loader(model, evaluation_loader, device))
        completed_epoch = epoch
        if epoch == 1 or epoch % 5 == 0 or best_accuracy >= args.target_accuracy:
            print(f"epoch={epoch:03d} tiny_train_accuracy={best_accuracy:.4f}", flush=True)
        if best_accuracy >= args.target_accuracy:
            break

    if best_accuracy < args.target_accuracy:
        raise RuntimeError(
            f"Tiny subset did not reach target accuracy {args.target_accuracy:.3f}; "
            f"best={best_accuracy:.3f}."
        )

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty smoke directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "native_xlstm_tiny_overfit.pt"
    torch.save({"model_state_dict": model.state_dict(), "model_kwargs": model_kwargs}, checkpoint)

    first_batch = next(iter(evaluation_loader))
    input_ids, lengths, _, _ = first_batch
    model.eval()
    with torch.no_grad():
        original_logits = model(input_ids.to(device), lengths.to(device)).cpu()
    reloaded = NativeXLSTMClassifier(**model_kwargs).to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    reloaded.load_state_dict(payload["model_state_dict"])
    reloaded.eval()
    with torch.no_grad():
        reloaded_logits = reloaded(input_ids.to(device), lengths.to(device)).cpu()
    torch.testing.assert_close(original_logits, reloaded_logits, rtol=0, atol=0)

    result = {
        "status": "ok",
        "backend": "native_pytorch_mlstm",
        "device": str(device),
        "samples": int(len(tiny)),
        "epochs_completed": completed_epoch,
        "tiny_train_accuracy": best_accuracy,
        "target_accuracy": args.target_accuracy,
        "checkpoint_reload_exact": True,
        "checkpoint": str(checkpoint),
    }
    (output_dir / "smoke_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
