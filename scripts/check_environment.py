from __future__ import annotations

import argparse
import json
import shutil
import sys
from importlib import metadata

import _bootstrap  # noqa: F401
import torch

from src.models import NativeXLSTMClassifier


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the compiler-free native mLSTM/xLSTM environment."
    )
    parser.add_argument("--strict-gpu", action="store_true", help="Fail unless CUDA is available.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = {
        "python": sys.version,
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "native_backend": "native_pytorch_mlstm",
        "native_backend_requires_nvcc": False,
        "native_backend_requires_conda": False,
        "nvcc_available": shutil.which("nvcc") is not None,
        "conda_available": shutil.which("conda") is not None,
        "packages": {
            name: package_version(name)
            for name in ("numpy", "pandas", "scikit-learn", "PyYAML", "torch")
        },
    }
    if args.strict_gpu and device.type != "cuda":
        raise SystemExit("CUDA is required by --strict-gpu but is unavailable.")

    model = NativeXLSTMClassifier(
        vocab_size=32,
        num_classes=7,
        embedding_dim=32,
        dropout=0.0,
        context_length=8,
        num_blocks=1,
        num_heads=4,
        proj_factor=2.0,
        qkv_proj_blocksize=4,
        conv1d_kernel_size=4,
    ).to(device)
    inputs = torch.randint(2, 32, (3, 8), device=device)
    lengths = torch.tensor([8, 5, 3], device=device)
    inputs[1, 5:] = 0
    inputs[2, 3:] = 0
    labels = torch.tensor([0, 1, 2], device=device)
    logits = model(inputs, lengths)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()
    report["native_forward_shape"] = list(logits.shape)
    report["native_loss_finite"] = bool(torch.isfinite(loss))
    report["native_backward_success"] = any(
        parameter.grad is not None for parameter in model.stack.parameters()
    )
    report["status"] = "ok"
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

