from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from src.data import validate_raw_dataset
from src.experiment import resolve_project_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the canonical UIT-VSMEC files.")
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    root = resolve_project_root(args.project_root)
    manifest = validate_raw_dataset(root / "data" / "raw" / "uit_vsmec")
    output = args.output or root / "results" / "validation" / "dataset_manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print(f"Saved dataset manifest: {output}")


if __name__ == "__main__":
    main()

