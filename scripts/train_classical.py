from __future__ import annotations

import argparse
import platform
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from time import perf_counter

import _bootstrap  # noqa: F401
import joblib
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from src.configuration import load_config, serializable_config
from src.data import load_prepared_data
from src.evaluation import classification_metrics, save_json, save_predictions
from src.experiment import resolve_project_root


def build_model(c_value: float, config: dict) -> Pipeline:
    return Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        (
                            "word_tfidf",
                            TfidfVectorizer(
                                analyzer="word",
                                ngram_range=tuple(config["word_ngram_range"]),
                                min_df=int(config["word_min_df"]),
                                max_features=int(config["word_max_features"]),
                            ),
                        ),
                        (
                            "char_tfidf",
                            TfidfVectorizer(
                                analyzer="char_wb",
                                ngram_range=tuple(config["char_ngram_range"]),
                                min_df=int(config["char_min_df"]),
                                max_features=int(config["char_max_features"]),
                            ),
                        ),
                    ]
                ),
            ),
            (
                "classifier",
                LinearSVC(
                    C=c_value,
                    loss=str(config["loss"]),
                    class_weight=config["class_weight"],
                    max_iter=5_000,
                    random_state=42,
                ),
            ),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the frozen seven-label SVM baseline.")
    parser.add_argument("--config", type=Path, default=Path("configs/final_classical_svm.yaml"))
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--experiment-group", default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--evaluate-test", action="store_true")
    args = parser.parse_args()

    root = resolve_project_root(args.project_root or args.config.resolve().parent)
    config = load_config(args.config)
    classical = config.get("classical")
    if not isinstance(classical, dict):
        raise ValueError("Configuration requires a classical section.")
    group = args.experiment_group or config["experiment_group"]
    output_base = args.output_root.resolve() if args.output_root else root / "results" / group
    run_dir = output_base / "best_svm_word_char_tfidf_seed42"
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing run: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    data_config = config["data"]
    raw_dir = Path(data_config["raw_dir"])
    if not raw_dir.is_absolute():
        raw_dir = root / raw_dir
    data = load_prepared_data(
        raw_dir,
        preprocessing=data_config["preprocessing"],
        max_len=int(data_config["max_len"]),
        min_token_frequency=int(data_config["min_token_frequency"]),
    )

    tuning_rows = []
    best_model = None
    best_c = None
    best_score = -1.0
    started = perf_counter()
    for c_value in classical["c_values"]:
        model = build_model(float(c_value), classical)
        model.fit(data.train["text"], data.train["label"])
        predictions = model.predict(data.validation["text"])
        metrics = classification_metrics(data.validation["label"], predictions)
        tuning_rows.append(
            {"C": float(c_value), **{key: metrics[key] for key in ("accuracy", "macro_f1", "weighted_f1")}}
        )
        if metrics["weighted_f1"] > best_score:
            best_score = float(metrics["weighted_f1"])
            best_c = float(c_value)
            best_model = model
    if best_model is None:
        raise RuntimeError("No SVM candidate completed.")

    pd.DataFrame(tuning_rows).to_csv(
        run_dir / "validation_tuning.csv", index=False, encoding="utf-8-sig"
    )
    validation_predictions = best_model.predict(data.validation["text"])
    validation_metrics = classification_metrics(data.validation["label"], validation_predictions)
    save_predictions(
        run_dir / "validation_predictions.csv",
        data.validation["text"],
        data.validation["label"],
        validation_predictions,
    )
    save_json(validation_metrics, run_dir / "metrics_validation.json")

    test_metrics = None
    if args.evaluate_test:
        test_predictions = best_model.predict(data.test["text"])
        test_metrics = classification_metrics(data.test["label"], test_predictions)
        save_predictions(
            run_dir / "test_predictions.csv",
            data.test["text"],
            data.test["label"],
            test_predictions,
        )
        save_json(test_metrics, run_dir / "metrics_test.json")
    joblib.dump(best_model, run_dir / "checkpoint_best.joblib")

    resolved = {
        **serializable_config(config),
        "classical": {**classical, "selected_c": best_c},
    }
    (run_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump(resolved, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    save_json(data.manifest, run_dir / "dataset_manifest.json")
    save_json(
        {
            "python": sys.version,
            "platform": platform.platform(),
            "scikit_learn": metadata.version("scikit-learn"),
        },
        run_dir / "environment.json",
    )
    coefficient_count = int(best_model.named_steps["classifier"].coef_.size)
    manifest = {
        "run_id": run_dir.name,
        "experiment_group_id": group,
        "task_id": config["task_id"],
        "model_name": "best_svm_word_char_tfidf",
        "model_backend": "scikit_learn",
        "seed": 42,
        "status": "complete",
        "test_evaluated": bool(args.evaluate_test),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "parameter_count": coefficient_count,
        "best_epoch": 0,
        "best_validation_weighted_f1": best_score,
        "training_seconds": perf_counter() - started,
        "peak_gpu_memory_bytes": None,
        "artifacts": {
            "checkpoint": "checkpoint_best.joblib",
            "config": "config_resolved.yaml",
            "dataset_manifest": "dataset_manifest.json",
            "environment": "environment.json",
            "history": "validation_tuning.csv",
            "validation_predictions": "validation_predictions.csv",
            "test_predictions": "test_predictions.csv" if args.evaluate_test else None,
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
    save_json(manifest, run_dir / "run_manifest.json")
    print(f"Selected C={best_c} using validation weighted-F1={best_score:.4f}")
    print(f"Completed run: {run_dir}")


if __name__ == "__main__":
    main()
