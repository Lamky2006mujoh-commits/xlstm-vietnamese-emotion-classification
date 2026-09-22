# Vietnamese Emotion Classification with xLSTM

A controlled comparison of recurrent, convolutional, native mLSTM/xLSTM, and classical TF-IDF models for seven-label Vietnamese social-media emotion classification on UIT-VSMEC.

The repository is designed for reproducible, isolated experiments. It keeps dataset validation, preprocessing, model configuration, training artifacts, predictions, and evaluation metadata together so that results can be checked after training.

> **Status:** The repository contains the experiment code and configuration files. Generated datasets, checkpoints, and results are intentionally excluded from Git. Do not treat historical or locally generated scores as repository results unless the corresponding artifacts are available and verified.

## Task

Each Vietnamese social-media sentence is assigned one of seven labels:

| Label | Description |
|---|---|
| Anger | Anger or frustration |
| Disgust | Disgust or strong aversion |
| Enjoyment | Happiness or enjoyment |
| Fear | Fear or anxiety |
| Other | Emotion outside the six categories, or no expressed emotion |
| Sadness | Sadness or disappointment |
| Surprise | Surprise or astonishment |

The canonical task identifier is `uit_vsmec_7label`.

## Implemented models

| Model/configuration | Implementation |
|---|---|
| LSTM | PyTorch unidirectional LSTM with packed sequences |
| BiLSTM | PyTorch bidirectional LSTM with packed sequences |
| CNN | Masked text CNN with odd-kernel convolutions and padding-aware max pooling |
| Native xLSTM | Native PyTorch **mLSTM-only** classifier with causal depthwise convolution, stabilized parallel mLSTM memory retrieval, residual blocks, and last-valid-token pooling |
| TF-IDF + Linear SVM | Word and character-level TF-IDF features combined with scikit-learn `LinearSVC` |

The native xLSTM implementation is adapted from the [NX-AI xLSTM repository](https://github.com/NX-AI/xlstm). The project deliberately isolates the mLSTM path and does not implement sLSTM, a mixed mLSTM/sLSTM stack, or xLSTM-Large. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

The native backend uses ordinary PyTorch tensor operations. It does **not** require the upstream package, Triton, `nvcc`, Conda, or custom CUDA extension compilation.

## Dataset

The code expects local UIT-VSMEC CSV files at:

```text
data/raw/uit_vsmec/
├── train.csv
├── validation.csv
└── test.csv
```

Every file must contain exactly these columns:

```text
Sentence,Emotion
```

The default strict validation contract requires:

| Split | Rows |
|---|---:|
| Training | 5,548 |
| Validation | 686 |
| Test | 693 |
| **Total** | **6,927** |

The validator also checks missing values, the complete seven-label set, duplicate and conflicting texts, split overlaps, and SHA-256 hashes. Raw data is ignored by Git and must be obtained separately from the canonical UIT-VSMEC source.

Dataset reference: [Emotion Recognition for Vietnamese Social Media Text](https://arxiv.org/abs/1911.09339).

## Preprocessing and data flow

The default configuration uses `preprocessing: light`:

- Collapse repeated whitespace.
- Lowercase text.
- Normalize the predefined social-media abbreviations in `src/preprocessing.py`.
- Build the vocabulary from normalized training text only.
- Keep tokens occurring at least twice; reserve `<pad>` and `<unk>`.
- Truncate neural inputs to 80 whitespace-separated tokens.
- Use 128-dimensional trainable embeddings in the final neural configurations.
- Pad batches dynamically and preserve sequence lengths for packed-sequence and last-valid-token models.

The alternative `raw` preprocessing mode only normalizes whitespace and does not lowercase or expand abbreviations.

## Configuration files

Configurations are YAML files under `configs/`.

- `final_lstm.yaml`, `final_bilstm.yaml`, and `final_cnn.yaml` are the controlled final neural configurations.
- `final_classical_svm.yaml` defines the validation-tuned word/character TF-IDF + Linear SVM baseline.
- `pilot_xlstm_*.yaml` files define native xLSTM pilots.
- `xlstm.yaml` is a compatibility alias for a rejected two-block validation pilot; it is not a frozen final configuration and must not be used for test evaluation.
- `smoke_xlstm_native.yaml` is intended for a fast native-backend smoke test.

The final neural configs use 20 training epochs, early stopping patience of 4, batch size 64, Adam-style learning-rate/weight-decay settings, gradient clipping at 1.0, and seeds `42, 123, 2026, 7, 999`. Configuration files are the source of truth; inspect them before launching an experiment.

## Installation

The verified environment uses Python 3.11 and PyTorch 2.11.0 with CUDA 12.6. The commands below target Windows PowerShell.

Create a virtual environment and install the matching PyTorch build:

```powershell
py -3.11 -m venv .venv-xlstm
.\.venv-xlstm\Scripts\python.exe -m pip install --upgrade pip
.\.venv-xlstm\Scripts\python.exe -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu126
.\.venv-xlstm\Scripts\python.exe -m pip install -r requirements.txt
```

For exact pinned versions, see [`requirements-lock.txt`](requirements-lock.txt). The lock file records the environment verified by the project, including Python 3.11.9 and the tested GPU environment.

Check the native backend before a full run:

```powershell
.\.venv-xlstm\Scripts\python.exe -B scripts\check_environment.py
```

Use `--strict-gpu` when CUDA is required:

```powershell
.\.venv-xlstm\Scripts\python.exe -B scripts\check_environment.py --strict-gpu
```

## Running experiments

Run commands from the repository root. Each run writes to a new directory and refuses to overwrite non-empty output directories.

### Neural models

Train one configured model using all seeds declared in its YAML file:

```powershell
.\.venv-xlstm\Scripts\python.exe -B scripts\train.py --config configs\final_lstm.yaml --device cuda --evaluate-test
.\.venv-xlstm\Scripts\python.exe -B scripts\train.py --config configs\final_bilstm.yaml --device cuda --evaluate-test
.\.venv-xlstm\Scripts\python.exe -B scripts\train.py --config configs\final_cnn.yaml --device cuda --evaluate-test
```

The `--evaluate-test` flag is intentional: it evaluates the already-inspected test split only after the configuration has been frozen. Omit it for validation-only runs.

Run the native xLSTM pilot without performing a test evaluation:

```powershell
.\.venv-xlstm\Scripts\python.exe -B scripts\train.py --config configs\pilot_xlstm_1block_128.yaml --device cuda
```

A single seed or an alternate output location can be supplied explicitly:

```powershell
.\.venv-xlstm\Scripts\python.exe -B scripts\train.py \
  --config configs\final_lstm.yaml \
  --seeds 42 \
  --device cuda \
  --output-root results\lstm_seed42
```

On Windows PowerShell, use one line or PowerShell's backtick continuation character instead of the Unix `\` continuation shown above.

### Classical baseline

The SVM script fits each candidate `C` value on the training split, selects the best value by validation weighted-F1, saves validation predictions, and optionally evaluates the test split:

```powershell
.\.venv-xlstm\Scripts\python.exe -B scripts\train_classical.py --config configs\final_classical_svm.yaml --evaluate-test
```

The default search is `C ∈ {0.25, 0.5, 1.0, 2.0, 4.0}`, with word n-grams from 1–3 and character-within-word n-grams from 1–7.

### Data validation

Validate the local dataset independently:

```powershell
.\.venv-xlstm\Scripts\python.exe -B scripts\validate_data.py
```

## Outputs and reproducibility

By default, runs are written below `results/<experiment_group>/`. Each run contains artifacts such as:

- `checkpoint_best.pt` for neural models or `checkpoint_best.joblib` for SVM.
- `config_resolved.yaml`.
- `dataset_manifest.json`, including split hashes and integrity checks.
- `environment.json`.
- Training history and validation metrics.
- Prediction CSV files.
- `run_manifest.json`, including model identity, seed, parameter count, timing, and test-evaluation status.

Summarize completed test-evaluated runs and reconstruct metrics from prediction files:

```powershell
.\.venv-xlstm\Scripts\python.exe -B scripts\summarize_runs.py --experiment-dir results\final_v1
```

The summarizer rejects incomplete runs, wrong task IDs, duplicate model/seed pairs, incorrect test-row counts, and mismatches between stored and reconstructed metrics. Results are ignored by Git; share them explicitly if they are needed for review.

## Tests

Run the unit-test suite with:

```powershell
.\.venv-xlstm\Scripts\python.exe -B -m unittest discover -s tests -v
```

The tests cover preprocessing, vocabulary construction, dataset contracts, model forward/backward behavior, sequence-length handling, metric reconstruction, and experiment artifact validation.

## Repository structure

```text
configs/       YAML model and experiment configurations
src/           Data contracts, preprocessing, models, training, and evaluation
scripts/       Training, validation, environment, and summarization entry points
tests/         Automated tests
data/          Local datasets; excluded from Git
results/       Generated run artifacts; excluded from Git
references/    Research reference material, when present
```

## Limitations and interpretation

- Scores depend on the exact local split and preprocessing configuration; do not compare them with published UIT-VSMEC scores without matching the protocol.
- The test split may have been inspected during project development. The code supports configuration freeze before test evaluation but cannot undo prior inspection.
- The dataset can contain duplicate texts, label conflicts, and cross-split text overlap; these are reported by the manifest rather than silently removed.
- The xLSTM implementation is a native, mLSTM-only sentence classifier, not a reproduction of every xLSTM architecture or the official package runtime.
- Neural models use randomly initialized trainable embeddings; pretrained Vietnamese embeddings are not part of the controlled implementation.
- Pilot configurations and smoke-test outputs must not be presented as final benchmark results.

## References

- [xLSTM: Extended Long Short-Term Memory](https://arxiv.org/abs/2405.04517)
- [Official NX-AI xLSTM implementation](https://github.com/NX-AI/xlstm)
- [Emotion Recognition for Vietnamese Social Media Text](https://arxiv.org/abs/1911.09339)
