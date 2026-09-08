<<<<<<< HEAD
# Vietnamese Emotion Classification with xLSTM

A comparative study of **mLSTM-based xLSTM for Vietnamese social-media emotion classification**, evaluated against LSTM, BiLSTM, CNN, and TF-IDF + Linear SVM baselines.

The project uses **UIT-VSMEC** and emphasizes reproducible experiments, validation-based model selection, evaluation across multiple random seeds, and transparent reporting of both positive and negative results.

## Overview

Emotion classification identifies the emotion expressed in a piece of text. This project takes a Vietnamese social-media sentence as input and predicts one of seven labels:

| Label | Description |
|---|---|
| Anger | Anger or frustration |
| Disgust | Disgust or strong aversion |
| Enjoyment | Happiness or enjoyment |
| Fear | Fear or anxiety |
| Other | Emotion outside the six categories, or no expressed emotion |
| Sadness | Sadness or disappointment |
| Surprise | Surprise or astonishment |

The central research question is:

> How does an mLSTM-only xLSTM classifier compare with conventional recurrent models for Vietnamese emotion classification under a shared data-processing and evaluation framework?

This is a focused classification study. It does not train a large language model or claim production readiness.

## Project Scope

The main experiment includes:

- Seven-label emotion classification on the project's local UIT-VSMEC splits.
- Shared text preprocessing and training-only vocabulary construction.
- Random, trainable embeddings for all neural models.
- A bounded hyperparameter search using validation data.
- Five-seed final evaluation for each neural architecture.
- Accuracy, macro-F1, weighted-F1, and per-class metrics.
- Paired error comparisons and bootstrap uncertainty estimates.
- Parameter counts, training time, inference throughput, and GPU memory measurements.
- Saved checkpoints, predictions, configurations, and reproducibility metadata.

Pretrained Vietnamese embeddings, the six-label experiment without `Other`, and evaluation on ViGoEmotions are separate extensions outside the completed primary experiment.

## Models

| Model | Implementation |
|---|---|
| LSTM | Packed-sequence, unidirectional sentence classifier |
| BiLSTM | Packed-sequence, bidirectional sentence classifier |
| CNN | Text CNN with padding-aware pooling |
| xLSTM | Native PyTorch mLSTM-only classifier with last-valid-token pooling |
| SVM | Word and character TF-IDF features with Linear SVM |

The xLSTM implementation adapts components from the official [NX-AI xLSTM repository](https://github.com/NX-AI/xlstm).

It uses **mLSTM blocks only**. It does not implement sLSTM, a mixed mLSTM/sLSTM stack, or xLSTM-Large. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for attribution.

## Dataset

The project uses the following local UIT-VSMEC splits:

| Split | Examples |
|---|---:|
| Training | 5,548 |
| Validation | 686 |
| Test | 693 |
| **Total** | **6,927** |

Expected layout:

```text
data/raw/uit_vsmec/
├── train.csv
├── validation.csv
└── test.csv
```

Each CSV must contain these columns:

```text
Sentence,Emotion
```

Raw data is excluded from version control. Obtain the canonical project splits separately before running the experiment.

The complete-study runner verifies their exact file hashes. Matching the number of rows alone is insufficient; re-exporting a CSV can also change its hash.

Dataset reference: [Emotion Recognition for Vietnamese Social Media Text](https://arxiv.org/abs/1911.09339).

## Experimental Protocol

### Preprocessing

- Normalize whitespace.
- Lowercase text and normalize a predefined set of social-media abbreviations.
- Build the vocabulary from training text only.
- Retain tokens occurring at least twice in the training split.
- Limit neural inputs to 80 whitespace-separated tokens.
- Use 128-dimensional random, trainable embeddings.

### Model Selection

Each neural architecture evaluates the same two training presets using validation seeds `42` and `123`.

The selected preset maximizes **mean validation weighted-F1**. Ties follow the preset order declared in the study configuration.

The selected configurations are saved before final test evaluation. Learning rate and regularization may differ between selected models, while the search budget and selection procedure remain shared.

### Final Evaluation

Each neural model is evaluated using:

```text
42, 123, 2026, 7, 999
```

The SVM selects its regularization parameter using validation data and has one final evaluated run.

Reported metrics are reconstructed from saved predictions. The workflow checks dataset identity, prediction-row alignment, stored scores, and completeness of the required model/seed matrix.

## Results

Results below are from the completed local seven-label experiment.

| Model | Mean weighted-F1 | Standard deviation |
|---|---:|---:|
| TF-IDF + Linear SVM | 58.79% | — |
| BiLSTM | 51.52% | 2.01 percentage points |
| LSTM | 50.61% | 0.54 percentage points |
| CNN | 50.06% | 1.36 percentage points |
| mLSTM-only xLSTM | 35.68% | 2.07 percentage points |

Neural results use five seeds. Standard deviations are sample standard deviations across those seeds. SVM seed variability was not estimated.

**The tested xLSTM configuration did not outperform the baselines.** This is a negative result for the evaluated configuration and protocol, not a general conclusion about all xLSTM architectures.

The generated report contains the complete metric tables, per-class results, confusion matrices, paired comparisons, uncertainty estimates, and resource measurements.

## Installation

The verified environment uses Python 3.11 and CUDA-enabled PyTorch. The commands below target Windows PowerShell.

Create a virtual environment:

```powershell
py -3.11 -m venv .venv-xlstm
```

Install PyTorch and the project dependencies:

```powershell
.\.venv-xlstm\Scripts\python.exe -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu126

.\.venv-xlstm\Scripts\python.exe -m pip install -r requirements.txt
```

The native mLSTM implementation uses standard PyTorch operations and does not require custom CUDA extension compilation.

## Running the Study

Run commands from the project root.

### Complete experiment

```powershell
.\.venv-xlstm\Scripts\python.exe -B scripts\run_complete_study.py --device cuda
```

This command:

1. Runs validation-only selection.
2. Saves the selected configurations.
3. Runs all five final neural seeds.
4. Trains and evaluates the validation-tuned SVM.
5. Reconstructs metrics and generates the results report.

### Resume an interrupted study

```powershell
.\.venv-xlstm\Scripts\python.exe -B scripts\run_complete_study.py --device cuda --resume
```

Completed runs are verified before reuse. Existing evidence is preserved. Changes to the protocol, source, dataset, or recorded environment require a new output directory.

### Rebuild the report without training

```powershell
.\.venv-xlstm\Scripts\python.exe -B scripts\run_complete_study.py --device cuda --analysis-only
```

Use the same recorded study environment.

### Run a workflow smoke test

```powershell
.\.venv-xlstm\Scripts\python.exe -B scripts\run_complete_study.py --device cuda --smoke
```

Smoke-test outputs are stored separately and must not be reported as research results.

## Outputs

The default full-study output directory is:

```text
results/uit_vsmec_complete_v1/
```

Open this file in a browser:

```text
results/uit_vsmec_complete_v1/research_report.html
```

Key outputs include:

| File | Contents |
|---|---|
| `research_report.html` | Complete research report |
| `stability_summary.csv` | Aggregate metrics and resource measurements |
| `aggregate_runs.csv` | Results for individual final runs |
| `selection_scores.csv` | Validation-only selection results |
| `per_class_summary.csv` | Mean per-class metrics |
| `confusion_by_seed.csv` | Confusion counts for individual runs |
| `paired_errors.csv` | Paired correctness comparisons |
| `paired_bootstrap.csv` | Conditional bootstrap intervals |
| `research_summary.json` | Machine-readable conclusion |
| `study_protocol.json` | Protocol and reproducibility metadata |
| `selected_configs/` | Configurations selected before test evaluation |

Individual run directories also contain checkpoints, prediction files, metrics, and integrity checksums.

Results are ignored by Git by default. Include selected result artifacts explicitly when sharing the project.

## Tests

```powershell
.\.venv-xlstm\Scripts\python.exe -B -m unittest discover -s tests -v
```

Tests cover preprocessing, vocabulary construction, model behavior, metric reconstruction, validation-only selection, bootstrap calculations, and rejection of inconsistent evidence.

## Repository Structure

```text
configs/       Model configurations and complete-study settings
src/           Data processing, models, training, and analysis
scripts/       Experiment entry points
tests/         Automated tests
data/          Local datasets; excluded from Git
results/       Generated experiments and reports; excluded from Git
notebooks/     Earlier exploratory and baseline experiments
references/    Research reference material
```

Use `scripts/run_complete_study.py` for the current complete experiment. Historical notebook results belong to their original experimental protocols.

## Limitations

- The local split's per-class counts differ from the published UIT-VSMEC benchmark, so published scores are not directly comparable.
- The test set was inspected during earlier project development. The current protocol prevents new test-driven selection but cannot undo that history.
- The dataset contains duplicates, one training-label conflict, and a train-validation text overlap.
- The hyperparameter search is deliberately limited, and model parameter counts are not identical.
- Bootstrap intervals are exploratory and conditional on the observed training seeds. They do not include hyperparameter-selection uncertainty.
- Only a one-block, mLSTM-only xLSTM configuration is evaluated in the completed study.
- The neural models use random embeddings rather than pretrained Vietnamese representations.

## References

- [xLSTM: Extended Long Short-Term Memory](https://arxiv.org/abs/2405.04517)
- [Official NX-AI xLSTM implementation](https://github.com/NX-AI/xlstm)
- [Emotion Recognition for Vietnamese Social Media Text](https://arxiv.org/abs/1911.09339)
=======
# xlstm-vietnamese-emotion-classification
>>>>>>> origin/main
