# MLP Comparison & Complex-Architecture Training Pipeline

A reproducible, CPU-only Python pipeline that trains several multilayer neural
network variants on a **user-supplied CSV**, compares them fairly against the
scikit-learn reference MLP, trains one **complex architecture**, and produces a
comparison table, plots and a final summary.

Nothing is downloaded and no sample data is invented: the dataset is always
chosen by you and read from disk.

## Features

- **2–3 deep MLP variants** plus the scikit-learn MLP reference:
  - `mlp_shallow` – single `(64,)` ReLU layer
  - `mlp_deep` – `(128, 64, 32)` with Dropout
  - `mlp_complex` – **complex architecture**: BatchNorm + residual block +
    Dropout + L2 regularisation (stacked hidden blocks)
  - `sklearn_mlp` – `(64, 32)` reference from scikit-learn
- Automatic **target detection** and **task inference** (classification vs
  regression).
- **Leakage-free** preprocessing: imputer / scaler / one-hot encoder are fitted
  on the training split only, with stratified train/val/test splits.
- **Fair comparison**: every model is scored through one shared evaluation
  path; classification models are compared on probability outputs so that
  ROC-AUC and thresholded metrics use a common footing.
- Reproducible: a single seed controls NumPy, Python and the DL backend.
- Headless: matplotlib uses the `Agg` backend and only writes PNG files.

## Installation

```bash
pip install -r requirements.txt
```

Deep learning is **optional** and uses one backend only — Keras/TensorFlow or
PyTorch, never both. The pipeline runs with just the scikit-learn reference if
no backend is installed. To enable the Keras variants:

```bash
pip install "tensorflow>=2.12"
```

and uncomment it in `requirements.txt`.

## Usage

Running the pipeline prompts for the CSV path at runtime:

```bash
python train.py
```

```text
Enter path to CSV: /path/to/your/dataset.csv
```

For scripted use you can skip the prompt and pass options directly:

```bash
python train.py --csv /path/to/dataset.csv --target label --output-dir outputs
python train.py --csv data.csv --no-keras        # sklearn MLP only
python train.py --csv data.csv --epochs 100 --seed 7
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--csv` | *(prompted)* | Path to the input CSV |
| `--target` | auto-detected | Target column name |
| `--output-dir` | `outputs` | Directory for artefacts |
| `--epochs` | `60` | Max epochs for deep models |
| `--batch-size` | `32` | Mini-batch size |
| `--seed` | `42` | Global random seed |
| `--max-rows` | `20000` | Subsample larger CSVs |
| `--no-keras` | off | Train only the sklearn MLP |

The CSV must contain at least one feature column and a target column. `train.py`
does **not** modify your data; everything is loaded into memory.

## Outputs

All artefacts are written to the output directory (default `outputs/`):

- `comparison_table.csv` – metrics for every model, sorted by the primary test
  metric (`accuracy` for classification, `r2` for regression).
- `training_curves.png` – loss / metric curves per model.
- `model_comparison.png` – bar chart of the primary metric.
- `confusion_matrix.png` – confusion matrix of the best model (classification).
- `target_distribution.png` – class / target distribution of the train split.

A textual summary (rows used, task, best model, improvement over the
scikit-learn baseline and artefact location) is printed at the end.

## Tests

```bash
make test          # or: python -m pytest -q tests
```

The suite covers target detection, task inference, leakage-free splitting,
the probability / ROC-AUC evaluation helpers on synthetic arrays, and an
end-to-end run on locally generated synthetic CSVs. It forces `use_keras=False`,
so it needs no deep-learning backend. TensorFlow/Keras is imported lazily, so
the package stays importable without it.

If the environment is missing numpy/pandas/scikit-learn/matplotlib/pytest,
`make test` prints a clear `SKIPPED` message instead of an ImportError; install
the requirements and re-run to execute the real suite.

## Project layout

```text
train.py                  CLI entry point (prompts for the CSV path)
mlp_pipeline/
  config.py               Configuration dataclass
  data.py                 Loading, target detection, splitting, preprocessing
  models.py               Keras variants, complex architecture, sklearn MLP
  evaluate.py             Shared metric computation
  plots.py                Headless (Agg) plotting
  pipeline.py             End-to-end orchestration
  seed.py                 Random-seed handling
tests/                    Pytest suite + synthetic-data fixtures
scripts/run_tests.py      `make test` bootstrap
```

## Reproducibility notes

- Set `--seed` (default `42`) to reproduce a run; NumPy, Python and the DL
  backend are all seeded from it.
- Preprocessing statistics come from the training split only — no leakage.
- Early stopping, a modest epoch budget and CSV subsampling keep CPU runtimes
  reasonable.
