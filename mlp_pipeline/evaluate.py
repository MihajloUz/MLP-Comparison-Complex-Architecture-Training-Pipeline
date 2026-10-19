"""Fair, shared metric computation for classification and regression models.

Every model - the scikit-learn reference MLP and the deep-learning variants -
is scored through exactly the same code path here.  Classification models are
compared on *probability* outputs (``predict_proba`` for scikit-learn, the
network head for the deep models) so that thresholded metrics and ROC-AUC are
computed on a common footing.  Label-only outputs are still handled gracefully.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

CLASSIFICATION_METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]
REGRESSION_METRICS = ["rmse", "mae", "r2"]


def metrics_columns(task: str) -> list[str]:
    """Column names produced by :func:`evaluate_predictions` for a task."""
    return list(CLASSIFICATION_METRICS if task == "classification" else REGRESSION_METRICS)


def primary_metric(task: str) -> str:
    """Metric used to rank models (higher is better for both tasks)."""
    return "accuracy" if task == "classification" else "r2"


def is_integer_like(values: np.ndarray) -> bool:
    """True when an array holds whole numbers (i.e. looks like class labels)."""
    values = np.asarray(values)
    if values.dtype.kind in "iu":
        return True
    if values.dtype.kind == "f":
        finite = values[np.isfinite(values)]
        return finite.size > 0 and bool(np.allclose(finite, np.round(finite)))
    return False


def to_probabilities(raw_pred: np.ndarray, n_classes: int) -> np.ndarray | None:
    """Return an ``(n_samples, n_classes)`` probability matrix when possible.

    Returns ``None`` when the input can only be interpreted as hard labels.
    """
    raw = np.asarray(raw_pred, dtype="float64")
    if raw.ndim == 2 and raw.shape[1] == n_classes and n_classes > 0:
        return raw
    if raw.ndim == 2 and raw.shape[1] == 1 and n_classes == 2:
        positive = raw[:, 0]
        return np.column_stack([1.0 - positive, positive])
    if raw.ndim == 1 and n_classes == 2 and not is_integer_like(raw):
        positive = raw
        return np.column_stack([1.0 - positive, positive])
    return None


def to_labels(raw_pred: np.ndarray, task: str, n_classes: int) -> np.ndarray:
    """Convert any model output into hard class labels / regression values."""
    raw = np.asarray(raw_pred)
    if task == "regression":
        return raw.reshape(-1).astype(float)
    probabilities = to_probabilities(raw, n_classes)
    if probabilities is not None:
        return probabilities.argmax(axis=1).astype(int)
    if raw.ndim == 1:
        return raw.astype(int)
    return raw.argmax(axis=1).astype(int)


def _safe_roc_auc(y_true: np.ndarray, raw_pred: np.ndarray, n_classes: int) -> float:
    """ROC-AUC that never raises, even for label-only or degenerate outputs."""
    if n_classes < 2:
        return float("nan")
    probabilities = to_probabilities(raw_pred, n_classes)
    try:
        if probabilities is not None:
            if n_classes == 2:
                return float(roc_auc_score(y_true, probabilities[:, 1]))
            return float(
                roc_auc_score(y_true, probabilities, multi_class="ovr", average="weighted")
            )
        raw = np.asarray(raw_pred, dtype="float64").reshape(-1)
        if n_classes == 2 and not is_integer_like(raw):
            return float(roc_auc_score(y_true, raw))
        return float("nan")
    except ValueError:
        return float("nan")


def evaluate_predictions(
    task: str, n_classes: int, y_true: np.ndarray, raw_pred: np.ndarray
) -> dict:
    """Return the metric dictionary for a single fitted model."""
    y_true = np.asarray(y_true)
    raw_pred = np.asarray(raw_pred)
    metrics: dict[str, float] = {}

    if task == "regression":
        y_pred = raw_pred.reshape(-1).astype(float)
        metrics["rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
        metrics["r2"] = float(r2_score(y_true, y_pred))
        return metrics

    y_pred = to_labels(raw_pred, task, n_classes)
    average = "binary" if n_classes == 2 else "weighted"
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["precision"] = float(precision_score(y_true, y_pred, average=average, zero_division=0))
    metrics["recall"] = float(recall_score(y_true, y_pred, average=average, zero_division=0))
    metrics["f1"] = float(f1_score(y_true, y_pred, average=average, zero_division=0))
    metrics["roc_auc"] = _safe_roc_auc(y_true, raw_pred, n_classes)
    return metrics


def confusion_matrix_for(n_classes: int, y_true, raw_pred) -> np.ndarray:
    """Confusion matrix using the same label decoding as the metrics."""
    from sklearn.metrics import confusion_matrix

    y_true = np.asarray(y_true)
    y_pred = to_labels(raw_pred, "classification", n_classes)
    labels = list(range(n_classes)) if n_classes else None
    return confusion_matrix(y_true, y_pred, labels=labels)
