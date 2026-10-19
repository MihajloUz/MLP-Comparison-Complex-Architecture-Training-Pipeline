"""Unit and end-to-end tests for the MLP comparison pipeline.

These tests never touch the network and never require TensorFlow: the deep
learning path is exercised only through the ``keras_available`` guard, and every
end-to-end run forces ``use_keras=False`` so the suite stays fast and runnable in
a plain CPU environment.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from mlp_pipeline import Config, run
from mlp_pipeline.data import (
    build_preprocessor,
    detect_target,
    infer_task,
    prepare_data,
    read_csv,
)
from mlp_pipeline.evaluate import (
    confusion_matrix_for,
    evaluate_predictions,
    is_integer_like,
    primary_metric,
    to_labels,
    to_probabilities,
)

# --------------------------------------------------------------------------- #
# Target detection & task inference
# --------------------------------------------------------------------------- #


def test_detect_target_prefers_common_name():
    df = pd.DataFrame({"a": [1, 2], "label": ["x", "y"], "z": [0.0, 1.0]})
    assert detect_target(df) == "label"


def test_detect_target_falls_back_to_last_column():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    assert detect_target(df) == "b"


def test_detect_target_explicit_and_missing():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    assert detect_target(df, explicit="a") == "a"
    with pytest.raises(ValueError):
        detect_target(df, explicit="missing")


def test_infer_task_regression_and_classification():
    assert infer_task(pd.Series([0.1, 0.2, 0.3, 1.5])) == "regression"
    assert infer_task(pd.Series([0, 1, 2, 1, 0])) == "classification"
    assert infer_task(pd.Series([True, False, True])) == "classification"
    assert infer_task(pd.Series(["cat", "dog"])) == "classification"


def test_infer_task_explicit_override():
    # Many distinct integers would default to regression, explicit wins.
    many = pd.Series(np.arange(50))
    assert infer_task(many) == "regression"
    assert infer_task(many, explicit="classification") == "classification"


def test_config_validation():
    with pytest.raises(ValueError):
        Config(test_size=1.5)
    with pytest.raises(ValueError):
        Config(problem_type="nonsense")


# --------------------------------------------------------------------------- #
# Data preparation: leakage-free splitting
# --------------------------------------------------------------------------- #


def test_read_csv_requires_feature_and_target(tmp_path):
    path = tmp_path / "one_col.csv"
    pd.DataFrame({"only": [1, 2, 3]}).to_csv(path, index=False)
    with pytest.raises(ValueError):
        read_csv(str(path))


def test_prepare_data_shapes_and_encoder_is_train_fitted(classification_csv):
    df = read_csv(classification_csv)
    target = detect_target(df)
    cfg = Config(seed=0, max_rows=1000)
    data = prepare_data(df, target, cfg)

    assert data.task == "classification"
    assert data.n_classes == 3
    assert len(data.class_names) == 3
    assert data.n_train + data.n_val + data.n_test == len(df)
    assert data.X_train.shape[1] == data.X_test.shape[1]
    assert data.y_train.max() < data.n_classes

    # StandardScaler is fitted on the training split only, so the scaled numeric
    # columns of the training matrix must have ~zero mean and ~unit std. If the
    # scaler had seen the test rows this would not hold as tightly.
    n_numeric = df.drop(columns=[target]).select_dtypes(include=[np.number]).shape[1]
    assert n_numeric > 0
    numeric_block = data.X_train[:, :n_numeric]
    assert np.allclose(numeric_block.mean(axis=0), 0.0, atol=1e-4)
    assert np.allclose(numeric_block.std(axis=0), 1.0, atol=1e-2)


def test_prepare_data_is_reproducible(classification_csv):
    df = read_csv(classification_csv)
    target = detect_target(df)
    first = prepare_data(df, target, Config(seed=7))
    second = prepare_data(df, target, Config(seed=7))
    assert np.array_equal(first.X_train, second.X_train)
    assert np.array_equal(first.y_test, second.y_test)


def test_prepare_data_regression(regression_csv):
    df = read_csv(regression_csv)
    data = prepare_data(df, detect_target(df), Config(seed=1))
    assert data.task == "regression"
    assert data.n_classes == 0
    assert data.y_train.dtype.kind == "f"


def test_build_preprocessor_handles_categoricals():
    X = pd.DataFrame({"num": [1.0, 2.0, 3.0], "cat": ["a", "b", "a"]})
    pre = build_preprocessor(X).fit(X)
    out = pre.transform(X)
    assert out.shape[0] == 3
    assert out.shape[1] >= 2


# --------------------------------------------------------------------------- #
# Evaluation helpers on synthetic arrays
# --------------------------------------------------------------------------- #


def test_is_integer_like():
    assert is_integer_like(np.array([0, 1, 2]))
    assert is_integer_like(np.array([0.0, 1.0, 2.0]))
    assert not is_integer_like(np.array([0.1, 0.9, 0.4]))


def test_to_probabilities_binary_scores_1d():
    scores = np.array([0.2, 0.8, 0.5])
    probs = to_probabilities(scores, n_classes=2)
    assert probs.shape == (3, 2)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert np.allclose(probs[:, 1], scores)


def test_to_probabilities_binary_scores_2d_column():
    scores = np.array([[0.1], [0.9]])
    probs = to_probabilities(scores, n_classes=2)
    assert probs.shape == (2, 2)
    assert np.allclose(probs[:, 1], [0.1, 0.9])


def test_to_probabilities_multiclass_softmax():
    softmax = np.array([[0.1, 0.2, 0.7], [0.6, 0.3, 0.1]])
    probs = to_probabilities(softmax, n_classes=3)
    assert np.allclose(probs, softmax)
    assert np.array_equal(to_labels(softmax, "classification", 3), [2, 0])


def test_to_probabilities_returns_none_for_labels():
    labels = np.array([0, 1, 1, 0])
    assert to_probabilities(labels, n_classes=2) is None
    assert np.array_equal(to_labels(labels, "classification", 2), labels)


def test_to_labels_regression_passthrough():
    values = np.array([[0.5], [1.5]])
    assert np.allclose(to_labels(values, "regression", 0), [0.5, 1.5])


def test_evaluate_predictions_classification_uses_probabilities():
    y_true = np.array([0, 1, 0, 1, 1, 0])
    probs = np.array([0.1, 0.9, 0.2, 0.8, 0.6, 0.4])
    metrics = evaluate_predictions("classification", 2, y_true, probs)
    assert metrics["accuracy"] == pytest.approx(1.0)
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert metrics["f1"] == pytest.approx(1.0)


def test_evaluate_predictions_multiclass_roc_auc():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 3, size=40)
    softmax = rng.random((40, 3))
    softmax /= softmax.sum(axis=1, keepdims=True)
    metrics = evaluate_predictions("classification", 3, y_true, softmax)
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert set(metrics) >= {"accuracy", "precision", "recall", "f1", "roc_auc"}


def test_evaluate_predictions_label_only_has_nan_roc_auc():
    y_true = np.array([0, 1, 0, 1, 1, 0])
    labels = np.array([0, 1, 0, 1, 1, 0])
    metrics = evaluate_predictions("classification", 2, y_true, labels)
    assert np.isnan(metrics["roc_auc"])
    assert metrics["accuracy"] == pytest.approx(1.0)


def test_evaluate_predictions_single_class_is_safe():
    y_true = np.zeros(4, dtype=int)
    metrics = evaluate_predictions("classification", 1, y_true, np.zeros(4, dtype=int))
    assert np.isnan(metrics["roc_auc"])


def test_evaluate_predictions_regression_metrics():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8])
    metrics = evaluate_predictions("regression", 0, y_true, y_pred)
    assert metrics["rmse"] >= 0.0
    assert metrics["r2"] == pytest.approx(1.0 - 0.1 / np.var(y_true), rel=0.2)


def test_confusion_matrix_for_shape():
    y_true = np.array([0, 1, 2, 2, 1, 0])
    scores = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.1, 0.2, 0.7],
                       [0.2, 0.2, 0.6], [0.3, 0.5, 0.2], [0.6, 0.3, 0.1]])
    cm = confusion_matrix_for(3, y_true, scores)
    assert cm.shape == (3, 3)
    assert cm.sum() == len(y_true)


def test_primary_metric():
    assert primary_metric("classification") == "accuracy"
    assert primary_metric("regression") == "r2"


# --------------------------------------------------------------------------- #
# End-to-end runs (sklearn-only path, no TensorFlow needed)
# --------------------------------------------------------------------------- #


def _run(csv_path, tmp_path, **cfg_kwargs):
    cfg = Config(
        seed=3,
        epochs=15,
        max_rows=400,
        output_dir=str(tmp_path / "out"),
        **cfg_kwargs,
    )
    return run(csv_path, cfg, use_keras=False, verbose=False)


def test_end_to_end_classification(tmp_path, classification_csv):
    bundle = _run(classification_csv, tmp_path)
    results = bundle["results"]
    assert not results.empty
    assert "sklearn_mlp" in set(results["model"])
    for column in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        assert column in results.columns

    out = tmp_path / "out"
    assert (out / "comparison_table.csv").is_file()
    assert (out / "confusion_matrix.png").is_file()
    assert (out / "model_comparison.png").is_file()
    assert (out / "target_distribution.png").is_file()
    # Confusion matrix is written for classification only.
    assert bundle["best_model"] in set(results["model"])


def test_end_to_end_regression(tmp_path, regression_csv):
    bundle = _run(regression_csv, tmp_path)
    results = bundle["results"]
    assert not results.empty
    for column in ("rmse", "mae", "r2"):
        assert column in results.columns
    out = tmp_path / "out"
    assert (out / "comparison_table.csv").is_file()
    assert not (out / "confusion_matrix.png").exists()


def test_run_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        run(str(tmp_path / "nope.csv"), Config(output_dir=str(tmp_path / "o")),
            use_keras=False, verbose=False)
