"""End-to-end orchestration: data -> models -> evaluation -> table/plots/summary."""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from .config import Config
from .data import detect_target, prepare_data, read_csv
from .evaluate import confusion_matrix_for, evaluate_predictions, metrics_columns, primary_metric
from .models import (
    KERAS_VARIANTS,
    SKLEARN_MODEL_NAME,
    keras_available,
    train_keras,
    train_sklearn,
)
from .plots import (
    plot_confusion,
    plot_model_comparison,
    plot_target_distribution,
    plot_training_curves,
)
from .seed import set_global_seed

KERAS_MODEL_ORDER = ["mlp_shallow", "mlp_deep", "mlp_complex"]


def _train_one(framework: str, name: str, data, cfg: Config):
    if framework == "keras":
        return train_keras(name, data, cfg)
    return train_sklearn(data, cfg)


def run(
    csv_path: str,
    cfg: Config | None = None,
    use_keras: bool = True,
    verbose: bool = True,
) -> dict:
    """Execute the full pipeline and return a result bundle."""
    cfg = cfg or Config()
    set_global_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)

    def log(*args):
        if verbose:
            print(*args, flush=True)

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    log("=" * 70)
    log("MLP comparison & complex-architecture training pipeline")
    log("=" * 70)
    raw_df = read_csv(csv_path)
    target = detect_target(raw_df, cfg.target_column)
    log(f"Loaded {len(raw_df)} rows / {raw_df.shape[1]} columns from {csv_path}")
    log(f"Target column: '{target}'")

    data = prepare_data(raw_df, target, cfg)
    log(f"Task: {data.task} | features: {data.n_features} | classes: {data.n_classes or '-'}")
    log(f"Split -> train={data.n_train} val={data.n_val} test={data.n_test}")

    plot_target_distribution(
        data.y_train,
        data.task,
        os.path.join(cfg.output_dir, "target_distribution.png"),
        class_names=data.class_names,
    )

    frameworks: list[tuple[str, str]] = []
    if use_keras:
        if keras_available():
            frameworks += [("keras", name) for name in KERAS_MODEL_ORDER]
        else:
            log("! TensorFlow/Keras not importable - skipping deep-learning variants.")
    frameworks.append(("sklearn", SKLEARN_MODEL_NAME))

    metric_cols = metrics_columns(data.task)
    rows = []
    histories: dict[str, dict] = {}
    raw_predictions: dict[str, np.ndarray] = {}

    for framework, name in frameworks:
        log(f"\n>> Training [{framework}] {name} ...")
        # scikit-learn's MLP emits ConvergenceWarning when the iteration budget
        # is exhausted; the fitted model is still usable, so silence only that
        # specific warning instead of every warning raised while training.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            model = _train_one(framework, name, data, cfg)
        # Probabilities for classification, continuous values for regression.
        raw = np.asarray(model.predict(data.X_test))
        raw_predictions[name] = raw
        metrics = evaluate_predictions(data.task, data.n_classes, data.y_test, raw)

        epochs_run = len(model.history.get("loss", [])) if model.history else 0
        row = {
            "model": name,
            "framework": framework,
            "description": model.description,
            "params": model.n_params,
            "train_seconds": round(model.train_seconds, 3),
            "epochs": epochs_run,
        }
        row.update({k: round(v, 5) for k, v in metrics.items()})
        rows.append(row)
        histories[name] = model.history or {}
        log(
            f"   params={model.n_params:,} time={model.train_seconds:.2f}s "
            f"epochs={epochs_run} " + " ".join(f"{k}={metrics[k]:.4f}" for k in metric_cols)
        )

    results = pd.DataFrame(rows)
    primary = primary_metric(data.task)
    results = results.sort_values(primary, ascending=False).reset_index(drop=True)

    table_path = os.path.join(cfg.output_dir, "comparison_table.csv")
    results.to_csv(table_path, index=False)
    log("\n" + "=" * 70)
    log(f"Comparison table (sorted by test {primary}, higher is better)")
    log("=" * 70)
    log(results.to_string(index=False))
    log(f"\nSaved comparison table -> {table_path}")

    plot_training_curves(histories, data.task, os.path.join(cfg.output_dir, "training_curves.png"))
    plot_model_comparison(results, primary, os.path.join(cfg.output_dir, "model_comparison.png"))

    best_name = str(results.iloc[0]["model"])
    best_metrics = results.iloc[0].to_dict()
    if data.task == "classification":
        cm = confusion_matrix_for(data.n_classes, data.y_test, raw_predictions[best_name])
        plot_confusion(cm, data.class_names, os.path.join(cfg.output_dir, "confusion_matrix.png"))

    baseline = results[results["model"] == SKLEARN_MODEL_NAME]
    log("\n" + "=" * 70)
    log("FINAL SUMMARY")
    log("=" * 70)
    log(f"Dataset rows used     : {data.n_total}")
    log(f"Task / target         : {data.task} / '{target}'")
    log(f"Input features        : {data.n_features}")
    log(f"Models compared       : {', '.join(results['model'])}")
    log(
        f"Best model ({primary:>8}) : {best_name} "
        f"({best_metrics[primary]:.4f}, framework={best_metrics['framework']})"
    )
    if not baseline.empty and baseline.iloc[0]["model"] != best_name:
        delta = best_metrics[primary] - baseline.iloc[0][primary]
        log(f"Improvement over sklearn MLP baseline : {delta:+.4f} {primary}")
    elif not baseline.empty:
        log("The sklearn MLP reference remained the strongest model.")
    log(f"Artefacts written to   : {os.path.abspath(cfg.output_dir)}")

    return {
        "results": results,
        "best_model": best_name,
        "dataset": data,
        "histories": histories,
        "predictions": raw_predictions,
        "config": cfg,
    }


__all__ = ["run", "KERAS_MODEL_ORDER", "KERAS_VARIANTS"]
