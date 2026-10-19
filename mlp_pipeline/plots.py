"""Headless (Agg) plotting helpers producing PNG artefacts."""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")  # must be set before pyplot is imported anywhere

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def _save(fig, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_training_curves(histories: dict, task: str, path: str) -> None:
    """Plot loss and the task metric for every model across epochs."""
    histories = {k: v for k, v in histories.items() if v}
    if not histories:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    metric_name = "accuracy" if task == "classification" else "mae"

    for name, hist in histories.items():
        if "loss" in hist:
            axes[0].plot(hist["loss"], label=f"{name} (train)")
        if "val_loss" in hist:
            axes[0].plot(hist["val_loss"], linestyle="--", label=f"{name} (val)")

    metric_key = next((k for k in ("accuracy", "mae") if k == metric_name), None)
    for name, hist in histories.items():
        key = metric_key if metric_key in hist else None
        if key is None:
            key = next((k for k in hist if k not in ("loss", "val_loss")), None)
        if key:
            axes[1].plot(hist[key], label=f"{name} (train)")
            if f"val_{key}" in hist:
                axes[1].plot(hist[f"val_{key}"], linestyle="--", label=f"{name} (val)")

    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[1].set_title(f"Metric ({metric_name})")
    axes[1].set_xlabel("epoch")
    for ax in axes:
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    _save(fig, path)


def plot_model_comparison(results: pd.DataFrame, metric: str, path: str) -> None:
    """Bar chart of the primary test metric for every model."""
    if results.empty or metric not in results.columns:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ordered = results.sort_values(metric, ascending=False)
    colors = ["#4C72B0" if fw == "keras" else "#DD8452" for fw in ordered["framework"]]
    ax.bar(ordered["model"], ordered[metric], color=colors)
    ax.set_title(f"Test {metric} by model")
    ax.set_ylabel(metric)
    ax.tick_params(axis="x", rotation=30)
    ax.grid(alpha=0.3, axis="y")
    _save(fig, path)


def plot_confusion(cm: np.ndarray, class_names, path: str) -> None:
    """Render a pre-computed confusion matrix for the best classification model.

    ``cm`` is the ``(n_classes, n_classes)`` matrix returned by
    :func:`mlp_pipeline.evaluate.confusion_matrix_for`; ``class_names`` maps the
    row/column indices back to human readable labels.
    """
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Confusion matrix (best model)")
    ticks = range(cm.shape[0]) if not class_names else range(len(class_names))
    labels = list(class_names) if class_names else [str(i) for i in ticks]
    ax.set_xticks(list(ticks))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticks(list(ticks))
    ax.set_yticklabels(labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax)
    _save(fig, path)


def plot_target_distribution(y, task: str, path: str, class_names=None) -> None:
    """Plot the target distribution to make class balance visible.

    For classification the raw encoded integers are mapped back to the decoded
    ``class_names`` so the axis matches the comparison table and confusion
    matrix.
    """
    fig, ax = plt.subplots(figsize=(6, 4))
    values = np.asarray(y)
    if task == "classification":
        uniq, counts = np.unique(values, return_counts=True)
        if class_names:
            names = [
                str(class_names[int(u)]) if 0 <= int(u) < len(class_names) else str(u)
                for u in uniq
            ]
        else:
            names = [str(u) for u in uniq]
        ax.bar(names, counts, color="#55A868")
        ax.set_ylabel("count")
        ax.set_title("Class distribution (train split)")
    else:
        ax.hist(values.astype(float), bins=30, color="#55A868")
        ax.set_ylabel("count")
        ax.set_title("Target distribution (train split)")
    ax.grid(alpha=0.3, axis="y")
    _save(fig, path)
