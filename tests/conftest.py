"""Shared pytest fixtures: tiny synthetic CSVs generated locally (no downloads)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression


@pytest.fixture(scope="session")
def classification_csv(tmp_path_factory):
    X, y = make_classification(
        n_samples=300,
        n_features=8,
        n_informative=5,
        n_classes=3,
        n_clusters_per_class=1,
        random_state=0,
    )
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    df["category"] = np.where(np.arange(len(df)) % 2 == 0, "a", "b")
    df["target"] = y
    path = tmp_path_factory.mktemp("data") / "classification.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture(scope="session")
def regression_csv(tmp_path_factory):
    X, y = make_regression(n_samples=250, n_features=6, noise=0.1, random_state=0)
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    df["target"] = y
    path = tmp_path_factory.mktemp("data") / "regression.csv"
    df.to_csv(path, index=False)
    return str(path)
