"""Dataset loading, target detection, splitting and leakage-free preprocessing."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from .config import Config

TARGET_CANDIDATES = ("target", "label", "class", "y", "outcome", "output", "response")
# A numeric target with more distinct values than this is treated as regression
# unless the caller supplies an explicit problem type.
MAX_NUMERIC_CLASSES = 10


@dataclass
class Dataset:
    """Container holding transformed feature matrices and encoded targets."""

    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    feature_names: list = field(default_factory=list)
    task: str = "classification"
    n_classes: int = 0
    class_names: list = field(default_factory=list)
    target: str = ""
    raw_target_test: np.ndarray | None = None

    @property
    def n_features(self) -> int:
        return int(self.X_train.shape[1])

    @property
    def n_train(self) -> int:
        return int(self.X_train.shape[0])

    @property
    def n_val(self) -> int:
        return int(self.X_val.shape[0])

    @property
    def n_test(self) -> int:
        return int(self.X_test.shape[0])

    @property
    def n_total(self) -> int:
        return self.n_train + self.n_val + self.n_test


def read_csv(path: str) -> pd.DataFrame:
    """Read a CSV file and normalise column names."""
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.dropna(axis=1, how="all")
    if df.shape[1] < 2:
        raise ValueError("CSV must contain at least one feature column and a target column.")
    return df


def detect_target(df: pd.DataFrame, explicit: str | None = None) -> str:
    """Pick the target column: explicit name, a common name, or the last column."""
    if explicit:
        if explicit not in df.columns:
            raise ValueError(f"Target column '{explicit}' not found in {list(df.columns)}")
        return explicit
    lookup = {c.lower(): c for c in df.columns}
    for candidate in TARGET_CANDIDATES:
        if candidate in lookup:
            return lookup[candidate]
    return df.columns[-1]


def infer_task(y: pd.Series, explicit: str | None = None) -> str:
    """Classify the prediction problem from the target column.

    ``explicit`` (``'classification'``/``'regression'``) always wins.  Otherwise
    a boolean/object target is classification and a numeric target is
    classification when it has at most ``MAX_NUMERIC_CLASSES`` distinct values,
    regression above that threshold.  Use an explicit ``problem_type`` for
    integer-encoded targets with many classes to avoid the regression fallback.
    """
    if explicit in ("classification", "regression"):
        return explicit
    if pd.api.types.is_bool_dtype(y):
        return "classification"
    if pd.api.types.is_numeric_dtype(y):
        if y.dropna().nunique() > MAX_NUMERIC_CLASSES:
            return "regression"
        return "classification"
    return "classification"


def _subsample(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    if len(df) <= cfg.max_rows:
        return df
    return df.sample(n=cfg.max_rows, random_state=cfg.seed).reset_index(drop=True)


def _split(X, y, task, test_size, seed):
    """Split while trying to stratify for classification."""
    stratify = y if task == "classification" else None
    counts = pd.Series(y).value_counts() if stratify is not None else None
    if counts is not None and counts.min() < 2:
        stratify = None
    try:
        return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=stratify)
    except ValueError:
        return train_test_split(X, y, test_size=test_size, random_state=seed)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build a ColumnTransformer that imputes and scales/one-hot encodes."""
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    if not num_cols and not cat_cols:
        raise ValueError("No feature columns available after dropping the target.")

    transformers = []
    if num_cols:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                num_cols,
            )
        )
    if cat_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                cat_cols,
            )
        )
    return ColumnTransformer(transformers, remainder="drop")


def prepare_data(df: pd.DataFrame, target: str, cfg: Config) -> Dataset:
    """Split then fit the preprocessing pipeline on the training split only.

    Fitting the imputer/scaler/encoder on the training rows exclusively avoids
    any information leakage from validation or test data.
    """
    df = df.dropna(subset=[target]).copy()
    if df.empty:
        raise ValueError("No rows left after dropping missing target values.")
    df = _subsample(df, cfg)

    y_raw = df[target]
    X = df.drop(columns=[target])
    task = infer_task(y_raw, cfg.problem_type)

    label_encoder = None
    if task == "classification":
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_raw.astype(str))
        class_names = [str(c) for c in label_encoder.classes_]
        n_classes = len(class_names)
    else:
        y = y_raw.astype(float).to_numpy()
        class_names = []
        n_classes = 0

    test_size = cfg.test_size
    val_ratio = cfg.val_size / (1.0 - test_size)

    X_tr, X_te, y_tr, y_te = _split(X, y, task, test_size, cfg.seed)
    X_tr, X_va, y_tr, y_va = _split(X_tr, y_tr, task, val_ratio, cfg.seed + 1)

    # Fit preprocessing on the training rows only (no leakage into val/test).
    preprocessor = build_preprocessor(X_tr)
    preprocessor.fit(X_tr)

    X_train = np.asarray(preprocessor.transform(X_tr), dtype="float32")
    X_val = np.asarray(preprocessor.transform(X_va), dtype="float32")
    X_test = np.asarray(preprocessor.transform(X_te), dtype="float32")

    y_train = np.asarray(y_tr)
    y_val = np.asarray(y_va)
    y_test = np.asarray(y_te)

    raw_target_test = None
    if label_encoder is not None:
        raw_target_test = label_encoder.inverse_transform(y_test.astype(int))

    return Dataset(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        feature_names=list(preprocessor.get_feature_names_out()),
        task=task,
        n_classes=n_classes,
        class_names=class_names,
        target=target,
        raw_target_test=raw_target_test,
    )
