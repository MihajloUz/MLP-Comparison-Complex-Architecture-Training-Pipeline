"""Model definitions: Keras MLP variants, a complex architecture and the sklearn MLP.

scikit-learn is imported eagerly (it is a hard dependency of the pipeline) while
TensorFlow/Keras is imported lazily, so the whole package - and the fast,
TensorFlow-free test path - stays importable in a CPU-only environment.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Config
from .seed import seed_keras

# Human readable description of every Keras variant.
KERAS_VARIANTS = {
    "mlp_shallow": "(64,) Dense-ReLU",
    "mlp_deep": "(128, 64, 32) Dense-ReLU + Dropout(0.2)",
    "mlp_complex": "residual + BatchNorm + Dropout + L2 (stacked hidden blocks)",
}

SKLEARN_MODEL_NAME = "sklearn_mlp"


@dataclass
class TrainedModel:
    """Bundle of a fitted model plus the metadata needed for reporting.

    ``predict`` returns probabilities for classification and continuous values
    for regression, so the evaluator treats every framework identically.
    """

    name: str
    framework: str
    model: object
    n_params: int
    task: str = "classification"
    n_classes: int = 2
    history: dict | None = None
    train_seconds: float = 0.0
    description: str = ""

    def raw_output(self, X: np.ndarray) -> np.ndarray:
        """Framework-native prediction (scores for Keras, values for sklearn)."""
        if self.framework == "keras":
            return np.asarray(self.model.predict(X, verbose=0))
        return np.asarray(self.model.predict(X))

    def predict_proba(self, X: np.ndarray) -> np.ndarray | None:
        """``(n_samples, n_classes)`` probabilities, or ``None`` if unavailable."""
        if self.framework == "keras":
            raw = self.raw_output(X)
            if raw.ndim == 1:
                raw = raw.reshape(-1, 1)
            if raw.shape[1] == 1 and self.n_classes == 2:
                positive = raw[:, 0]
                return np.column_stack([1.0 - positive, positive])
            return raw
        if hasattr(self.model, "predict_proba"):
            return np.asarray(self.model.predict_proba(X))
        return None

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Probabilities for classification, continuous predictions for regression."""
        if self.task == "regression":
            return self.raw_output(X).reshape(-1)
        probabilities = self.predict_proba(X)
        if probabilities is not None:
            return probabilities
        return self.raw_output(X)


def keras_available() -> bool:
    """True when TensorFlow/Keras can be imported in this environment."""
    try:
        import tensorflow  # noqa: F401

        return True
    except Exception:
        return False


def _dense_block(keras, x, units, dropout, l2, batchnorm):
    layers = keras.layers
    reg = keras.regularizers.l2(l2) if l2 else None
    x = layers.Dense(units, kernel_regularizer=reg)(x)
    if batchnorm:
        x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    if dropout:
        x = layers.Dropout(dropout)(x)
    return x


def _output_head(keras, x, task, n_classes, l2):
    layers = keras.layers
    reg = keras.regularizers.l2(l2) if l2 else None
    if task == "regression":
        return layers.Dense(1, activation="linear", kernel_regularizer=reg)(x)
    if n_classes == 2:
        return layers.Dense(1, activation="sigmoid", kernel_regularizer=reg)(x)
    return layers.Dense(n_classes, activation="softmax", kernel_regularizer=reg)(x)


def build_keras_model(name: str, input_dim: int, task: str, n_classes: int, cfg: Config):
    """Create and compile a Keras model for the requested variant."""
    import tensorflow as tf
    from tensorflow import keras

    keras.utils.set_random_seed(cfg.seed)
    layers = keras.layers
    inputs = keras.Input(shape=(input_dim,), name="features")

    if name == "mlp_shallow":
        x = _dense_block(keras, inputs, 64, 0.0, 0.0, False)
    elif name == "mlp_deep":
        x = _dense_block(keras, inputs, 128, 0.2, 0.0, False)
        x = _dense_block(keras, x, 64, 0.2, 0.0, False)
        x = _dense_block(keras, x, 32, 0.0, 0.0, False)
    elif name == "mlp_complex":
        x = _dense_block(keras, inputs, 128, 0.0, 1e-4, True)
        # Residual block: projection + non-linear branch, summed then activated.
        skip = layers.Dense(128, kernel_regularizer=keras.regularizers.l2(1e-4))(x)
        skip = layers.BatchNormalization()(skip)
        branch = layers.Dense(128, kernel_regularizer=keras.regularizers.l2(1e-4))(x)
        branch = layers.BatchNormalization()(branch)
        branch = layers.Activation("relu")(branch)
        branch = layers.Dropout(0.3)(branch)
        x = layers.Add()([skip, branch])
        x = layers.Activation("relu")(x)
        x = _dense_block(keras, x, 64, 0.3, 1e-4, True)
        x = _dense_block(keras, x, 32, 0.1, 1e-4, False)
    else:
        raise ValueError(f"Unknown Keras variant: {name}")

    outputs = _output_head(keras, x, task, n_classes, l2=0.0 if name != "mlp_complex" else 1e-4)
    model = keras.Model(inputs, outputs, name=name)
    _compile(keras, tf, model, task, n_classes, cfg.learning_rate)
    return model


def _compile(keras, tf, model, task, n_classes, lr) -> None:
    optimizer = keras.optimizers.Adam(learning_rate=lr)
    if task == "regression":
        model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
    elif n_classes == 2:
        model.compile(
            optimizer=optimizer,
            loss="binary_crossentropy",
            metrics=[
                tf.keras.metrics.BinaryAccuracy(name="accuracy"),
                tf.keras.metrics.AUC(name="auc"),
            ],
        )
    else:
        model.compile(
            optimizer=optimizer,
            loss="sparse_categorical_crossentropy",
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )


def train_keras(name: str, data, cfg: Config) -> TrainedModel:
    """Build, train (with early stopping) and return a Keras model."""
    import time

    from tensorflow import keras

    seed_keras(cfg.seed)
    model = build_keras_model(name, data.n_features, data.task, data.n_classes, cfg)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=cfg.patience,
            restore_best_weights=True,
            min_delta=1e-4,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=max(2, cfg.patience // 2),
            min_lr=1e-5,
        ),
    ]

    start = time.perf_counter()
    history = model.fit(
        data.X_train,
        data.y_train,
        validation_data=(data.X_val, data.y_val),
        epochs=cfg.epochs,
        batch_size=min(cfg.batch_size, max(1, data.n_train)),
        callbacks=callbacks,
        verbose=0,
        shuffle=True,
    )
    elapsed = time.perf_counter() - start

    return TrainedModel(
        name=name,
        framework="keras",
        model=model,
        n_params=int(model.count_params()),
        task=data.task,
        n_classes=data.n_classes,
        history={k: [float(v) for v in vals] for k, vals in history.history.items()},
        train_seconds=elapsed,
        description=KERAS_VARIANTS.get(name, ""),
    )


def train_sklearn(data, cfg: Config) -> TrainedModel:
    """Train the scikit-learn reference MLP with a comparable budget."""
    import time

    from sklearn.neural_network import MLPClassifier, MLPRegressor

    common = dict(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=min(cfg.batch_size, max(1, data.n_train)),
        learning_rate_init=cfg.learning_rate,
        max_iter=max(200, cfg.epochs * 5),
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=cfg.patience,
        random_state=cfg.seed,
        tol=1e-4,
    )

    model = MLPClassifier(**common) if data.task == "classification" else MLPRegressor(**common)

    start = time.perf_counter()
    model.fit(data.X_train, data.y_train)
    elapsed = time.perf_counter() - start

    n_params = int(sum(c.size for c in model.coefs_) + sum(b.size for b in model.intercepts_))
    history = {"loss": [float(v) for v in model.loss_curve_]}

    return TrainedModel(
        name=SKLEARN_MODEL_NAME,
        framework="sklearn",
        model=model,
        n_params=n_params,
        task=data.task,
        n_classes=data.n_classes,
        history=history,
        train_seconds=elapsed,
        description="(64, 32) Dense-ReLU, Adam, early stopping",
    )
