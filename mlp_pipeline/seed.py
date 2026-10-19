"""Random-seed handling for reproducible runs."""

from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int) -> None:
    """Seed every library used by the pipeline that is already importable."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Keep TensorFlow quiet (headless, no GUI, no noisy C++ logs).
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

    random.seed(seed)
    np.random.seed(seed)

    try:  # scikit-learn uses numpy's global RNG but seed it explicitly if present
        from sklearn.utils import check_random_state  # noqa: F401

        check_random_state(seed)
    except Exception:  # pragma: no cover - sklearn always present in practice
        pass


def seed_keras(seed: int) -> None:
    """Seed a (already imported) TensorFlow/Keras backend for determinism."""
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:  # pragma: no cover - not fatal on unsupported ops
        pass
