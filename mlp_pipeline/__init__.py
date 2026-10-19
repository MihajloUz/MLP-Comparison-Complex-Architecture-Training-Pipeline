"""Reproducible MLP comparison and training pipeline.

This package trains several multilayer neural network variants plus the
scikit-learn reference MLP, evaluates them fairly on a held-out test split and
produces a comparison table, plots and a textual summary.
"""

from .config import Config
from .pipeline import run

__all__ = ["Config", "run"]
__version__ = "1.0.0"
