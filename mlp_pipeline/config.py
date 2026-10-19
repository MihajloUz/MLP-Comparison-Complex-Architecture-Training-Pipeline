"""Central configuration for the training pipeline."""

from __future__ import annotations

from dataclasses import dataclass

PROBLEM_TYPES = (None, "classification", "regression")


@dataclass
class Config:
    """Configuration controlling data handling and training.

    All defaults are chosen so that a full run stays CPU friendly.
    """

    seed: int = 42
    test_size: float = 0.20
    val_size: float = 0.15
    max_rows: int = 20000
    epochs: int = 60
    batch_size: int = 32
    patience: int = 8
    learning_rate: float = 1e-3
    output_dir: str = "outputs"
    target_column: str | None = None
    # Optional explicit problem type; when None it is inferred from the target.
    problem_type: str | None = None

    def __post_init__(self) -> None:
        if not 0.0 < self.test_size < 1.0:
            raise ValueError("test_size must be in (0, 1)")
        if not 0.0 <= self.val_size < 1.0 - self.test_size:
            raise ValueError("val_size must leave at least test_size of the data")
        if self.problem_type not in PROBLEM_TYPES:
            raise ValueError("problem_type must be None, 'classification' or 'regression'")
