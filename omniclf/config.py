"""Run configuration: the single source of truth for an experiment.

A :class:`RunConfig` is serialisable to JSON and is written verbatim into every
run directory, so any reported number can be reproduced by re-running with the
stored configuration file.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any


@dataclass
class RunConfig:
    """Everything that defines one experiment.

    Attributes
    ----------
    dataset:
        Path to the input CSV. The target is taken from ``target_column``
        (name or index; ``-1`` means the last column).
    scaler, selector, cv, models:
        Registry keys resolved by :mod:`omniclf.preprocessing`,
        :mod:`omniclf.feature_selection`, :mod:`omniclf.cross_val` and
        :mod:`omniclf.models`.
    n_features:
        Number of features requested from the selector (ignored by selectors
        that determine the count themselves, e.g. ``variance`` / ``lasso``).
    tune:
        Run a grid search over each model's registered hyper-parameter grid
        inside the training folds (nested cross-validation).
    """

    # --- data -----------------------------------------------------------
    dataset: str = "dataset.csv"
    target_column: str | int = -1
    drop_columns: list[str] = field(default_factory=list)
    drop_target_values: list[str] = field(default_factory=list)
    min_class_count: int = 2
    holdout_size: float = 0.20

    # --- pipeline -------------------------------------------------------
    scaler: str = "standard"
    selector: str = "anova"
    n_features: int = 10
    models: list[str] = field(default_factory=lambda: ["random_forest"])
    class_weight_balanced: bool = True

    # --- validation -----------------------------------------------------
    cv: str = "stratified_kfold"
    n_splits: int = 5
    n_repeats: int = 1
    tune: bool = False
    permutation_importance: bool = True

    # --- execution ------------------------------------------------------
    random_state: int = 42
    n_jobs: int = -1
    output_dir: str = "runs"
    run_name: str | None = None
    make_plots: bool = True
    make_pdf: bool = True

    def __post_init__(self) -> None:
        if not 0.0 < self.holdout_size < 0.9:
            raise ValueError("holdout_size must lie in (0, 0.9)")
        if self.n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        if self.n_features < 1:
            raise ValueError("n_features must be at least 1")
        if isinstance(self.models, str):
            self.models = [self.models]
        if not self.models:
            raise ValueError("at least one model must be requested")

    # -- (de)serialisation ------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))
        return path

    @classmethod
    def load(cls, path: str | Path) -> RunConfig:
        """Build a config from a JSON file, ignoring unknown keys."""
        raw = json.loads(Path(path).read_text())
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> RunConfig:
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"unknown configuration keys: {sorted(unknown)}")
        return cls(**raw)
