"""Dataset loading, schema inference and the train / hold-out split.

Only operations that are *row-independent* happen here (dropping unusable
columns, encoding the label, splitting). Everything that learns a statistic
from the data -- imputation values, category vocabularies, scaling constants --
is deferred to the pipeline in :mod:`omniclf.pipeline`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from omniclf.config import RunConfig

LOGGER = logging.getLogger(__name__)

#: Columns with more distinct values than this are treated as free-text /
#: identifiers rather than categories, and dropped.
MAX_CATEGORY_CARDINALITY = 50


@dataclass
class Dataset:
    """A loaded, split dataset plus the metadata reports need."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    numeric_columns: list[str]
    categorical_columns: list[str]
    class_names: list[str]
    target_name: str
    dropped_columns: list[str]
    n_rows_raw: int
    n_rows_used: int

    @property
    def n_features_in(self) -> int:
        return len(self.numeric_columns) + len(self.categorical_columns)

    @property
    def n_classes(self) -> int:
        return len(self.class_names)

    @property
    def is_binary(self) -> bool:
        return self.n_classes == 2

    def class_distribution(self) -> dict[str, int]:
        values, counts = np.unique(
            np.concatenate([self.y_train, self.y_test]), return_counts=True
        )
        return {self.class_names[int(v)]: int(c) for v, c in zip(values, counts, strict=True)}

    def imbalance_ratio(self) -> float:
        """Majority-to-minority class ratio; 1.0 means perfectly balanced."""
        counts = list(self.class_distribution().values())
        return max(counts) / max(min(counts), 1)


def _resolve_target(df: pd.DataFrame, target: str | int) -> str:
    if isinstance(target, int):
        return str(df.columns[target])
    if target not in df.columns:
        raise KeyError(
            f"target column {target!r} not in dataset; columns are {list(df.columns)}"
        )
    return target


def _dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee unique column names.

    Survey exports repeat headers (this dataset carries two 'Anemia level'
    columns). ``read_csv`` already suffixes repeats, but a DataFrame handed in
    from anywhere else may not be, and duplicate names make every later column
    selection ambiguous.
    """
    counts: dict[str, int] = {}
    renamed = []
    for col in df.columns:
        n = counts.get(col, 0)
        counts[col] = n + 1
        renamed.append(col if n == 0 else f"{col} ({n + 1})")
    df.columns = renamed
    return df


def load_dataset(config: RunConfig) -> Dataset:
    """Read the CSV named by ``config`` and return a split :class:`Dataset`."""
    path = Path(config.dataset)
    if not path.exists():
        raise FileNotFoundError(f"dataset not found: {path.resolve()}")

    df = _dedupe_columns(pd.read_csv(path, low_memory=False))
    n_rows_raw = len(df)
    LOGGER.info("loaded %s: %d rows x %d columns", path.name, n_rows_raw, df.shape[1])

    target_name = _resolve_target(df, config.target_column)
    dropped: list[str] = []

    for col in config.drop_columns:
        if col in df.columns and col != target_name:
            df = df.drop(columns=col)
            dropped.append(f"{col} (requested)")

    # Rows without a label carry no supervision signal.
    df = df[df[target_name].notna()]
    df = df[df[target_name].astype(str).str.strip() != ""]
    if config.drop_target_values:
        unusable = {v.strip().lower() for v in config.drop_target_values}
        df = df[~df[target_name].astype(str).str.strip().str.lower().isin(unusable)]

    # Rare classes cannot be stratified across folds.
    counts = df[target_name].value_counts()
    rare = counts[counts < max(config.min_class_count, 2)].index.tolist()
    if rare:
        LOGGER.warning("dropping %d class(es) with too few rows: %s", len(rare), rare)
        df = df[~df[target_name].isin(rare)]

    if df.empty:
        raise ValueError("no usable rows remain after cleaning the target column")

    y_raw = df[target_name].astype(str).str.strip()
    class_names = sorted(y_raw.unique())
    if len(class_names) < 2:
        raise ValueError(
            f"target {target_name!r} has a single class after cleaning; "
            "classification needs at least two"
        )
    y = y_raw.map({name: i for i, name in enumerate(class_names)}).to_numpy()

    X = df.drop(columns=[target_name])

    # Drop columns that cannot inform a model: all-missing, constant, or
    # high-cardinality free text (names, IDs, open-ended answers).
    for col in list(X.columns):
        series = X[col]
        if series.isna().all():
            X = X.drop(columns=col)
            dropped.append(f"{col} (all missing)")
        elif series.nunique(dropna=True) <= 1:
            X = X.drop(columns=col)
            dropped.append(f"{col} (constant)")
        elif (
            not pd.api.types.is_numeric_dtype(series)
            and not pd.api.types.is_bool_dtype(series)
            and series.nunique(dropna=True) > MAX_CATEGORY_CARDINALITY
        ):
            X = X.drop(columns=col)
            dropped.append(f"{col} (high-cardinality text)")

    if X.empty:
        raise ValueError("no usable feature columns remain after cleaning")

    numeric = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]

    stratify = y if min(np.bincount(y)) >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.holdout_size,
        random_state=config.random_state,
        stratify=stratify,
    )

    LOGGER.info(
        "%d usable rows -> %d train / %d hold-out; %d numeric + %d categorical features; %d classes",
        len(X),
        len(X_train),
        len(X_test),
        len(numeric),
        len(categorical),
        len(class_names),
    )

    return Dataset(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        numeric_columns=numeric,
        categorical_columns=categorical,
        class_names=class_names,
        target_name=target_name,
        dropped_columns=dropped,
        n_rows_raw=n_rows_raw,
        n_rows_used=len(X),
    )
