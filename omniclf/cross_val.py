"""Cross-validation strategies.

``stratified_kfold`` is the default and the right choice for almost every
classification problem: it keeps the class proportions of the full sample in
every fold, which matters a great deal on the imbalanced targets that survey
data typically produces.
"""

from __future__ import annotations

from collections.abc import Callable

from sklearn.model_selection import (
    BaseCrossValidator,
    GroupKFold,
    KFold,
    RepeatedStratifiedKFold,
    ShuffleSplit,
    StratifiedKFold,
    StratifiedShuffleSplit,
    TimeSeriesSplit,
)

#: Registry key -> (label, factory(n_splits, n_repeats, random_state), needs_groups).
SPLITTERS: dict[str, tuple[str, Callable[[int, int, int], BaseCrossValidator], bool]] = {
    "stratified_kfold": (
        "Stratified K-Fold",
        lambda n, r, rs: StratifiedKFold(n_splits=n, shuffle=True, random_state=rs),
        False,
    ),
    "kfold": (
        "K-Fold",
        lambda n, r, rs: KFold(n_splits=n, shuffle=True, random_state=rs),
        False,
    ),
    "repeated_stratified": (
        "Repeated Stratified K-Fold",
        lambda n, r, rs: RepeatedStratifiedKFold(
            n_splits=n, n_repeats=max(r, 2), random_state=rs
        ),
        False,
    ),
    "group_kfold": ("Group K-Fold", lambda n, r, rs: GroupKFold(n_splits=n), True),
    "time_series": ("Time Series Split", lambda n, r, rs: TimeSeriesSplit(n_splits=n), False),
    "shuffle_split": (
        "Shuffle Split",
        lambda n, r, rs: ShuffleSplit(n_splits=n, test_size=0.25, random_state=rs),
        False,
    ),
    "stratified_shuffle": (
        "Stratified Shuffle Split",
        lambda n, r, rs: StratifiedShuffleSplit(
            n_splits=n, test_size=0.25, random_state=rs
        ),
        False,
    ),
}


def build_cv(
    key: str, n_splits: int = 5, n_repeats: int = 1, random_state: int = 42
) -> BaseCrossValidator:
    """Return a cross-validation splitter for a registry ``key``."""
    try:
        return SPLITTERS[key][1](n_splits, n_repeats, random_state)
    except KeyError:
        raise KeyError(
            f"unknown cv strategy {key!r}; available: {sorted(SPLITTERS)}"
        ) from None


def requires_groups(key: str) -> bool:
    """``True`` for splitters that cannot run without a ``groups`` vector."""
    return SPLITTERS[key][2]


def describe(key: str) -> str:
    return SPLITTERS[key][0]
