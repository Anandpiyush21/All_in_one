"""Feature-selection / dimensionality-reduction strategies.

All selectors are supervised-safe: they are refitted inside every
cross-validation fold, so the selected subset is never chosen with knowledge of
the validation rows. Selecting features once on the full dataset -- a very
common mistake -- inflates accuracy by several points on wide data.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (
    RFE,
    SelectFromModel,
    SelectKBest,
    VarianceThreshold,
    f_classif,
    mutual_info_classif,
)
from sklearn.linear_model import LogisticRegression


def _mutual_info(X, y, random_state: int = 0):
    """Picklable wrapper around :func:`mutual_info_classif`.

    A bare ``lambda`` here would break ``n_jobs > 1``, because joblib pickles
    the estimator when it ships it to a worker process.
    """
    return mutual_info_classif(X, y, random_state=random_state)

#: Registry key -> (label, factory(n_features, random_state), rationale).
SELECTORS: dict[str, tuple[str, Callable[[int, int], object], str]] = {
    "anova": (
        "SelectKBest (ANOVA F-test)",
        lambda k, rs: SelectKBest(score_func=f_classif, k=k),
        "univariate filter: keeps the k features with the largest between-class F ratio",
    ),
    "mutual_info": (
        "SelectKBest (mutual information)",
        lambda k, rs: SelectKBest(
            score_func=partial(_mutual_info, random_state=rs), k=k
        ),
        "non-parametric filter: captures non-linear feature/target dependence",
    ),
    "chi2_like": (
        "SelectKBest (F-statistic)",
        lambda k, rs: SelectKBest(score_func=f_classif, k=k),
        "alias of the ANOVA filter, kept for parity with the original tool",
    ),
    "variance": (
        "VarianceThreshold",
        lambda k, rs: VarianceThreshold(threshold=0.01),
        "unsupervised filter: drops near-constant features before modelling",
    ),
    "rfe": (
        "Recursive Feature Elimination",
        lambda k, rs: RFE(
            estimator=RandomForestClassifier(
                n_estimators=100, random_state=rs, n_jobs=-1
            ),
            n_features_to_select=k,
            step=0.2,
        ),
        "wrapper: repeatedly drops the weakest 20% by impurity importance",
    ),
    "pca": (
        "Principal Component Analysis",
        lambda k, rs: PCA(n_components=k, random_state=rs),
        "projection onto k orthogonal directions of maximum variance",
    ),
    "lasso": (
        "L1-penalised selection (LASSO)",
        lambda k, rs: SelectFromModel(
            LogisticRegression(
                penalty="l1", solver="liblinear", C=0.1, random_state=rs
            ),
            max_features=k,
        ),
        "embedded: keeps features with non-zero weight under an L1 penalty",
    ),
    "model_based": (
        "Tree importance selection",
        lambda k, rs: SelectFromModel(
            RandomForestClassifier(n_estimators=200, random_state=rs, n_jobs=-1),
            max_features=k,
            threshold=-np.inf,
        ),
        "embedded: keeps the k features with the highest ensemble importance",
    ),
    "none": (
        "No selection",
        lambda k, rs: "passthrough",
        "baseline using the full feature set",
    ),
}


def build_selector(key: str, n_features: int, random_state: int = 0):
    """Return an unfitted selector for a registry ``key``."""
    try:
        return SELECTORS[key][1](n_features, random_state)
    except KeyError:
        raise KeyError(
            f"unknown selector {key!r}; available: {sorted(SELECTORS)}"
        ) from None


def describe(key: str) -> str:
    return SELECTORS[key][0]
