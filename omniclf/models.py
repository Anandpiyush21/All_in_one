"""Classifier registry (the former, misspelled ``modals.py``).

Each entry carries a factory *and* a hyper-parameter grid. The grids are kept
deliberately small so that nested cross-validation (``--tune``) stays tractable
on a laptop: the point of this project is methodological correctness, not a
record-breaking leaderboard score.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from sklearn.base import BaseEstimator
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

#: Models that accept ``class_weight="balanced"``.
_SUPPORTS_CLASS_WEIGHT = {
    "random_forest",
    "extra_trees",
    "svm",
    "logistic_regression",
    "decision_tree",
    "hist_gradient_boosting",
}

#: Registry key -> (label, factory(random_state), grid, notes).
MODELS: dict[str, tuple[str, Callable[[int], BaseEstimator], dict[str, list[Any]], str]] = {
    "random_forest": (
        "Random Forest",
        lambda rs: RandomForestClassifier(n_estimators=300, random_state=rs, n_jobs=-1),
        {
            "clf__n_estimators": [200, 500],
            "clf__max_depth": [None, 12],
            "clf__min_samples_leaf": [1, 5],
        },
        "bagged decision trees; strong tabular baseline, low tuning sensitivity",
    ),
    "extra_trees": (
        "Extremely Randomised Trees",
        lambda rs: ExtraTreesClassifier(n_estimators=300, random_state=rs, n_jobs=-1),
        {"clf__n_estimators": [300, 600], "clf__min_samples_leaf": [1, 5]},
        "randomised split thresholds; lower variance than a plain forest",
    ),
    "hist_gradient_boosting": (
        "Histogram Gradient Boosting",
        lambda rs: HistGradientBoostingClassifier(random_state=rs),
        {"clf__learning_rate": [0.05, 0.1], "clf__max_leaf_nodes": [31, 63]},
        "LightGBM-style boosting from scikit-learn; best accuracy/time trade-off",
    ),
    "gradient_boosting": (
        "Gradient Boosting",
        lambda rs: GradientBoostingClassifier(random_state=rs),
        {"clf__learning_rate": [0.05, 0.1], "clf__n_estimators": [100, 200]},
        "classical stage-wise boosting; slower but well understood",
    ),
    "logistic_regression": (
        "Logistic Regression",
        lambda rs: LogisticRegression(max_iter=2000, random_state=rs),
        {"clf__C": [0.1, 1.0, 10.0]},
        "linear, calibrated and interpretable; the reference model",
    ),
    "svm": (
        "Support Vector Machine (RBF)",
        lambda rs: SVC(probability=True, random_state=rs),
        {"clf__C": [1.0, 10.0], "clf__gamma": ["scale", 0.1]},
        "maximum-margin classifier; needs scaled inputs, scales poorly past ~10k rows",
    ),
    "knn": (
        "K-Nearest Neighbours",
        lambda rs: KNeighborsClassifier(n_jobs=-1),
        {"clf__n_neighbors": [5, 11, 21], "clf__weights": ["uniform", "distance"]},
        "instance-based; sensitive to scaling and to the curse of dimensionality",
    ),
    "naive_bayes": (
        "Gaussian Naive Bayes",
        lambda rs: GaussianNB(),
        {"clf__var_smoothing": [1e-9, 1e-7]},
        "strong independence assumption; fast probabilistic baseline",
    ),
    "decision_tree": (
        "Decision Tree",
        lambda rs: DecisionTreeClassifier(random_state=rs),
        {"clf__max_depth": [None, 8, 16], "clf__min_samples_leaf": [1, 10]},
        "single interpretable tree; high variance on its own",
    ),
    "lda": (
        "Linear Discriminant Analysis",
        lambda rs: LinearDiscriminantAnalysis(),
        {"clf__solver": ["svd", "lsqr"]},
        "generative linear model; a useful sanity check against logistic regression",
    ),
    "mlp": (
        "Neural Network (MLP)",
        lambda rs: MLPClassifier(max_iter=500, random_state=rs),
        {"clf__hidden_layer_sizes": [(64,), (128, 64)], "clf__alpha": [1e-4, 1e-2]},
        "feed-forward network; included for completeness, rarely wins on tabular data",
    ),
}


def build_model(
    key: str, random_state: int = 42, class_weight_balanced: bool = False
) -> BaseEstimator:
    """Instantiate a classifier, optionally re-weighting for class imbalance."""
    try:
        estimator = MODELS[key][1](random_state)
    except KeyError:
        raise KeyError(f"unknown model {key!r}; available: {sorted(MODELS)}") from None
    if class_weight_balanced and key in _SUPPORTS_CLASS_WEIGHT:
        estimator.set_params(class_weight="balanced")
    return estimator


def param_grid(key: str) -> dict[str, list[Any]]:
    """Hyper-parameter grid, keyed for use inside the ``clf`` pipeline step."""
    return dict(MODELS[key][2])


def describe(key: str) -> str:
    return MODELS[key][0]
