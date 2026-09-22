"""Cross-validated and hold-out evaluation.

Metric choice matters more than model choice on imbalanced data. Plain
accuracy is reported because it is expected, but *balanced accuracy*, *macro
F1* and *Matthews' correlation coefficient* are the numbers to read: a
degenerate majority-class classifier scores well on the first and near zero on
the last.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.pipeline import Pipeline

from omniclf.config import RunConfig
from omniclf.cross_val import build_cv
from omniclf.data import Dataset
from omniclf.models import describe as describe_model
from omniclf.models import param_grid
from omniclf.pipeline import build_pipeline, selected_feature_names

LOGGER = logging.getLogger(__name__)

#: Scores collected for every fold. ``_macro`` variants weight each class
#: equally, which is what we want when the minority class is the point.
CV_METRICS = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "precision_macro": "precision_macro",
    "recall_macro": "recall_macro",
    "f1_macro": "f1_macro",
    "mcc": "matthews_corrcoef",
}


@dataclass
class ModelResult:
    """Everything measured for a single model."""

    key: str
    label: str
    cv_scores: dict[str, np.ndarray]
    holdout: dict[str, float]
    confusion: np.ndarray
    report_text: str
    fit_seconds: float
    best_params: dict[str, Any] | None = None
    selected_features: list[str] | None = None
    importances: list[tuple[str, float, float]] = field(default_factory=list)
    estimator: Pipeline | None = None

    @property
    def cv_mean(self) -> dict[str, float]:
        return {k: float(np.mean(v)) for k, v in self.cv_scores.items()}

    @property
    def cv_std(self) -> dict[str, float]:
        return {k: float(np.std(v, ddof=1)) if len(v) > 1 else 0.0 for k, v in self.cv_scores.items()}

    def summary_row(self) -> dict[str, Any]:
        mean, std = self.cv_mean, self.cv_std
        row: dict[str, Any] = {"model": self.label}
        for metric in CV_METRICS:
            row[f"cv_{metric}"] = round(mean[metric], 4)
            row[f"cv_{metric}_std"] = round(std[metric], 4)
        for metric, value in self.holdout.items():
            row[f"holdout_{metric}"] = round(value, 4)
        row["fit_seconds"] = round(self.fit_seconds, 2)
        return row

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.key,
            "label": self.label,
            "cv_mean": {k: round(v, 6) for k, v in self.cv_mean.items()},
            "cv_std": {k: round(v, 6) for k, v in self.cv_std.items()},
            "cv_folds": {k: [round(float(x), 6) for x in v] for k, v in self.cv_scores.items()},
            "holdout": {k: round(v, 6) for k, v in self.holdout.items()},
            "confusion_matrix": self.confusion.tolist(),
            "best_params": self.best_params,
            "n_selected_features": (
                len(self.selected_features) if self.selected_features is not None else None
            ),
            "top_features": [
                {"feature": f, "importance": round(m, 6), "std": round(s, 6)}
                for f, m, s in self.importances[:15]
            ],
            "fit_seconds": round(self.fit_seconds, 3),
        }


def _holdout_metrics(
    estimator: Pipeline, X, y_true: np.ndarray, n_classes: int
) -> tuple[dict[str, float], np.ndarray, str, np.ndarray]:
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        precision_score,
        recall_score,
    )

    y_pred = estimator.predict(X)
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }

    if hasattr(estimator, "predict_proba"):
        try:
            proba = estimator.predict_proba(X)
            if n_classes == 2:
                metrics["roc_auc"] = float(roc_auc_score(y_true, proba[:, 1]))
            else:
                metrics["roc_auc"] = float(
                    roc_auc_score(y_true, proba, multi_class="ovr", average="macro")
                )
        except ValueError:  # a class missing from the hold-out split
            LOGGER.debug("ROC-AUC unavailable for this split")

    report = classification_report(y_true, y_pred, zero_division=0)
    return metrics, confusion_matrix(y_true, y_pred), report, y_pred


def evaluate_model(
    config: RunConfig, dataset: Dataset, model_key: str, groups: np.ndarray | None = None
) -> ModelResult:
    """Cross-validate one model, refit it on all training data, score the hold-out."""
    pipeline = build_pipeline(config, dataset, model_key)
    cv = build_cv(config.cv, config.n_splits, config.n_repeats, config.random_state)
    started = time.perf_counter()

    LOGGER.info("evaluating %s (%d-fold %s)", describe_model(model_key), config.n_splits, config.cv)
    cv_out = cross_validate(
        pipeline,
        dataset.X_train,
        dataset.y_train,
        cv=cv,
        groups=groups,
        scoring=CV_METRICS,
        n_jobs=config.n_jobs,
        error_score="raise",
    )
    cv_scores = {name: cv_out[f"test_{name}"] for name in CV_METRICS}

    best_params: dict[str, Any] | None = None
    final = clone(pipeline)
    if config.tune:
        grid = param_grid(model_key)
        LOGGER.info("tuning %s over %d parameter(s)", model_key, len(grid))
        search = GridSearchCV(
            final,
            grid,
            scoring="f1_macro",
            cv=build_cv("stratified_kfold", 3, 1, config.random_state),
            n_jobs=config.n_jobs,
            refit=True,
        )
        search.fit(dataset.X_train, dataset.y_train)
        final = search.best_estimator_
        best_params = {k: str(v) for k, v in search.best_params_.items()}
    else:
        final.fit(dataset.X_train, dataset.y_train)

    metrics, cm, report, _ = _holdout_metrics(
        final, dataset.X_test, dataset.y_test, dataset.n_classes
    )
    features = selected_feature_names(final)

    importances: list[tuple[str, float, float]] = []
    if config.permutation_importance and features:
        importances = _permutation_importances(config, final, dataset, features)

    return ModelResult(
        key=model_key,
        label=describe_model(model_key),
        cv_scores=cv_scores,
        holdout=metrics,
        confusion=cm,
        report_text=report,
        fit_seconds=time.perf_counter() - started,
        best_params=best_params,
        selected_features=features,
        importances=importances,
        estimator=final,
    )


def _permutation_importances(
    config: RunConfig, estimator: Pipeline, dataset: Dataset, feature_names: list[str]
) -> list[tuple[str, float, float]]:
    """Model-agnostic importance measured on the hold-out set.

    Permutation importance is computed on *unseen* data and applies to any
    estimator, unlike impurity importance, which is biased towards
    high-cardinality features and is measured on training data.
    """
    # Permute the raw input columns: that keeps the attribution interpretable
    # in terms of the original survey questions rather than one-hot dummies.
    columns = list(dataset.X_test.columns)
    sample = min(len(dataset.X_test), 3000)
    X = dataset.X_test.iloc[:sample]
    y = dataset.y_test[:sample]
    try:
        result = permutation_importance(
            estimator,
            X,
            y,
            scoring="f1_macro",
            n_repeats=5,
            random_state=config.random_state,
            n_jobs=config.n_jobs,
        )
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.warning("permutation importance failed: %s", exc)
        return []
    ranked = sorted(
        zip(columns, result.importances_mean, result.importances_std, strict=True),
        key=lambda t: t[1],
        reverse=True,
    )
    return [(str(f), float(m), float(s)) for f, m, s in ranked]


def compare_models(
    results: list[ModelResult], metric: str = "f1_macro", alpha: float = 0.05
) -> list[dict[str, Any]]:
    """Paired comparison of every model against the best one.

    Fold scores are paired (the same splits are used for every model), so a
    paired t-test on the per-fold differences is the appropriate test. With the
    usual 5 folds this is under-powered, so the verdict is reported as a hint,
    not a proof -- which is exactly how such comparisons should be read.
    """
    if len(results) < 2:
        return []

    ranked = sorted(results, key=lambda r: r.cv_mean[metric], reverse=True)
    champion = ranked[0]
    rows: list[dict[str, Any]] = []
    for challenger in ranked[1:]:
        a = np.asarray(champion.cv_scores[metric], dtype=float)
        b = np.asarray(challenger.cv_scores[metric], dtype=float)
        n = min(len(a), len(b))
        diff = a[:n] - b[:n]
        if n < 2 or np.allclose(diff, 0):
            p_value = 1.0
        else:
            p_value = float(stats.ttest_rel(a[:n], b[:n]).pvalue)
        rows.append(
            {
                "best": champion.label,
                "versus": challenger.label,
                "mean_difference": round(float(diff.mean()), 4),
                "p_value": round(p_value, 4),
                "significant": bool(p_value < alpha),
            }
        )
    return rows
