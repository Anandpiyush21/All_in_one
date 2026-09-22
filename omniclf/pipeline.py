"""Assembly of the end-to-end estimator.

The pipeline has four stages::

    prep  ->  scale  ->  select  ->  clf

``prep`` is a :class:`~sklearn.compose.ColumnTransformer` that imputes and
one-hot encodes categorical columns and imputes numeric ones. Because the whole
chain is a single estimator, ``cross_validate`` refits *all four* stages on each
training fold. That is the structural fix for the leakage in the original
version of this project, which scaled and selected features once on the full
dataset (and, worse, re-fitted the transformers separately on the hold-out set,
so train and test features no longer even referred to the same quantities).
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from omniclf.config import RunConfig
from omniclf.data import Dataset
from omniclf.feature_selection import build_selector
from omniclf.models import build_model
from omniclf.preprocessing import build_scaler


def build_preprocessor(dataset: Dataset) -> ColumnTransformer:
    """Impute + encode, learning every constant from the training fold only."""
    numeric = Pipeline(
        [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "encode",
                OneHotEncoder(handle_unknown="ignore", min_frequency=0.01, sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric, dataset.numeric_columns),
            ("cat", categorical, dataset.categorical_columns),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_pipeline(config: RunConfig, dataset: Dataset, model_key: str) -> Pipeline:
    """Return the unfitted four-stage pipeline for one model."""
    # One-hot encoding expands the column count, so cap the request at a value
    # the selector can actually satisfy on the widest possible design matrix.
    n_features = max(1, min(config.n_features, _max_expanded_features(dataset)))
    return Pipeline(
        [
            ("prep", build_preprocessor(dataset)),
            ("scale", build_scaler(config.scaler)),
            ("select", build_selector(config.selector, n_features, config.random_state)),
            (
                "clf",
                build_model(
                    model_key,
                    random_state=config.random_state,
                    class_weight_balanced=config.class_weight_balanced,
                ),
            ),
        ]
    )


def _max_expanded_features(dataset: Dataset) -> int:
    """Lower bound on the post-encoding column count.

    ``OneHotEncoder`` only ever widens the matrix, so the raw column count is a
    safe floor and keeps ``SelectKBest(k=...)`` from raising on narrow data.
    """
    return max(dataset.n_features_in, 1)


def selected_feature_names(fitted: Pipeline) -> list[str] | None:
    """Names surviving the ``select`` step of a *fitted* pipeline.

    Returns ``None`` for projections such as PCA, whose outputs are linear
    combinations rather than original features.
    """
    try:
        names = list(fitted.named_steps["prep"].get_feature_names_out())
    except Exception:  # pragma: no cover - defensive
        return None

    select = fitted.named_steps.get("select")
    if select is None or select == "passthrough":
        return names
    if not hasattr(select, "get_support"):
        return None  # PCA and friends
    return [n for n, keep in zip(names, select.get_support(), strict=True) if keep]
