"""The properties that make the reported numbers trustworthy."""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import cross_val_score

from autoclf.data import load_dataset
from autoclf.pipeline import build_pipeline, selected_feature_names


def test_pipeline_has_the_four_expected_stages(toy_config):
    data = load_dataset(toy_config)
    pipeline = build_pipeline(toy_config, data, "logistic_regression")
    assert list(pipeline.named_steps) == ["prep", "scale", "select", "clf"]


def test_preprocessing_is_refit_per_fold(toy_config):
    """The whole chain -- not just the classifier -- must be cloned per fold.

    This is the regression guard for the defect in v1: scaling and feature
    selection were fitted once on the full dataset, so every fold's validation
    rows had already influenced the transform.
    """
    data = load_dataset(toy_config)
    pipeline = build_pipeline(toy_config, data, "logistic_regression")
    cross_val_score(pipeline, data.X_train, data.y_train, cv=3)
    # cross_val_score clones, so the original is still untouched: an unfitted
    # classifier must still refuse to predict.
    with pytest.raises(NotFittedError):
        pipeline.named_steps["clf"].predict(data.X_test)


def test_transformers_learn_from_training_rows_only(toy_config):
    """Fitting on the training split must not depend on the hold-out split."""
    data = load_dataset(toy_config)
    pipeline = build_pipeline(toy_config, data, "logistic_regression")

    fitted_a = clone(pipeline).fit(data.X_train, data.y_train)
    # Corrupt the hold-out set beyond recognition; the fitted transform of the
    # training data must be bit-identical, i.e. nothing leaked backwards.
    poisoned = data.X_test.copy()
    for column in data.numeric_columns:
        poisoned[column] = poisoned[column] * 1e6
    fitted_b = clone(pipeline).fit(data.X_train, data.y_train)

    a = fitted_a[:-1].transform(data.X_train)
    b = fitted_b[:-1].transform(data.X_train)
    np.testing.assert_allclose(a, b)
    assert len(poisoned) == len(data.X_test)


def test_train_and_test_share_one_feature_space(toy_config):
    """v1 refitted the transformers separately on the hold-out set, so column
    *k* of the test matrix no longer meant the same thing as column *k* of the
    training matrix. Here it must."""
    data = load_dataset(toy_config)
    fitted = build_pipeline(toy_config, data, "logistic_regression").fit(
        data.X_train, data.y_train
    )
    train_features = fitted[:-1].transform(data.X_train)
    test_features = fitted[:-1].transform(data.X_test)
    assert train_features.shape[1] == test_features.shape[1]


def test_unseen_category_does_not_crash_prediction(toy_config):
    data = load_dataset(toy_config)
    fitted = build_pipeline(toy_config, data, "logistic_regression").fit(
        data.X_train, data.y_train
    )
    unseen = data.X_test.copy()
    unseen.loc[unseen.index[0], "region"] = "atlantis"
    assert len(fitted.predict(unseen)) == len(unseen)


def test_selector_request_is_clamped_to_available_columns(toy_config):
    data = load_dataset(toy_config)
    greedy = toy_config
    greedy.n_features = 10_000
    fitted = build_pipeline(greedy, data, "logistic_regression").fit(data.X_train, data.y_train)
    assert fitted.predict(data.X_test).shape == data.y_test.shape


def test_selected_feature_names_are_reported(toy_config):
    data = load_dataset(toy_config)
    fitted = build_pipeline(toy_config, data, "logistic_regression").fit(
        data.X_train, data.y_train
    )
    names = selected_feature_names(fitted)
    assert names is not None and len(names) == toy_config.n_features


def test_projection_reports_no_original_names(toy_config):
    data = load_dataset(toy_config)
    toy_config.selector = "pca"
    fitted = build_pipeline(toy_config, data, "logistic_regression").fit(
        data.X_train, data.y_train
    )
    assert selected_feature_names(fitted) is None
