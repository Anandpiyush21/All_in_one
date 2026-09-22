import numpy as np

from omniclf.data import load_dataset
from omniclf.evaluate import CV_METRICS, compare_models, evaluate_model


def test_all_metrics_are_collected(toy_config):
    data = load_dataset(toy_config)
    result = evaluate_model(toy_config, data, "logistic_regression")
    assert set(result.cv_scores) == set(CV_METRICS)
    assert all(len(v) == toy_config.n_splits for v in result.cv_scores.values())
    assert set(CV_METRICS).issubset(result.holdout)


def test_signal_is_learned(toy_config):
    """The toy target is a deterministic function of a latent variable the
    features track, so a linear model must beat chance by a wide margin."""
    data = load_dataset(toy_config)
    result = evaluate_model(toy_config, data, "logistic_regression")
    assert result.cv_mean["balanced_accuracy"] > 0.75
    assert result.holdout["mcc"] > 0.5


def test_cross_validation_and_holdout_agree(toy_config):
    """A gap here would point at leakage or at an unstable split."""
    data = load_dataset(toy_config)
    result = evaluate_model(toy_config, data, "logistic_regression")
    gap = abs(result.cv_mean["f1_macro"] - result.holdout["f1_macro"])
    assert gap < 0.15


def test_confusion_matrix_accounts_for_every_holdout_row(toy_config):
    data = load_dataset(toy_config)
    result = evaluate_model(toy_config, data, "logistic_regression")
    assert result.confusion.sum() == len(data.y_test)
    assert result.confusion.shape == (data.n_classes, data.n_classes)


def test_binary_roc_auc_is_reported(toy_config):
    data = load_dataset(toy_config)
    result = evaluate_model(toy_config, data, "logistic_regression")
    assert 0.5 < result.holdout["roc_auc"] <= 1.0


def test_tuning_records_the_chosen_parameters(toy_config):
    toy_config.tune = True
    data = load_dataset(toy_config)
    result = evaluate_model(toy_config, data, "logistic_regression")
    assert result.best_params and all(k.startswith("clf__") for k in result.best_params)


def test_permutation_importance_ranks_the_informative_columns_first(toy_config):
    toy_config.permutation_importance = True
    data = load_dataset(toy_config)
    result = evaluate_model(toy_config, data, "logistic_regression")
    top_two = {name for name, _, _ in result.importances[:2]}
    assert top_two & {"informative_a", "informative_b"}


def test_comparison_is_paired_and_ordered(toy_config):
    data = load_dataset(toy_config)
    results = [
        evaluate_model(toy_config, data, key)
        for key in ("logistic_regression", "naive_bayes")
    ]
    comparisons = compare_models(results)
    assert len(comparisons) == 1
    row = comparisons[0]
    assert row["mean_difference"] >= 0  # the champion is listed first
    assert 0.0 <= row["p_value"] <= 1.0
    assert isinstance(row["significant"], bool)


def test_comparison_needs_two_models(toy_config):
    data = load_dataset(toy_config)
    assert compare_models([evaluate_model(toy_config, data, "naive_bayes")]) == []


def test_summary_row_is_flat_and_serialisable(toy_config):
    data = load_dataset(toy_config)
    result = evaluate_model(toy_config, data, "logistic_regression")
    row = result.summary_row()
    assert row["model"] == "Logistic Regression"
    assert all(not isinstance(v, (list, dict, np.ndarray)) for v in row.values())
