"""Every registry entry must actually build, and build something usable."""

import pickle

import pytest
from sklearn.base import BaseEstimator

from autoclf.cross_val import SPLITTERS, build_cv, requires_groups
from autoclf.feature_selection import SELECTORS, build_selector
from autoclf.models import MODELS, build_model, param_grid
from autoclf.preprocessing import SCALERS, build_scaler


@pytest.mark.parametrize("key", sorted(SCALERS))
def test_every_scaler_builds(key):
    scaler = build_scaler(key)
    assert scaler == "passthrough" or hasattr(scaler, "fit_transform")


@pytest.mark.parametrize("key", sorted(SELECTORS))
def test_every_selector_builds_and_pickles(key):
    selector = build_selector(key, n_features=3, random_state=0)
    # joblib pickles estimators to ship them to worker processes, so a
    # lambda hidden in a registry entry would break n_jobs > 1.
    pickle.loads(pickle.dumps(selector))


@pytest.mark.parametrize("key", sorted(SPLITTERS))
def test_every_splitter_builds(key):
    splitter = build_cv(key, n_splits=3, n_repeats=2, random_state=0)
    assert splitter.get_n_splits() >= 3
    assert isinstance(requires_groups(key), bool)


@pytest.mark.parametrize("key", sorted(MODELS))
def test_every_model_builds(key):
    assert isinstance(build_model(key, random_state=0), BaseEstimator)


@pytest.mark.parametrize("key", sorted(MODELS))
def test_every_grid_targets_the_clf_step(key):
    grid = param_grid(key)
    assert grid, f"{key} has an empty grid"
    assert all(name.startswith("clf__") for name in grid)
    estimator = build_model(key, random_state=0)
    for name in grid:
        assert name.removeprefix("clf__") in estimator.get_params()


def test_balanced_weights_applied_only_where_supported():
    assert build_model("random_forest", class_weight_balanced=True).class_weight == "balanced"
    # GaussianNB has no class_weight parameter; asking for one must not raise.
    build_model("naive_bayes", class_weight_balanced=True)


@pytest.mark.parametrize(
    "builder, key",
    [(build_scaler, "nope"), (build_model, "nope")],
)
def test_unknown_keys_raise_with_a_helpful_message(builder, key):
    with pytest.raises(KeyError, match="available"):
        builder(key)
