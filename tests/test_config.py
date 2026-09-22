import pytest

from autoclf.config import RunConfig


def test_defaults_are_valid():
    config = RunConfig()
    assert config.models == ["random_forest"]
    assert 0 < config.holdout_size < 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"holdout_size": 0.0},
        {"holdout_size": 0.95},
        {"n_splits": 1},
        {"n_features": 0},
        {"models": []},
    ],
)
def test_invalid_values_are_rejected(kwargs):
    with pytest.raises(ValueError):
        RunConfig(**kwargs)


def test_single_model_string_is_wrapped():
    assert RunConfig(models="svm").models == ["svm"]


def test_round_trips_through_json(tmp_path):
    original = RunConfig(scaler="robust", models=["svm", "knn"], n_splits=7)
    path = original.save(tmp_path / "config.json")
    assert RunConfig.load(path) == original


def test_unknown_keys_are_rejected():
    with pytest.raises(ValueError, match="unknown configuration keys"):
        RunConfig.from_dict({"scaler": "standard", "learning_rate": 0.1})
