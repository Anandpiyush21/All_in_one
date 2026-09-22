import numpy as np
import pandas as pd
import pytest

from autoclf.config import RunConfig
from autoclf.data import load_dataset


def test_split_sizes_and_schema(toy_config):
    data = load_dataset(toy_config)
    assert len(data.X_train) + len(data.X_test) == data.n_rows_used
    assert len(data.X_test) == pytest.approx(data.n_rows_used * 0.2, abs=2)
    assert data.class_names == ["no", "yes"]
    assert set(data.numeric_columns) == {"informative_a", "informative_b", "noise"}
    assert set(data.categorical_columns) == {"region", "grade"}


def test_constant_column_is_dropped(toy_config):
    data = load_dataset(toy_config)
    assert "constant" not in data.X_train.columns
    assert any("constant" in note for note in data.dropped_columns)


def test_split_is_stratified(toy_config):
    data = load_dataset(toy_config)
    train_share = data.y_train.mean()
    test_share = data.y_test.mean()
    assert abs(train_share - test_share) < 0.05


def test_rows_without_a_label_are_dropped(tmp_path, toy_frame):
    frame = toy_frame.copy()
    frame.loc[:19, "label"] = np.nan
    path = tmp_path / "missing.csv"
    frame.to_csv(path, index=False)
    data = load_dataset(RunConfig(dataset=str(path), target_column="label"))
    assert data.n_rows_used == len(frame) - 20


def test_target_values_can_be_excluded(tmp_path, toy_frame):
    frame = toy_frame.copy()
    frame.loc[:29, "label"] = "Don't know"
    path = tmp_path / "dk.csv"
    frame.to_csv(path, index=False)
    data = load_dataset(
        RunConfig(dataset=str(path), target_column="label", drop_target_values=["Don't know"])
    )
    assert data.class_names == ["no", "yes"]


def test_high_cardinality_text_is_dropped(tmp_path, toy_frame):
    frame = toy_frame.copy()
    frame["respondent_id"] = [f"id-{i}" for i in range(len(frame))]
    path = tmp_path / "ids.csv"
    frame.to_csv(path, index=False)
    data = load_dataset(RunConfig(dataset=str(path), target_column="label"))
    assert "respondent_id" not in data.X_train.columns


def test_duplicate_headers_are_disambiguated(tmp_path):
    path = tmp_path / "dupes.csv"
    path.write_text("a,a,label\n1,2,x\n3,4,y\n2,1,x\n4,3,y\n")
    data = load_dataset(RunConfig(dataset=str(path), target_column="label", holdout_size=0.5))
    assert len(set(data.X_train.columns)) == len(data.X_train.columns)


def test_single_class_target_is_rejected(tmp_path, toy_frame):
    frame = toy_frame.copy()
    frame["label"] = "only"
    path = tmp_path / "one.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="single class"):
        load_dataset(RunConfig(dataset=str(path), target_column="label"))


def test_missing_file_is_reported_clearly():
    with pytest.raises(FileNotFoundError):
        load_dataset(RunConfig(dataset="does-not-exist.csv"))


def test_imbalance_ratio(tmp_path):
    frame = pd.DataFrame({"x": range(100), "label": ["a"] * 80 + ["b"] * 20})
    path = tmp_path / "imb.csv"
    frame.to_csv(path, index=False)
    data = load_dataset(RunConfig(dataset=str(path), target_column="label"))
    assert data.imbalance_ratio() == pytest.approx(4.0)
