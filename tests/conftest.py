"""Shared fixtures.

The tests run on a small synthetic frame rather than the bundled survey data so
that the suite finishes in seconds and does not depend on the CSV being present.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from autoclf.config import RunConfig


@pytest.fixture(scope="session")
def toy_frame() -> pd.DataFrame:
    """400 rows: 3 informative numeric columns, 2 categorical, some missingness."""
    rng = np.random.default_rng(0)
    n = 400
    signal = rng.normal(size=n)
    frame = pd.DataFrame(
        {
            "informative_a": signal + rng.normal(scale=0.4, size=n),
            "informative_b": -signal + rng.normal(scale=0.6, size=n),
            "noise": rng.normal(size=n),
            "constant": 1.0,
            "region": rng.choice(["north", "south", "east"], size=n),
            "grade": rng.choice(["low", "high", None], size=n, p=[0.45, 0.45, 0.10]),
            "label": np.where(signal > 0, "yes", "no"),
        }
    )
    frame.loc[rng.choice(n, size=20, replace=False), "informative_a"] = np.nan
    return frame


@pytest.fixture(scope="session")
def toy_csv(tmp_path_factory, toy_frame) -> str:
    path = tmp_path_factory.mktemp("data") / "toy.csv"
    toy_frame.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def toy_config(toy_csv, tmp_path) -> RunConfig:
    return RunConfig(
        dataset=toy_csv,
        target_column="label",
        scaler="standard",
        selector="anova",
        n_features=4,
        models=["logistic_regression"],
        cv="stratified_kfold",
        n_splits=3,
        permutation_importance=False,
        output_dir=str(tmp_path / "runs"),
        run_name="test",
        n_jobs=1,
    )
