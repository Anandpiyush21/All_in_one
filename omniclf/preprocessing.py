"""Scaling strategies (the former ``normalization.py``).

Every scaler is returned *unfitted*; it is fitted by the enclosing pipeline on
the training portion of each fold only, which is what keeps the reported scores
honest.
"""

from __future__ import annotations

from collections.abc import Callable

from sklearn.base import TransformerMixin
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)

#: Registry key -> (human label, factory, one-line rationale).
SCALERS: dict[str, tuple[str, Callable[[], TransformerMixin], str]] = {
    "standard": (
        "StandardScaler",
        lambda: StandardScaler(),
        "zero mean / unit variance; default for distance- and gradient-based learners",
    ),
    "minmax": (
        "MinMaxScaler",
        lambda: MinMaxScaler(),
        "squashes each feature to [0, 1]; keeps sparsity of bounded counts",
    ),
    "robust": (
        "RobustScaler",
        lambda: RobustScaler(),
        "median / IQR scaling; preferred when outliers are present",
    ),
    "normalizer": (
        "Normalizer",
        lambda: Normalizer(),
        "scales each *sample* to unit norm; for direction-sensitive data",
    ),
    "maxabs": (
        "MaxAbsScaler",
        lambda: MaxAbsScaler(),
        "divides by the max absolute value; sign- and sparsity-preserving",
    ),
    "power": (
        "PowerTransformer",
        lambda: PowerTransformer(method="yeo-johnson", standardize=True),
        "Yeo-Johnson transform towards Gaussianity; handles skew and zeros",
    ),
    "quantile": (
        "QuantileTransformer",
        lambda: QuantileTransformer(output_distribution="normal", random_state=0),
        "rank-based mapping to a normal distribution; robust to heavy tails",
    ),
    "none": (
        "No scaling",
        lambda: "passthrough",  # type: ignore[return-value]
        "baseline for scale-invariant models such as trees",
    ),
}


def build_scaler(key: str):
    """Return an unfitted scaler for a registry ``key``."""
    try:
        return SCALERS[key][1]()
    except KeyError:
        raise KeyError(
            f"unknown scaler {key!r}; available: {sorted(SCALERS)}"
        ) from None


def describe(key: str) -> str:
    """Human-readable label used in reports."""
    return SCALERS[key][0]
