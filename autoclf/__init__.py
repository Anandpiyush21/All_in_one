"""autoclf -- an automated, leakage-free pipeline for tabular classification.

The package turns a raw CSV into a validated, reproducible classification
experiment: preprocessing, feature selection, model fitting, cross-validated
evaluation and a publication-style report are all driven from a single
:class:`~autoclf.config.RunConfig`.

Design rule enforced throughout: *every* data-dependent transformation lives
inside a scikit-learn :class:`~sklearn.pipeline.Pipeline`, so it is re-fitted
from scratch on each training fold and never sees validation or hold-out rows.
"""

from autoclf.config import RunConfig
from autoclf.pipeline import build_pipeline

__all__ = ["RunConfig", "build_pipeline", "__version__"]
__version__ = "2.0.0"
