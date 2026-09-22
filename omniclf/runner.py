"""Experiment orchestration: config in, run directory out."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from omniclf import report
from omniclf.config import RunConfig
from omniclf.cross_val import requires_groups
from omniclf.data import Dataset, load_dataset
from omniclf.evaluate import ModelResult, compare_models, evaluate_model

LOGGER = logging.getLogger(__name__)


@dataclass
class RunOutcome:
    """Handles to everything a finished run produced."""

    run_dir: Path
    dataset: Dataset
    results: list[ModelResult]
    comparisons: list[dict[str, Any]]
    artefacts: dict[str, Path]

    @property
    def best(self) -> ModelResult:
        return max(self.results, key=lambda r: r.cv_mean["f1_macro"])


def prepare_run_dir(config: RunConfig) -> Path:
    """Create ``<output_dir>/<run name>`` and return it."""
    name = config.run_name or datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(config.output_dir) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def run(config: RunConfig) -> RunOutcome:
    """Execute the full experiment described by ``config``."""
    if requires_groups(config.cv):
        raise ValueError(
            f"cv strategy {config.cv!r} needs a grouping column, which this tool does not "
            "collect; use 'stratified_kfold' instead"
        )

    run_dir = prepare_run_dir(config)
    config.save(run_dir / "config.json")
    LOGGER.info("run directory: %s", run_dir)

    dataset = load_dataset(config)
    if dataset.imbalance_ratio() > 3 and not config.class_weight_balanced:
        LOGGER.warning(
            "class imbalance is %.1f:1 and class weighting is off; macro metrics will suffer",
            dataset.imbalance_ratio(),
        )

    results: list[ModelResult] = []
    for model_key in config.models:
        results.append(evaluate_model(config, dataset, model_key))
        latest = results[-1]
        LOGGER.info(
            "%s: cv macro-F1 %.4f ± %.4f | hold-out %.4f (%.1fs)",
            latest.label,
            latest.cv_mean["f1_macro"],
            latest.cv_std["f1_macro"],
            latest.holdout["f1_macro"],
            latest.fit_seconds,
        )

    comparisons = compare_models(results)
    artefacts: dict[str, Path] = {
        "config": run_dir / "config.json",
        "results_json": report.write_json(run_dir / "results.json", config, dataset, results),
        "summary_csv": report.write_csv(run_dir / "summary.csv", results),
    }

    figures: list[Path] = []
    if config.make_plots:
        figures = report.build_figures(config, dataset, results, run_dir / "figures")
        artefacts["plots_pdf"] = report.write_figure_sheet(figures, run_dir / "plots.pdf")
    if config.make_pdf:
        artefacts["report_pdf"] = report.write_pdf(
            run_dir / "report.pdf", config, dataset, results, figures, comparisons
        )

    return RunOutcome(run_dir, dataset, results, comparisons, artefacts)
