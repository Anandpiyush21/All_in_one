"""Command-line interface.

Three ways in, one engine behind them::

    python main.py                               # guided prompts (original UX)
    python main.py --models random_forest svm    # fully scripted
    python main.py --config runs/.../config.json # exact replay of a past run
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from autoclf import __version__
from autoclf.config import RunConfig
from autoclf.cross_val import SPLITTERS
from autoclf.feature_selection import SELECTORS
from autoclf.models import MODELS
from autoclf.preprocessing import SCALERS
from autoclf.runner import RunOutcome, run

LOGGER = logging.getLogger("autoclf")


def build_parser() -> argparse.ArgumentParser:
    defaults = RunConfig()
    parser = argparse.ArgumentParser(
        prog="autoclf",
        description="Automated, leakage-free classification pipeline for tabular data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"autoclf {__version__}")

    data = parser.add_argument_group("data")
    data.add_argument("--dataset", default=defaults.dataset, help="input CSV")
    data.add_argument(
        "--target", default="-1",
        help="target column name, or an index such as -1 for the last column",
    )
    data.add_argument("--drop-columns", nargs="*", default=[], help="columns to exclude")
    data.add_argument(
        "--drop-target-values", nargs="*", default=[],
        help="label values to discard, e.g. \"Don't know\"",
    )
    data.add_argument("--holdout-size", type=float, default=defaults.holdout_size)

    pipe = parser.add_argument_group("pipeline")
    pipe.add_argument("--scaler", choices=sorted(SCALERS), default=defaults.scaler)
    pipe.add_argument("--selector", choices=sorted(SELECTORS), default=defaults.selector)
    pipe.add_argument("--n-features", type=int, default=defaults.n_features)
    pipe.add_argument(
        "--models", nargs="+", choices=sorted(MODELS), default=defaults.models,
        help="one or more classifiers to benchmark against each other",
    )
    pipe.add_argument("--all-models", action="store_true", help="benchmark every registered model")
    pipe.add_argument(
        "--no-class-weight", dest="class_weight_balanced", action="store_false",
        help="disable balanced class weights (they are on by default)",
    )

    val = parser.add_argument_group("validation")
    val.add_argument("--cv", choices=sorted(SPLITTERS), default=defaults.cv)
    val.add_argument("--n-splits", type=int, default=defaults.n_splits)
    val.add_argument("--n-repeats", type=int, default=defaults.n_repeats)
    val.add_argument("--tune", action="store_true", help="grid search inside the training folds")
    val.add_argument("--no-importance", dest="permutation_importance", action="store_false")

    out = parser.add_argument_group("execution and output")
    out.add_argument("--output-dir", default=defaults.output_dir)
    out.add_argument("--run-name", default=None, help="directory name (default: a timestamp)")
    out.add_argument("--random-state", type=int, default=defaults.random_state)
    out.add_argument("--n-jobs", type=int, default=defaults.n_jobs)
    out.add_argument("--no-plots", dest="make_plots", action="store_false")
    out.add_argument("--no-pdf", dest="make_pdf", action="store_false")
    out.add_argument("-v", "--verbose", action="store_true", help="debug-level logging")

    mode = parser.add_argument_group("mode")
    mode.add_argument(
        "-i", "--interactive", action="store_true",
        help="ask for the options instead of reading them from flags",
    )
    mode.add_argument("--config", help="replay a saved config.json (other flags are ignored)")
    mode.add_argument("--list-options", action="store_true", help="print the registries and exit")
    return parser


def configure_logging(run_dir: Path | None = None, verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if run_dir is not None:
        handlers.append(logging.FileHandler(run_dir / "run.log"))
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-7s %(name)s  %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)


def print_options() -> None:
    for title, registry, idx in (
        ("Scalers", SCALERS, 2),
        ("Feature selectors", SELECTORS, 2),
        ("Cross-validation", SPLITTERS, 0),
        ("Models", MODELS, 3),
    ):
        print(f"\n{title}")
        for key, entry in registry.items():
            print(f"  {key:<24} {entry[0]:<34} {entry[idx] if idx else ''}")
    print()


def _parse_target(raw: str) -> str | int:
    try:
        return int(raw)
    except ValueError:
        return raw


def config_from_args(args: argparse.Namespace) -> RunConfig:
    """Turn parsed flags into a :class:`RunConfig`."""
    if args.config:
        return RunConfig.load(args.config)
    models = sorted(MODELS) if args.all_models else list(args.models)
    return RunConfig(
        dataset=args.dataset,
        target_column=_parse_target(args.target),
        drop_columns=list(args.drop_columns),
        drop_target_values=list(args.drop_target_values),
        holdout_size=args.holdout_size,
        scaler=args.scaler,
        selector=args.selector,
        n_features=args.n_features,
        models=models,
        class_weight_balanced=args.class_weight_balanced,
        cv=args.cv,
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
        tune=args.tune,
        permutation_importance=args.permutation_importance,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        output_dir=args.output_dir,
        run_name=args.run_name,
        make_plots=args.make_plots,
        make_pdf=args.make_pdf,
    )


def summarise(outcome: RunOutcome) -> None:
    """Print the leaderboard and the paths of everything written."""
    best = outcome.best
    width = max(len(r.label) for r in outcome.results) + 2
    print("\n" + "=" * (width + 46))
    print(f"{'model':<{width}}{'cv macro-F1':>16}{'hold-out F1':>15}{'MCC':>9}")
    print("-" * (width + 46))
    for r in sorted(outcome.results, key=lambda r: r.cv_mean["f1_macro"], reverse=True):
        marker = " *" if r is best else "  "
        print(
            f"{r.label:<{width}}"
            f"{r.cv_mean['f1_macro']:>11.4f} ±{r.cv_std['f1_macro']:.3f}"
            f"{r.holdout['f1_macro']:>15.4f}{r.holdout['mcc']:>9.4f}{marker}"
        )
    print("=" * (width + 46))
    for comparison in outcome.comparisons:
        verdict = "significant" if comparison["significant"] else "not significant"
        print(
            f"  {comparison['best']} vs {comparison['versus']}: "
            f"Δ={comparison['mean_difference']:+.4f}, p={comparison['p_value']:.4f} ({verdict})"
        )
    print("\nArtefacts:")
    for name, path in outcome.artefacts.items():
        print(f"  {name:<14} {path}")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_options:
        print_options()
        return 0

    configure_logging(verbose=args.verbose)

    # No arguments at all: fall back to the guided flow, as the original did.
    interactive = args.interactive or (argv is None and len(sys.argv) == 1)
    if interactive and not args.config:
        from autoclf.interactive import configure_interactively

        config = configure_interactively(config_from_args(args))
    else:
        config = config_from_args(args)

    try:
        from autoclf.runner import prepare_run_dir

        run_dir = prepare_run_dir(config)
        configure_logging(run_dir=run_dir, verbose=args.verbose)
        config.run_name = run_dir.name
        outcome = run(config)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 2
    except KeyboardInterrupt:  # pragma: no cover
        LOGGER.warning("interrupted")
        return 130

    summarise(outcome)
    return 0
