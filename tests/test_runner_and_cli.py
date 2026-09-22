"""End-to-end: a run must produce a complete, self-describing directory."""

import json

import pytest

from omniclf.cli import build_parser, config_from_args, main
from omniclf.config import RunConfig
from omniclf.runner import run


@pytest.fixture(scope="module")
def completed(tmp_path_factory, toy_csv):
    """One full run, shared by the artefact tests -- it writes a PDF, so it is
    the slowest fixture in the suite and is deliberately built only once."""
    config = RunConfig(
        dataset=toy_csv,
        target_column="label",
        n_features=4,
        n_splits=3,
        models=["logistic_regression", "naive_bayes"],
        permutation_importance=False,
        output_dir=str(tmp_path_factory.mktemp("runs")),
        run_name="e2e",
        n_jobs=1,
    )
    return run(config), config


def test_run_writes_every_artefact(completed):
    outcome, _ = completed
    for name in ("config", "results_json", "summary_csv", "plots_pdf", "report_pdf"):
        path = outcome.artefacts[name]
        assert path.exists(), name
        assert path.stat().st_size > 0, name


def test_report_pdf_is_a_real_pdf(completed):
    outcome, _ = completed
    assert outcome.artefacts["report_pdf"].read_bytes().startswith(b"%PDF")


def test_results_json_is_complete(completed):
    outcome, _ = completed
    payload = json.loads(outcome.artefacts["results_json"].read_text())
    assert payload["config"]["models"] == ["logistic_regression", "naive_bayes"]
    assert len(payload["results"]) == 2
    assert payload["dataset"]["classes"] == ["no", "yes"]
    assert payload["environment"]["scikit_learn"]


def test_run_is_reproducible_from_its_own_config(completed, tmp_path):
    outcome, _ = completed
    replay = RunConfig.load(outcome.artefacts["config"])
    replay.output_dir = str(tmp_path / "replay")
    replay.run_name = "replay"
    replay.make_pdf = replay.make_plots = False
    second = run(replay)
    for a, b in zip(outcome.results, second.results, strict=True):
        assert a.cv_mean["f1_macro"] == pytest.approx(b.cv_mean["f1_macro"])
        assert a.holdout["mcc"] == pytest.approx(b.holdout["mcc"])


def test_best_is_chosen_by_macro_f1(completed):
    outcome, _ = completed
    assert outcome.best.cv_mean["f1_macro"] == max(
        r.cv_mean["f1_macro"] for r in outcome.results
    )


def test_group_cv_is_refused_rather_than_crashing(toy_config):
    toy_config.cv = "group_kfold"
    with pytest.raises(ValueError, match="grouping column"):
        run(toy_config)


# -- CLI ---------------------------------------------------------------------


def test_flags_map_onto_the_config():
    args = build_parser().parse_args(
        ["--scaler", "robust", "--models", "svm", "knn", "--n-splits", "7", "--tune"]
    )
    config = config_from_args(args)
    assert (config.scaler, config.models, config.n_splits, config.tune) == (
        "robust", ["svm", "knn"], 7, True,
    )


def test_all_models_expands_the_registry():
    from omniclf.models import MODELS

    config = config_from_args(build_parser().parse_args(["--all-models"]))
    assert config.models == sorted(MODELS)


def test_target_accepts_a_name_or_an_index():
    assert config_from_args(build_parser().parse_args(["--target", "-1"])).target_column == -1
    assert config_from_args(build_parser().parse_args(["--target", "y"])).target_column == "y"


def test_class_weighting_is_on_unless_disabled():
    assert config_from_args(build_parser().parse_args([])).class_weight_balanced
    assert not config_from_args(
        build_parser().parse_args(["--no-class-weight"])
    ).class_weight_balanced


def test_cli_runs_end_to_end(toy_csv, tmp_path):
    code = main(
        [
            "--dataset", toy_csv, "--target", "label",
            "--models", "naive_bayes", "--n-splits", "3", "--n-features", "4",
            "--no-plots", "--no-pdf", "--no-importance",
            "--output-dir", str(tmp_path / "cli"), "--run-name", "run1", "--n-jobs", "1",
        ]
    )
    assert code == 0
    assert (tmp_path / "cli" / "run1" / "results.json").exists()


def test_cli_reports_a_missing_dataset_without_a_traceback(tmp_path, capsys):
    assert main(["--dataset", "nope.csv", "--output-dir", str(tmp_path), "--no-plots", "--no-pdf"]) == 2


def test_list_options_exits_cleanly(capsys):
    assert main(["--list-options"]) == 0
    assert "Scalers" in capsys.readouterr().out
