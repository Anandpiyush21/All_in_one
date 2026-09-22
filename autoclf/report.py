"""Artefact generation: PDF report, figure sheet, JSON and CSV.

The PDF is written with ReportLab's Platypus flowables rather than absolute
canvas coordinates. The original version of this project positioned every
string by hand, which made the layout collapse as soon as the number of folds
or metrics changed; flowables paginate themselves.
"""

from __future__ import annotations

import csv
import json
import logging
import platform
from datetime import datetime
from pathlib import Path
from typing import Any

from matplotlib.backends.backend_pdf import PdfPages
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from autoclf import __version__, plots
from autoclf.config import RunConfig
from autoclf.cross_val import describe as describe_cv
from autoclf.data import Dataset
from autoclf.evaluate import CV_METRICS, ModelResult, compare_models
from autoclf.feature_selection import describe as describe_selector
from autoclf.preprocessing import describe as describe_scaler

LOGGER = logging.getLogger(__name__)

ACCENT = colors.HexColor("#2a78d6")
INK = colors.HexColor("#0b0b0b")
INK_SOFT = colors.HexColor("#52514e")
RULE = colors.HexColor("#e4e3df")
BAND = colors.HexColor("#f5f7fa")


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title", parent=base["Title"], fontSize=19, leading=23, textColor=INK, alignment=TA_LEFT
        ),
        "subtitle": ParagraphStyle(
            "subtitle", parent=base["Normal"], fontSize=9.5, leading=13, textColor=INK_SOFT
        ),
        "h2": ParagraphStyle(
            "h2", parent=base["Heading2"], fontSize=12.5, leading=16, spaceBefore=14,
            spaceAfter=6, textColor=ACCENT,
        ),
        "h3": ParagraphStyle(
            "h3", parent=base["Heading3"], fontSize=10.5, leading=14, spaceBefore=10,
            spaceAfter=4, textColor=INK,
        ),
        "body": ParagraphStyle(
            "body", parent=base["BodyText"], fontSize=9, leading=13, textColor=INK
        ),
        "mono": ParagraphStyle(
            "mono", parent=base["Code"], fontSize=7.4, leading=9.2, textColor=INK
        ),
    }


def _table(rows: list[list[Any]], widths: list[float], highlight_first_row: bool = True) -> Table:
    table = Table(rows, colWidths=widths, repeatRows=1 if highlight_first_row else 0)
    style = [
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.8),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("LINEBELOW", (0, 0), (-1, -2), 0.4, RULE),
    ]
    if highlight_first_row:
        style += [
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("BACKGROUND", (0, 0), (-1, 0), BAND),
            ("VALIGN", (0, 0), (-1, 0), "BOTTOM"),
            ("LINEBELOW", (0, 0), (-1, 0), 0.8, ACCENT),
        ]
    table.setStyle(TableStyle(style))
    return table


def _kv_table(pairs: list[tuple[str, Any]], width: float) -> Table:
    rows = [[Paragraph(f"<b>{k}</b>", _styles()["body"]), str(v)] for k, v in pairs]
    table = Table(rows, colWidths=[width * 0.34, width * 0.66])
    table.setStyle(
        TableStyle(
            [
                ("FONTSIZE", (0, 0), (-1, -1), 8.2),
                ("TEXTCOLOR", (0, 0), (-1, -1), INK),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("LINEBELOW", (0, 0), (-1, -2), 0.4, RULE),
            ]
        )
    )
    return table


def _footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(INK_SOFT)
    canvas.drawString(20 * mm, 12 * mm, f"autoclf v{__version__} — automated classification report")
    canvas.drawRightString(A4[0] - 20 * mm, 12 * mm, f"page {doc.page}")
    canvas.setStrokeColor(RULE)
    canvas.line(20 * mm, 15 * mm, A4[0] - 20 * mm, 15 * mm)
    canvas.restoreState()


def write_pdf(
    path: Path,
    config: RunConfig,
    dataset: Dataset,
    results: list[ModelResult],
    figures: list[Path],
    comparisons: list[dict[str, Any]],
) -> Path:
    """Render the full experiment report."""
    st = _styles()
    doc = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=18 * mm,
        bottomMargin=20 * mm,
        title="Automated classification report",
        author="autoclf",
    )
    usable = doc.width
    best = max(results, key=lambda r: r.cv_mean["f1_macro"])
    story: list[Any] = []

    # -- cover / summary --------------------------------------------------
    story += [
        Paragraph("Automated Classification Report", st["title"]),
        Spacer(1, 3),
        Paragraph(
            f"dataset <b>{Path(config.dataset).name}</b> &nbsp;·&nbsp; target "
            f"<b>{dataset.target_name}</b> &nbsp;·&nbsp; generated "
            f"{datetime.now():%Y-%m-%d %H:%M}",
            st["subtitle"],
        ),
        Spacer(1, 12),
        Paragraph("1. Experimental setup", st["h2"]),
        _kv_table(
            [
                ("Rows (raw / usable)", f"{dataset.n_rows_raw:,} / {dataset.n_rows_used:,}"),
                ("Train / hold-out", f"{len(dataset.X_train):,} / {len(dataset.X_test):,}"),
                ("Features in", f"{len(dataset.numeric_columns)} numeric + "
                                f"{len(dataset.categorical_columns)} categorical"),
                ("Classes", f"{dataset.n_classes} ({', '.join(dataset.class_names)})"),
                ("Imbalance ratio", f"{dataset.imbalance_ratio():.2f} : 1"),
                ("Scaling", describe_scaler(config.scaler)),
                ("Feature selection", f"{describe_selector(config.selector)} "
                                      f"(k = {config.n_features})"),
                ("Validation", f"{describe_cv(config.cv)}, {config.n_splits} splits"),
                ("Hyper-parameter search", "grid search inside training folds"
                 if config.tune else "registry defaults"),
                ("Class weighting", "balanced" if config.class_weight_balanced else "none"),
                ("Random seed", config.random_state),
                ("Environment", f"Python {platform.python_version()} on {platform.system()}"),
            ],
            usable,
        ),
        Spacer(1, 8),
        Paragraph(
            "Every preprocessing step — imputation, encoding, scaling and feature selection — "
            "is refitted inside each training fold, so no statistic derived from validation or "
            "hold-out rows can reach the model. Cross-validation scores are therefore unbiased "
            "estimates of out-of-sample performance, and the hold-out column is an independent "
            "confirmation on data touched exactly once.",
            st["body"],
        ),
    ]

    # -- results table ----------------------------------------------------
    # Header labels are wrapped in Paragraphs: plain strings do not break, and
    # "balanced_accuracy" is wide enough to run into its neighbour.
    metric_keys = list(CV_METRICS)
    head_style = ParagraphStyle(
        "th", parent=st["body"], fontName="Helvetica-Bold", fontSize=7.4, leading=8.6
    )
    short = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced accuracy",
        "precision_macro": "precision (macro)",
        "recall_macro": "recall (macro)",
        "f1_macro": "F1 (macro)",
        "mcc": "MCC",
    }
    header = (
        [Paragraph("Model", head_style)]
        + [Paragraph(short[m], head_style) for m in metric_keys]
        + [Paragraph("hold-out F1", head_style), Paragraph("fit (s)", head_style)]
    )
    rows = [header]
    cell_style = ParagraphStyle("td", parent=st["body"], fontSize=7.8, leading=9.2)
    for r in sorted(results, key=lambda r: r.cv_mean["f1_macro"], reverse=True):
        rows.append(
            # A Paragraph so long model names wrap inside the column instead of
            # spilling over the next one.
            [Paragraph(r.label, cell_style)]
            + [f"{r.cv_mean[m]:.3f}\n±{r.cv_std[m]:.3f}" for m in metric_keys]
            + [f"{r.holdout['f1_macro']:.3f}", f"{r.fit_seconds:.0f}"]
        )
    widths = (
        [usable * 0.205]
        + [usable * 0.104] * len(metric_keys)
        + [usable * 0.095, usable * 0.076]
    )
    story += [
        Paragraph("2. Cross-validated results", st["h2"]),
        Paragraph(
            "Mean ± standard deviation across folds. On an imbalanced target, balanced accuracy, "
            "macro F1 and MCC are the informative columns; plain accuracy can be high for a model "
            "that never predicts the minority class.",
            st["body"],
        ),
        Spacer(1, 6),
        _table(rows, widths),
        Spacer(1, 10),
        Paragraph(
            f"<b>Selected model: {best.label}</b> — macro F1 "
            f"{best.cv_mean['f1_macro']:.3f} ± {best.cv_std['f1_macro']:.3f} in cross-validation, "
            f"{best.holdout['f1_macro']:.3f} on the hold-out set "
            f"(balanced accuracy {best.holdout['balanced_accuracy']:.3f}, "
            f"MCC {best.holdout['mcc']:.3f}"
            + (f", ROC-AUC {best.holdout['roc_auc']:.3f}" if "roc_auc" in best.holdout else "")
            + ").",
            st["body"],
        ),
    ]

    if comparisons:
        comp_rows = [["Best model", "Compared with", "Δ macro F1", "p (paired t)", "significant"]]
        for c in comparisons:
            comp_rows.append(
                [c["best"], c["versus"], f"{c['mean_difference']:+.4f}",
                 f"{c['p_value']:.4f}", "yes" if c["significant"] else "no"]
            )
        story += [KeepTogether([
            Paragraph("3. Statistical comparison", st["h3"]),
            Paragraph(
                "Paired t-test on per-fold macro F1 (identical splits for every model). "
                f"With {config.n_splits} folds the test has little power, so a non-significant "
                "row means “not distinguishable here”, not “equivalent”.",
                st["body"],
            ),
            Spacer(1, 6),
            _table(comp_rows, [usable * 0.24, usable * 0.24, usable * 0.16, usable * 0.18, usable * 0.18]),
        ])]

    if best.best_params:
        story += [
            Paragraph("Selected hyper-parameters", st["h3"]),
            _kv_table(list(best.best_params.items()), usable),
        ]

    # -- per-class detail -------------------------------------------------
    story += [
        Spacer(1, 4),
        Paragraph("4. Best model in detail", st["h2"]),
        KeepTogether([
            Paragraph(f"Per-class hold-out performance — {best.label}", st["h3"]),
            Paragraph(f"<pre>{best.report_text}</pre>", st["mono"]),
        ]),
    ]
    if best.importances:
        imp_rows = [["Rank", "Feature", "Δ macro F1 when shuffled", "sd"]]
        for i, (feature, mean, sd) in enumerate(best.importances[:12], start=1):
            imp_rows.append([str(i), feature, f"{mean:+.4f}", f"{sd:.4f}"])
        story += [KeepTogether([
            Paragraph("Permutation importance (hold-out)", st["h3"]),
            Paragraph(
                "Each input column is shuffled in turn and the loss of macro F1 recorded. "
                "Unlike impurity-based importance this is measured on unseen data and is not "
                "biased towards high-cardinality columns.",
                st["body"],
            ),
            Spacer(1, 6),
            _table(imp_rows, [usable * 0.08, usable * 0.56, usable * 0.22, usable * 0.14]),
        ])]
    if dataset.dropped_columns:
        story += [
            Paragraph("Columns removed during loading", st["h3"]),
            Paragraph(", ".join(dataset.dropped_columns), st["body"]),
        ]

    # -- figures ----------------------------------------------------------
    if figures:
        story.append(PageBreak())
        story.append(Paragraph("5. Figures", st["h2"]))
        for fig_path in figures:
            story += [_fitted_image(fig_path, usable), Spacer(1, 10)]

    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    LOGGER.info("report written to %s", path)
    return path


def _fitted_image(path: Path, max_width: float) -> Image:
    """Scale a PNG to the text width while preserving its aspect ratio."""
    from PIL import Image as PILImage  # pillow ships with matplotlib

    with PILImage.open(path) as img:
        w, h = img.size
    scale = min(max_width / w, 1.0)
    return Image(str(path), width=w * scale, height=h * scale)


def write_figure_sheet(figures: list[Path], path: Path) -> Path:
    """Bundle the PNG figures into a standalone ``plots.pdf``."""
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    with PdfPages(path) as pdf:
        for fig_path in figures:
            image = mpimg.imread(fig_path)
            h, w = image.shape[:2]
            fig = plt.figure(figsize=(8.27, 8.27 * h / w))
            ax = fig.add_axes((0, 0, 1, 1))
            ax.imshow(image)
            ax.axis("off")
            pdf.savefig(fig, dpi=200)
            plt.close(fig)
    return path


def write_json(path: Path, config: RunConfig, dataset: Dataset, results: list[ModelResult]) -> Path:
    """Machine-readable results, for downstream analysis or a thesis appendix."""
    import sklearn

    payload = {
        "autoclf_version": __version__,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "scikit_learn": sklearn.__version__,
        },
        "config": config.to_dict(),
        "dataset": {
            "path": str(Path(config.dataset).resolve()),
            "target": dataset.target_name,
            "rows_raw": dataset.n_rows_raw,
            "rows_used": dataset.n_rows_used,
            "n_train": len(dataset.X_train),
            "n_holdout": len(dataset.X_test),
            "numeric_columns": dataset.numeric_columns,
            "categorical_columns": dataset.categorical_columns,
            "dropped_columns": dataset.dropped_columns,
            "classes": dataset.class_names,
            "class_distribution": dataset.class_distribution(),
            "imbalance_ratio": round(dataset.imbalance_ratio(), 4),
        },
        "results": [r.to_dict() for r in results],
        "comparisons": compare_models(results),
    }
    path.write_text(json.dumps(payload, indent=2))
    return path


def write_csv(path: Path, results: list[ModelResult]) -> Path:
    """Flat summary table, one row per model."""
    rows = [r.summary_row() for r in sorted(results, key=lambda r: r.cv_mean["f1_macro"], reverse=True)]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def build_figures(config: RunConfig, dataset: Dataset, results: list[ModelResult], out: Path) -> list[Path]:
    """Render every figure the report uses and return their paths in order."""
    plots.apply_style()
    out.mkdir(parents=True, exist_ok=True)
    best = max(results, key=lambda r: r.cv_mean["f1_macro"])
    figures: list[Path] = [
        plots.class_distribution(dataset.class_distribution(), dataset.target_name, out / "01_class_distribution.png"),
        plots.cv_distribution(results, "f1_macro", out / "02_cv_f1_macro.png"),
        plots.cv_vs_holdout(results, "f1_macro", out / "03_cv_vs_holdout.png"),
        plots.confusion(best.confusion, dataset.class_names, best.label, out / "04_confusion_matrix.png"),
    ]
    if dataset.is_binary:
        roc_path = plots.roc(results, dataset.X_test, dataset.y_test, out / "05_roc_curves.png")
        if roc_path:
            figures.append(roc_path)
    imp_path = plots.importances(best.importances, best.label, out / "06_permutation_importance.png")
    if imp_path:
        figures.append(imp_path)
    return figures
