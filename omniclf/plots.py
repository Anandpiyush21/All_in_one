"""Figures for the report.

House style, applied to every figure:

* one accent hue for single-series charts, a fixed eight-slot categorical order
  for multi-series ones (assigned by entity, never recycled by rank);
* a single-hue blue ramp for the confusion matrix, because a magnitude needs a
  sequential scale and never a rainbow;
* recessive grid and axes, thin marks, values printed directly on the marks so
  identity and magnitude never depend on colour alone.

The palette below was checked for colour-vision-deficiency separation
(worst adjacent pair delta-E 9.1, target >= 8) rather than picked by eye.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: reports are generated on servers too
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import roc_curve

#: Fixed categorical order -- slot i always means "the i-th model", so a
#: filtered chart never repaints the survivors.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
ACCENT = SERIES[0]
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e4e3df"
SURFACE = "#ffffff"

#: Single-hue sequential ramp (light -> dark) for magnitude encodings.
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "omniclf_blue", ["#eef5fd", "#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]
)


#: ``boxplot(vert=False)`` was deprecated in Matplotlib 3.11 in favour of
#: ``orientation=``; support both so the report renders on either.
_HORIZONTAL = (
    {"orientation": "horizontal"}
    if tuple(int(p) for p in matplotlib.__version__.split(".")[:2]) >= (3, 11)
    else {"vert": False}
)


def apply_style() -> None:
    """Install the house style into the global matplotlib rc."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_SOFT,
            "axes.titlecolor": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.titlepad": 12,
            "axes.labelsize": 9,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": INK_SOFT,
            "ytick.color": INK_SOFT,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "font.size": 9.5,
            "figure.dpi": 130,
            "savefig.bbox": "tight",
        }
    )


def _despine(ax, keep=("left", "bottom")) -> None:
    for side, spine in ax.spines.items():
        spine.set_visible(side in keep)


def class_distribution(distribution: dict[str, int], target: str, path: Path) -> Path:
    """Horizontal bar of the label distribution -- the imbalance in one look."""
    names = list(distribution)
    counts = [distribution[n] for n in names]
    total = sum(counts)
    fig, ax = plt.subplots(figsize=(7, 0.55 * len(names) + 1.8))
    ax.barh(names, counts, color=ACCENT, height=0.6)
    for y, c in enumerate(counts):
        ax.text(c + total * 0.01, y, f"{c:,} ({c / total:.1%})", va="center", fontsize=8.5, color=INK)
    ax.set_xlim(0, max(counts) * 1.22)
    ax.invert_yaxis()
    ax.set_xlabel("rows")
    ax.set_title(f"Class distribution — {target}")
    ax.grid(axis="y", visible=False)
    _despine(ax)
    fig.savefig(path)
    plt.close(fig)
    return path


def cv_distribution(results, metric: str, path: Path) -> Path:
    """Per-fold score spread. The spread is the point: a mean without it is noise."""
    ranked = sorted(results, key=lambda r: r.cv_mean[metric])  # best ends up on top
    labels = [r.label for r in ranked]
    data = [np.asarray(r.cv_scores[metric], dtype=float) for r in ranked]
    fig, ax = plt.subplots(figsize=(8, 0.62 * len(labels) + 2.0))
    box = ax.boxplot(
        data,
        widths=0.55,
        **_HORIZONTAL,
        patch_artist=True,
        medianprops={"color": INK, "linewidth": 1.6},
        whiskerprops={"color": INK_SOFT, "linewidth": 1.0},
        capprops={"color": INK_SOFT, "linewidth": 1.0},
        flierprops={"markersize": 3, "markerfacecolor": INK_SOFT, "markeredgecolor": "none"},
    )
    for patch in box["boxes"]:
        patch.set_facecolor("#cde2fb")
        patch.set_edgecolor(ACCENT)
        patch.set_linewidth(1.2)
    rng = np.random.default_rng(0)
    for i, scores in enumerate(data, start=1):
        ax.scatter(
            scores,
            i + rng.uniform(-0.12, 0.12, size=len(scores)),
            s=16,
            color=ACCENT,
            edgecolor=SURFACE,
            linewidth=0.8,
            zorder=3,
        )
        # Anchored to the axes, not the data, so a wide score range cannot
        # push the annotation off the canvas.
        ax.annotate(
            f"μ={scores.mean():.3f}",
            xy=(1.015, i),
            xycoords=("axes fraction", "data"),
            va="center",
            fontsize=8,
            color=INK_SOFT,
            annotation_clip=False,
        )
    ax.set_yticks(range(1, len(labels) + 1), labels)
    ax.set_xlabel(metric.replace("_", " "))
    ax.set_title(f"Cross-validated {metric.replace('_', ' ')} per fold")
    ax.grid(axis="y", visible=False)
    _despine(ax)
    fig.savefig(path)
    plt.close(fig)
    return path


def cv_vs_holdout(results, metric: str, path: Path) -> Path:
    """Cross-validation mean against the untouched hold-out score.

    A large gap between the two is the signature of an over-fitted or leaking
    pipeline, so the comparison belongs in every report.
    """
    ranked = sorted(results, key=lambda r: r.cv_mean[metric], reverse=True)
    labels = [r.label for r in ranked]
    y = np.arange(len(labels))
    cv_mean = [r.cv_mean[metric] for r in ranked]
    cv_std = [r.cv_std[metric] for r in ranked]
    holdout = [r.holdout.get(metric, np.nan) for r in ranked]

    fig, ax = plt.subplots(figsize=(8, 0.62 * len(labels) + 2.2))
    ax.barh(y, cv_mean, height=0.46, color=ACCENT, label="cross-validation (mean ± sd)")
    ax.errorbar(cv_mean, y, xerr=cv_std, fmt="none", ecolor=INK_SOFT, elinewidth=1, capsize=3)
    ax.scatter(
        holdout, y, s=54, marker="D", color=SERIES[1], edgecolor=SURFACE, linewidth=1.2,
        zorder=4, label="hold-out",
    )
    for yi, (c, h) in enumerate(zip(cv_mean, holdout, strict=True)):
        edge = max(c + (cv_std[yi] or 0), h if np.isfinite(h) else 0)
        ax.text(edge + 0.025, yi, f"{c:.3f} / {h:.3f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, min(1.0, max(max(cv_mean), np.nanmax(holdout))) * 1.35)
    ax.set_xlabel(metric.replace("_", " "))
    ax.set_title(f"Generalisation check — {metric.replace('_', ' ')}")
    # Below the axes: inside the plot it would sit on the lowest bar, above it
    # would collide with the title.
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.16), ncol=2)
    ax.grid(axis="y", visible=False)
    _despine(ax)
    fig.savefig(path)
    plt.close(fig)
    return path


def confusion(matrix: np.ndarray, class_names: list[str], title: str, path: Path) -> Path:
    """Row-normalised confusion matrix; counts printed so colour is never the only channel."""
    totals = matrix.sum(axis=1, keepdims=True)
    normalised = np.divide(matrix, np.maximum(totals, 1))
    size = 1.0 + 0.72 * len(class_names)
    fig, ax = plt.subplots(figsize=(max(size + 1.6, 5.2), max(size, 4.0)))
    im = ax.imshow(normalised, cmap=SEQUENTIAL, vmin=0, vmax=1)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j, i, f"{matrix[i, j]:,}\n{normalised[i, j]:.1%}",
                ha="center", va="center", fontsize=8.5,
                color="#ffffff" if normalised[i, j] > 0.62 else INK,
            )
    ax.set_xticks(range(len(class_names)), class_names, rotation=30, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_title(f"Confusion matrix (hold-out) — {title}")
    ax.grid(visible=False)
    _despine(ax, keep=())
    fig.colorbar(im, ax=ax, shrink=0.78, label="share of actual class")
    fig.savefig(path)
    plt.close(fig)
    return path


def roc(results, X_test, y_test, path: Path) -> Path | None:
    """Binary ROC curves, one line per model, in fixed slot order."""
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    drawn = 0
    for i, result in enumerate(results):
        estimator = result.estimator
        if estimator is None or not hasattr(estimator, "predict_proba"):
            continue
        try:
            scores = estimator.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, scores)
        except (ValueError, IndexError):
            continue
        auc = result.holdout.get("roc_auc", float("nan"))
        ax.plot(
            fpr, tpr, linewidth=2, color=SERIES[i % len(SERIES)],
            label=f"{result.label} (AUC {auc:.3f})",
        )
        drawn += 1
    if not drawn:
        plt.close(fig)
        return None
    ax.plot([0, 1], [0, 1], linewidth=1, linestyle="--", color=INK_SOFT, label="chance")
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.set_title("ROC curves (hold-out)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right")
    _despine(ax)
    fig.savefig(path)
    plt.close(fig)
    return path


def importances(ranked, title: str, path: Path, top_n: int = 12) -> Path | None:
    """Permutation importance with its own uncertainty, largest first."""
    ranked = [r for r in ranked[:top_n]]
    if not ranked:
        return None
    names = [r[0] if len(r[0]) <= 46 else r[0][:43] + "…" for r in ranked]
    means = [r[1] for r in ranked]
    errs = [r[2] for r in ranked]
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8.4, 0.42 * len(names) + 2.0))
    ax.barh(y, means, height=0.6, color=ACCENT)
    ax.errorbar(means, y, xerr=errs, fmt="none", ecolor=INK_SOFT, elinewidth=1, capsize=3)
    span = max(max(means), 1e-6)
    for yi, (m, e) in enumerate(zip(means, errs, strict=True)):
        ax.text(m + e + span * 0.04, yi, f"{m:+.4f}", va="center", fontsize=8, color=INK)
    ax.set_xlim(right=(max(m + e for m, e in zip(means, errs, strict=True)) + span * 0.22))
    ax.axvline(0, color=INK_SOFT, linewidth=1)
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlabel("drop in macro F1 when the column is shuffled")
    ax.set_title(f"Permutation importance — {title}")
    ax.grid(axis="y", visible=False)
    _despine(ax)
    fig.savefig(path)
    plt.close(fig)
    return path
