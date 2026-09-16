"""Figures that explain the two summary statistics HW1 Q1 asks about: IQR and skewness.

Separate from p1_figures.py by responsibility: that module describes the *dataset*
(what the variables look like); this one explains the *statistics* (how to read the
numbers in the descriptive-statistics table). Style helpers are imported, not copied.

Chart choice, and why:
  q1_fig1  histogram + mean/median lines  -> skewness is the GAP between two lines
  q1_fig2  box-plot anatomy               -> the box IS the IQR; the fence is derived from it
  q1_fig3  category-split histogram       -> WHY skew is 2.71 (two populations, not a tail)

Run:  .venv/bin/python hw1/code/p1_stats_figures.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from p1_figures import (
    AXIS_LABELS,
    DATA_PATH,
    FIG_DIR,
    GRID,
    INK_PRIMARY,
    INK_SECONDARY,
    SERIES_BLUE,
    SERIES_ORANGE,
    _style_axes,
)

import matplotlib.pyplot as plt  # isort: skip  (p1_figures already set the Agg backend)

__all__ = ["build_all_figures"]

# Ascending skew, so the mean/median gap visibly widens across the four panels.
SKEW_ORDER = ["serving_size", "sodium", "calories", "sugars"]


def create_skewness_panels(df: pd.DataFrame) -> plt.Figure:
    """Skewness made visible: the horizontal gap between the median and the mean."""
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5))

    for ax, col in zip(axes.flat, SKEW_ORDER):
        values = df[col].to_numpy(dtype=float)
        mean, median = values.mean(), np.median(values)
        std, skew = values.std(ddof=1), df[col].skew()

        ax.hist(values, bins=20, color=SERIES_BLUE, edgecolor="white", linewidth=0.8)
        top = ax.get_ylim()[1]

        # The shaded span between the two lines is the whole point of the figure:
        # it is what "skew" looks like before it is collapsed into one number.
        ax.axvspan(median, mean, color=SERIES_ORANGE, alpha=0.18, linewidth=0)
        ax.axvline(median, color=INK_PRIMARY, linewidth=2)
        ax.axvline(mean, color=SERIES_ORANGE, linewidth=2, linestyle="--")

        ax.text(median, top * 0.99, f"median {median:,.0f} ", color=INK_PRIMARY,
                fontsize=8, ha="right", va="top")
        ax.text(mean, top * 0.99, f" mean {mean:,.0f}", color=SERIES_ORANGE,
                fontsize=8, ha="left", va="top")

        # Reported in SD units, never raw units: a raw gap is not comparable across
        # variables (sodium's 43.75 mg gap is SMALLER in shape terms than sugars' 6.29 g).
        ax.text(0.98, 0.62, f"gap = {(mean - median) / std:.2f} SD", transform=ax.transAxes,
                color=INK_SECONDARY, fontsize=9, ha="right")

        verdict = "use median + IQR" if abs(skew) > 1 else "mean + SD is fine"
        ax.set_title(f"{AXIS_LABELS[col]}   skew {skew:.2f}  ->  {verdict}",
                     color=INK_PRIMARY, fontsize=11, loc="left")
        ax.set_xlabel(AXIS_LABELS[col], color=INK_SECONDARY, fontsize=9)
        ax.set_ylabel("Number of items", color=INK_SECONDARY, fontsize=9)
        _style_axes(ax)

    fig.legend(
        handles=[
            plt.Line2D([0], [0], color=INK_PRIMARY, linewidth=2, label="Median (resistant)"),
            plt.Line2D([0], [0], color=SERIES_ORANGE, linewidth=2, linestyle="--",
                       label="Mean (pulled toward the long tail)"),
        ],
        loc="lower center", ncol=2, frameon=False, fontsize=9, labelcolor=INK_SECONDARY,
    )
    fig.suptitle(
        "Skewness is the gap between the median and the mean — it widens as skew grows "
        "(0.24 -> 2.71)",
        color=INK_PRIMARY, fontsize=13, x=0.01, ha="left",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    return fig


def _draw_box(ax, values: np.ndarray) -> None:
    """The same box is drawn on both halves of the broken axis."""
    bp = ax.boxplot(
        values, orientation="horizontal", positions=[0], widths=0.40, patch_artist=True,
        flierprops={"marker": "o", "markersize": 6, "markerfacecolor": SERIES_ORANGE,
                    "markeredgecolor": "white"},
        medianprops={"color": "white", "linewidth": 2},
        whiskerprops={"color": INK_SECONDARY, "linewidth": 1.2},
        capprops={"color": INK_SECONDARY, "linewidth": 1.2},
    )
    bp["boxes"][0].set(facecolor=SERIES_BLUE, edgecolor="white", linewidth=1.5)


def create_iqr_anatomy(df: pd.DataFrame) -> plt.Figure:
    """Box-plot anatomy on sugars: where the IQR is, and what it decides.

    The x axis is BROKEN. Drawn to scale the box would occupy a fifth of the width
    (Q1-Q3 is 3-11 g while the outliers reach 93 g), leaving no room to label its
    parts - which is exactly what this figure exists to do.
    """
    values = df["sugars"].to_numpy(dtype=float)
    q1, median, q3 = np.percentile(values, [25, 50, 75])
    iqr = q3 - q1
    fence = q3 + 1.5 * iqr
    whisker = values[values <= fence].max()
    outliers = values[values > fence]

    fig, (ax, ax_out) = plt.subplots(
        1, 2, figsize=(12, 4.6), sharey=True,
        gridspec_kw={"width_ratios": [2.6, 1], "wspace": 0.06},
    )
    for axis in (ax, ax_out):
        _draw_box(axis, values)
        axis.set_ylim(-0.62, 0.58)
        axis.set_yticks([])
        _style_axes(axis)
        axis.grid(axis="y", visible=False)
        axis.grid(axis="x", color=GRID, linewidth=0.8)

    ax.set_xlim(-1.5, 26)
    ax_out.set_xlim(52, 97)
    ax.spines["right"].set_visible(False)
    ax_out.spines["left"].set_visible(False)
    # Slanted tick marks are the conventional "axis is broken here" signal.
    for axis, x in ((ax, 1.0), (ax_out, 0.0)):
        axis.plot([x, x], [0, 1], transform=axis.transAxes, clip_on=False,
                  color=GRID, linewidth=1.2, marker=[(-0.4, -1), (0.4, 1)],
                  markersize=8, markeredgecolor=INK_SECONDARY, linestyle="none")

    # Callouts alternate above/below and are pushed sideways so no two labels touch.
    for x, label, dx, dy, va in [
        (q1, f"Q1 = {q1:.0f}\n25% of items below", -18, -30, "top"),
        (median, f"median = {median:.0f}", 0, 34, "bottom"),
        (q3, f"Q3 = {q3:.0f}\n75% of items below", 22, -30, "top"),
        (whisker, f"whisker stops at {whisker:.0f}\nlast value inside the fence", 4, 34, "bottom"),
    ]:
        ax.annotate(label, xy=(x, 0.21 if va == "bottom" else -0.21), textcoords="offset points",
                    xytext=(dx, dy), ha="center", va=va, fontsize=9, color=INK_SECONDARY,
                    arrowprops={"arrowstyle": "-", "color": GRID, "linewidth": 1.2})

    # IQR bracket: the width of the blue box IS this number.
    ax.annotate("", xy=(q1, -0.50), xytext=(q3, -0.50),
                arrowprops={"arrowstyle": "<->", "color": SERIES_ORANGE, "linewidth": 1.6})
    ax.text(q3 + 1.2, -0.50, f"IQR = {q3:.0f} - {q1:.0f} = {iqr:.0f}", color=SERIES_ORANGE,
            fontsize=10, ha="left", va="center", fontweight="bold")

    ax.axvline(fence, color=INK_SECONDARY, linewidth=1.2, linestyle=":")
    # Set vertically alongside the line: the band above the whisker is the only
    # clear space left, and a horizontal label there would run into the whisker callout.
    ax.text(fence - 0.5, -0.06, f"fence = Q3 + 1.5 x IQR = {fence:.0f}",
            color=INK_SECONDARY, fontsize=9, ha="center", va="center", rotation=90)

    ax_out.text(0.5, 0.86,
                f"{len(outliers)} points past the fence\nevery one is a milkshake "
                f"({outliers.min():.0f}-{outliers.max():.0f} g)",
                transform=ax_out.transAxes, ha="center", va="top",
                fontsize=9, color=SERIES_ORANGE)
    ax_out.text(0.5, 0.06,
                f"SD = {values.std(ddof=1):.2f}  inflated by these 12\n"
                f"IQR = {iqr:.2f}  unaffected by them",
                transform=ax_out.transAxes, ha="center", va="bottom",
                fontsize=9, color=INK_SECONDARY)

    ax.set_xlabel(AXIS_LABELS["sugars"], color=INK_SECONDARY, fontsize=9)
    ax_out.set_xlabel("(outliers, axis broken)", color=INK_SECONDARY, fontsize=9)
    fig.suptitle("Anatomy of a box plot: the box is the IQR, and the IQR decides "
                 "which points are outliers (sugars, n = 126)",
                 color=INK_PRIMARY, fontsize=13, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def create_skew_cause(df: pd.DataFrame) -> plt.Figure:
    """Why sugars' skew is 2.71: a mixture of two populations, not one long tail."""
    is_shake = df["type"] == "Milkshake"
    values = df["sugars"]
    mean, median = values.mean(), values.median()
    bins = np.linspace(0, values.max(), 25)

    fig, ax = plt.subplots(figsize=(11.5, 5.5))
    ax.hist(values[~is_shake], bins=bins, color=SERIES_BLUE, edgecolor="white", linewidth=0.8,
            label=f"All other items (n = {(~is_shake).sum()})")
    ax.hist(values[is_shake], bins=bins, color=SERIES_ORANGE, edgecolor="white", linewidth=0.8,
            label=f"Milkshakes (n = {is_shake.sum()})")

    top = ax.get_ylim()[1]
    gap_lo, gap_hi = values[values < 30].max(), values[values > 30].min()
    ax.axvspan(gap_lo, gap_hi, color=GRID, alpha=0.7, linewidth=0, zorder=0)
    ax.text((gap_lo + gap_hi) / 2, top * 0.55,
            f"no item at all\nbetween {gap_lo:.0f} and {gap_hi:.0f} g",
            ha="center", va="center", fontsize=9, color=INK_SECONDARY)

    ax.axvline(median, color=INK_PRIMARY, linewidth=2)
    ax.axvline(mean, color=SERIES_ORANGE, linewidth=2, linestyle="--")
    ax.text(median, top * 0.99, f"median {median:.0f} ", ha="right", va="top",
            fontsize=9, color=INK_PRIMARY)
    ax.text(mean, top * 0.99, f" mean {mean:.2f}", ha="left", va="top",
            fontsize=9, color=SERIES_ORANGE)
    ax.annotate(
        f"{(values < mean).mean():.0%} of items sit BELOW the mean —\n"
        "the mean is not a typical value here",
        xy=(mean, top * 0.62), xytext=(30, top * 0.82), fontsize=9, color=SERIES_ORANGE,
        arrowprops={"arrowstyle": "->", "color": SERIES_ORANGE, "linewidth": 1},
    )

    ax.set_title("Skew 2.71 is not a long tail — it is two populations with a hole between them",
                 color=INK_PRIMARY, fontsize=13, loc="left")
    ax.set_xlabel(AXIS_LABELS["sugars"], color=INK_SECONDARY, fontsize=9)
    ax.set_ylabel("Number of items", color=INK_SECONDARY, fontsize=9)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY, loc="upper right",
              bbox_to_anchor=(1.0, 0.72))
    _style_axes(ax)
    fig.tight_layout()
    return fig


def build_all_figures() -> list[Path]:
    """Render the three Q1 statistics figures and return the written paths."""
    df = pd.read_excel(DATA_PATH)
    FIG_DIR.mkdir(exist_ok=True)

    builders = {
        "q1_fig1_skewness_mean_vs_median.png": create_skewness_panels,
        "q1_fig2_iqr_boxplot_anatomy.png": create_iqr_anatomy,
        "q1_fig3_skew_cause_two_populations.png": create_skew_cause,
    }
    written: list[Path] = []
    for name, build in builders.items():
        fig = build(df)
        path = FIG_DIR / name
        fig.savefig(path, dpi=150, facecolor="#fcfcfb")
        plt.close(fig)
        written.append(path)
    return written


if __name__ == "__main__":
    for path in build_all_figures():
        print("[][] wrote:", path)
