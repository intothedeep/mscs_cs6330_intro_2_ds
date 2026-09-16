"""Figures for HW1 Question 1 (dataset summary + "what I learned").

Chart forms follow lecture 5 (Stat and Visualization) and lecture 6 (Data
Distribution):
  - histogram + normal overlay   -> L5 s12/s14, L6 s4-s7
  - box plot grouped by category -> L5 s20/s21 ("Comparing Samples with Box Plots")

Run:  .venv/bin/python hw1/code/p1_figures.py
"""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # headless: write PNG files, never open a GUI window
import matplotlib.pyplot as plt

__all__ = ["build_all_figures"]

DATA_PATH = Path(__file__).parent.parent / "00_doc" / "01_Fast foods - Nutritional information.xlsx"
FIG_DIR = Path(__file__).parent.parent / "figures"

NUMERIC_COLS = ["serving_size", "calories", "sodium", "sugars"]
AXIS_LABELS = {
    "serving_size": "Serving size (g)",
    "calories": "Calories (kcal)",
    "sodium": "Sodium (mg)",
    "sugars": "Sugars (g)",
}

# Validated categorical slots 1-2 (see dataviz palette); only two series appear
# per chart, so no further slots are needed.
SERIES_BLUE = "#2a78d6"
SERIES_ORANGE = "#eb6834"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e4e1"


def _style_axes(ax: plt.Axes) -> None:
    """Recessive grid and axes so the data marks carry the chart."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_SECONDARY, labelsize=9, length=0)


def _normal_pdf(x: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Normal PDF from mean/std only — avoids a scipy dependency (L6 s10)."""
    return np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))


def create_histograms(df: pd.DataFrame) -> plt.Figure:
    """2x2 histograms with a fitted normal curve: does each variable look normal?"""
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))

    for ax, col in zip(axes.flat, NUMERIC_COLS):
        values = df[col].to_numpy(dtype=float)
        mean, std, skew = values.mean(), values.std(ddof=1), df[col].skew()

        ax.hist(values, bins=20, density=True, color=SERIES_BLUE, edgecolor="white", linewidth=0.8)
        grid = np.linspace(values.min(), values.max(), 300)
        ax.plot(grid, _normal_pdf(grid, mean, std), color=SERIES_ORANGE, linewidth=2)

        if col == "serving_size":
            # A flat uniform PDF is the honest reference for this one (L6 s11):
            # its skew is near zero but the shape is flat, not bell-shaped.
            span = values.max() - values.min()
            ax.plot(grid, np.full_like(grid, 1 / span), color=INK_SECONDARY,
                    linewidth=1.5, linestyle=":")
            ax.text(values.max(), 1 / span, "uniform  ", color=INK_SECONDARY,
                    fontsize=8, ha="right", va="bottom")

        ax.axvline(mean, color=INK_SECONDARY, linewidth=1, linestyle="--")
        # Direct label beats a legend entry for a single reference line.
        ax.text(
            mean, ax.get_ylim()[1] * 0.96, f"  mean {mean:,.0f}",
            color=INK_SECONDARY, fontsize=8, va="top",
        )
        ax.set_title(
            f"{AXIS_LABELS[col]}  ·  skew {skew:.2f}",
            color=INK_PRIMARY, fontsize=11, loc="left",
        )
        ax.set_xlabel(AXIS_LABELS[col], color=INK_SECONDARY, fontsize=9)
        ax.set_ylabel("Density", color=INK_SECONDARY, fontsize=9)
        _style_axes(ax)

    # One legend for the whole figure: both series appear in every panel.
    fig.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color=SERIES_BLUE, label="Observed data"),
            plt.Line2D([0], [0], color=SERIES_ORANGE, linewidth=2, label="Fitted normal curve"),
        ],
        loc="lower center", ncol=2, frameon=False, fontsize=9, labelcolor=INK_SECONDARY,
    )
    fig.suptitle(
        "Distribution of each nutrition variable vs. a fitted normal curve (n = 126)",
        color=INK_PRIMARY, fontsize=13, x=0.01, ha="left",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    return fig


def create_boxplots_by_type(df: pd.DataFrame) -> plt.Figure:
    """All four quantitative variables by food type — why the pooled shapes differ."""
    order = df.groupby("type")["calories"].median().sort_values().index.tolist()
    fig, axes = plt.subplots(1, len(NUMERIC_COLS), figsize=(20, 5.5), sharey=True)

    for ax, col in zip(axes, NUMERIC_COLS):
        groups = [df.loc[df["type"] == t, col].to_numpy(dtype=float) for t in order]
        bp = ax.boxplot(
            groups, orientation="horizontal", tick_labels=order, widths=0.55, patch_artist=True,
            flierprops={"marker": "o", "markersize": 4, "markerfacecolor": SERIES_ORANGE,
                        "markeredgecolor": "none"},
            medianprops={"color": "white", "linewidth": 1.5},
            whiskerprops={"color": INK_SECONDARY, "linewidth": 1},
            capprops={"color": INK_SECONDARY, "linewidth": 1},
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(SERIES_BLUE)
            patch.set_edgecolor("white")  # 2px surface gap between adjacent fills
            patch.set_linewidth(1.5)
        ax.set_title(AXIS_LABELS[col], color=INK_PRIMARY, fontsize=11, loc="left")
        ax.set_xlabel(AXIS_LABELS[col], color=INK_SECONDARY, fontsize=9)
        _style_axes(ax)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=GRID, linewidth=0.8)

    # Anchor on the Milkshake row itself; +1 because boxplot positions are 1-based.
    shake_row = order.index("Milkshake") + 1
    axes[NUMERIC_COLS.index("sugars")].annotate(
        "Milkshakes are the\nsugar outlier group",
        xy=(64, shake_row), xytext=(26, shake_row - 1.5),
        color=INK_SECONDARY, fontsize=9,
        arrowprops={"arrowstyle": "->", "color": INK_SECONDARY, "linewidth": 0.8},
    )
    fig.suptitle(
        "Serving size, calories, sodium and sugars by food type "
        "(box = IQR, whisker = 1.5 x IQR, dots = outliers)",
        color=INK_PRIMARY, fontsize=13, x=0.01, ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def create_sugars_split(df: pd.DataFrame) -> plt.Figure:
    """The sugars histogram is bimodal: milkshakes vs. everything else."""
    is_shake = df["type"] == "Milkshake"
    bins = np.linspace(0, df["sugars"].max(), 25)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(df.loc[~is_shake, "sugars"], bins=bins, color=SERIES_BLUE,
            edgecolor="white", linewidth=0.8,
            label=f"All other items (n = {(~is_shake).sum()})")
    ax.hist(df.loc[is_shake, "sugars"], bins=bins, color=SERIES_ORANGE,
            edgecolor="white", linewidth=0.8,
            label=f"Milkshakes (n = {is_shake.sum()})")

    ax.set_title(
        "Sugars is bimodal, not normal — the right tail is entirely milkshakes",
        color=INK_PRIMARY, fontsize=13, loc="left",
    )
    ax.set_xlabel(AXIS_LABELS["sugars"], color=INK_SECONDARY, fontsize=9)
    ax.set_ylabel("Number of items", color=INK_SECONDARY, fontsize=9)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_SECONDARY)
    _style_axes(ax)
    fig.tight_layout()
    return fig


def build_all_figures() -> list[Path]:
    """Render every figure to hw1/figures/ and return the written paths."""
    df = pd.read_excel(DATA_PATH)
    FIG_DIR.mkdir(exist_ok=True)

    builders = {
        "fig1_histograms.png": create_histograms,
        "fig2_boxplot_by_type.png": create_boxplots_by_type,
        "fig3_sugars_bimodal.png": create_sugars_split,
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
