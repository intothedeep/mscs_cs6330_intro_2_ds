"""Deck 5 (Basic Statistics and Visualization) -> figures on the fast-food data.

One function per slide topic. Each returns the saved path.
"""

import numpy as np
import pandas as pd
from data import SHORT_LABEL, TYPE_ORDER, series_by_type
from matplotlib import pyplot as plt
from style import ACCENT, CATEGORICAL, INK, MUTED, SECOND, save

__all__ = ["build_all"]

RNG = np.random.default_rng(6330)


def _quantiles(df: pd.DataFrame) -> str:
    """Slides 3-8: quantiles are the cut lines that split sorted data into equal groups."""
    cal = df["calories"].sort_values().to_numpy()
    n = cal.size
    frac = np.arange(1, n + 1) / n  # sample fraction of each ordered value

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(frac, cal, color=ACCENT, lw=2)
    for q, name in [(0.25, "lower quartile"), (0.50, "median"), (0.75, "upper quartile")]:
        v = np.quantile(cal, q)
        ax.vlines(q, cal.min(), v, color=MUTED, lw=1, ls=":")
        ax.hlines(v, 0, q, color=MUTED, lw=1, ls=":")
        ax.plot([q], [v], "o", ms=8, color=SECOND, mec="white", mew=1.5, zorder=3)
        ax.annotate(f"{name}\n{int(q * 100)}% quantile = {v:.0f} cal",
                    (q, v), textcoords="offset points", xytext=(8, -26),
                    fontsize=9, color=INK)
    ax.set(xlabel="sample fraction below this value",
           ylabel="calories",
           title="Quantiles of calories: the value below which k% of items fall")
    ax.set_xlim(0, 1)
    return str(save(fig, "05_01_quantiles"))


def _histogram_bins(df: pd.DataFrame) -> str:
    """Slide 12: histogram approximates the probability function - and 'how to pick bins?'."""
    cal = df["calories"]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9), sharey=False)
    for ax, bins, note in zip(axes, [5, 15, 50], ["too few - hides shape",
                                                  "about right", "too many - noise"]):
        ax.hist(cal, bins=bins, color=ACCENT, edgecolor="white", linewidth=0.8)
        ax.set_title(f"{bins} bins\n{note}", fontsize=11)
        ax.set_xlabel("calories")
    axes[0].set_ylabel("number of menu items")
    fig.suptitle("Histogram of calories - the bin count IS a modelling choice",
                 fontsize=13, fontweight="bold", color=INK, y=1.04)
    return str(save(fig, "05_02_histogram_bins"))


def _small_multiples(df: pd.DataFrame) -> str:
    """Slide 13: one histogram per group on a SHARED axis, so shapes cluster visually."""
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 5.6), sharex=True, sharey=True)
    bins = np.linspace(100, 1300, 19)
    for ax, kind, color in zip(axes.ravel(), TYPE_ORDER, CATEGORICAL):
        v = df.loc[df["type"] == kind, "calories"]
        ax.hist(v, bins=bins, color=color, edgecolor="white", linewidth=0.7)
        ax.axvline(v.median(), color=INK, lw=1.4, ls="--")
        ax.set_title(f"{kind}  (n={v.size}, median {v.median():.0f})", fontsize=10)
    for ax in axes[-1]:
        ax.set_xlabel("calories")
    for ax in axes[:, 0]:
        ax.set_ylabel("items")
    fig.suptitle("Small multiples: same bins, same axes - only the shape changes",
                 fontsize=13, fontweight="bold", color=INK, y=1.0)
    fig.tight_layout()
    return str(save(fig, "05_03_small_multiples"))


def _density_line(df: pd.DataFrame) -> str:
    """Slide 14: enough bins -> the outline becomes a density curve; easy to overlay."""
    from scipy import stats

    pairs = [("Burger", CATEGORICAL[0]), ("Chicken Nuggets", CATEGORICAL[1])]
    grid = np.linspace(0, 1400, 400)

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    for (kind, color) in pairs:
        v = df.loc[df["type"] == kind, "calories"]
        kde = stats.gaussian_kde(v)
        density = kde(grid)
        ax.fill_between(grid, density, color=color, alpha=0.18)
        ax.plot(grid, density, color=color, lw=2, label=kind)
        peak = grid[density.argmax()]
        ax.annotate(kind, (peak, density.max()), textcoords="offset points",
                    xytext=(6, 8), color=INK, fontsize=10)
    ax.set(xlabel="calories", ylabel="relative frequency (density)",
           title="Density lines let two distributions share one axis")
    ax.legend(loc="upper right")
    return str(save(fig, "05_04_density_line"))


def _error_bar(df: pd.DataFrame) -> str:
    """Slide 15: mean +/- SD side by side when there are too many groups to overlay."""
    stats_by = (df.groupby("type")["calories"].agg(["mean", "std"]).reindex(TYPE_ORDER))
    x = np.arange(len(TYPE_ORDER))

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.errorbar(x, stats_by["mean"], yerr=stats_by["std"], fmt="o-", color=ACCENT,
                ms=9, capsize=5, elinewidth=1.6, mec="white", mew=1.5)
    for xi, (m, s) in enumerate(zip(stats_by["mean"], stats_by["std"])):
        ax.annotate(f"{m:.0f}", (xi, m), textcoords="offset points",
                    xytext=(10, 6), fontsize=9, color=INK)
    ax.set_xticks(x, [SHORT_LABEL[t] for t in TYPE_ORDER])
    ax.set(ylabel="calories", title="Mean calories by food type (bars = +/- 1 SD)")
    return str(save(fig, "05_05_error_bar"))


def _error_variants(df: pd.DataFrame) -> str:
    """Slide 16: the two variants - error cloud, and bar plot with error bars."""
    stats_by = (df.groupby("type")["calories"].agg(["mean", "std"]).reindex(TYPE_ORDER))
    x = np.arange(len(TYPE_ORDER))
    labels = [SHORT_LABEL[t] for t in TYPE_ORDER]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.3))
    ax1.fill_between(x, stats_by["mean"] - stats_by["std"], stats_by["mean"] + stats_by["std"],
                     color=ACCENT, alpha=0.18)
    ax1.plot(x, stats_by["mean"], color=ACCENT, lw=2, marker="o", ms=8, mec="white", mew=1.5)
    ax1.set_xticks(x, labels)
    ax1.set(ylabel="calories", title="Error cloud")

    ax2.bar(x, stats_by["mean"], yerr=stats_by["std"], color=ACCENT, width=0.62,
            capsize=5, error_kw={"elinewidth": 1.6, "ecolor": MUTED})
    ax2.set_xticks(x, labels)
    ax2.set(ylabel="calories", title="Bar plot with error bars")
    fig.suptitle("Two ways to show the same mean +/- SD summary",
                 fontsize=13, fontweight="bold", color=INK, y=1.03)
    return str(save(fig, "05_06_error_variants"))


def _violin(df: pd.DataFrame) -> str:
    """Slides 17-18: violin keeps the full shape a mean +/- SD summary throws away."""
    groups, labels = series_by_type(df, "calories")
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    parts = ax.violinplot(groups, showmedians=True, widths=0.82)
    for body, color in zip(parts["bodies"], CATEGORICAL):
        body.set_facecolor(color)
        body.set_alpha(0.45)
        body.set_edgecolor(color)
        body.set_linewidth(1.2)
    for key in ("cbars", "cmins", "cmaxes", "cmedians"):
        parts[key].set_edgecolor(MUTED)
        parts[key].set_linewidth(1.2)
    ax.set_xticks(np.arange(1, len(labels) + 1), labels)
    ax.set(ylabel="calories",
           title="Violin plot: full distribution shape per food type")
    return str(save(fig, "05_07_violin"))


def _violin_split(df: pd.DataFrame) -> str:
    """Slide 18: combined violin - one violin per group, split by a second variable."""
    sub = df[df["type"].isin(["Burger", "Breaded Chicken Sandwich", "Grilled Chicken Sandwich"])]
    kinds = ["Burger", "Breaded Chicken Sandwich", "Grilled Chicken Sandwich"]
    high = sub["sodium"] >= sub["sodium"].median()

    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    for side, (mask, color, name) in enumerate(
        [(~high, CATEGORICAL[0], "sodium below median"), (high, CATEGORICAL[1], "sodium at/above median")]
    ):
        data = [sub.loc[(sub["type"] == k) & mask, "calories"].to_numpy() for k in kinds]
        pos = np.arange(1, len(kinds) + 1)
        parts = ax.violinplot(data, positions=pos, widths=0.9, showextrema=False)
        for body in parts["bodies"]:
            verts = body.get_paths()[0].vertices
            centre = verts[:, 0].mean()
            # clip each violin to one side of its tick -> the two groups share a spine
            verts[:, 0] = np.clip(verts[:, 0], centre, np.inf) if side else \
                np.clip(verts[:, 0], -np.inf, centre)
            body.set_facecolor(color)
            body.set_alpha(0.5)
            body.set_edgecolor("white")   # surface gap where the two halves meet
            body.set_linewidth(1.5)
        ax.plot([], [], color=color, lw=6, alpha=0.5, label=name)
    ax.set_xticks(np.arange(1, len(kinds) + 1), ["Burger", "Breaded\nChicken", "Grilled\nChicken"])
    ax.set(ylabel="calories", title="Combined violin: calories split by sodium level")
    ax.legend(loc="upper right")
    return str(save(fig, "05_08_violin_combined"))


def _barcode(df: pd.DataFrame) -> str:
    """Slide 19: one row per restaurant, one thin bar per item, bold bar = that row's mean.

    Fits the slide's own criteria: ranges are compatible (all calories), few points
    per row (5-13 items), and it costs almost no vertical space.
    """
    order = df.groupby("restaurant")["calories"].mean().sort_values().index
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    for i, rest in enumerate(order):
        v = df.loc[df["restaurant"] == rest, "calories"]
        ax.vlines(v, i - 0.32, i + 0.32, color=ACCENT, lw=1.2, alpha=0.75)
        ax.vlines(v.mean(), i - 0.42, i + 0.42, color=INK, lw=2.6)
    ax.set_yticks(range(len(order)), list(order))
    ax.set(xlabel="calories", title="Barcode chart: every menu item, by restaurant")
    ax.annotate("bold bar = restaurant mean", (0.99, 0.02), xycoords="axes fraction",
                ha="right", fontsize=9, color=MUTED)
    ax.grid(axis="y", visible=False)
    ax.invert_yaxis()
    return str(save(fig, "05_09_barcode"))


def _boxplot_anatomy(df: pd.DataFrame) -> str:
    """Slide 20: box = IQR, whisker = furthest point within 1.5*IQR, rest = outliers."""
    v = df["sugars"]
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    iqr = q3 - q1
    hi = v[v <= q3 + 1.5 * iqr].max()

    fig, ax = plt.subplots(figsize=(8.6, 3.6))
    bp = ax.boxplot(v, vert=False, widths=0.45, patch_artist=True,
                    flierprops={"marker": "o", "ms": 5, "mfc": SECOND,
                                "mec": SECOND, "alpha": 0.7})
    bp["boxes"][0].set(facecolor=ACCENT, alpha=0.3, edgecolor=ACCENT, linewidth=1.6)
    for key in ("whiskers", "caps", "medians"):
        for art in bp[key]:
            art.set(color=MUTED if key != "medians" else INK, linewidth=1.8)
    # The box occupies only the first fifth of the axis (outliers stretch it to 93),
    # so labels fan out with leader lines instead of stacking above the box.
    marks = [
        (q1, f"Q1 = {q1:.1f}", (-40, -58), "right"),
        (med, f"median = {med:.1f}", (-14, 62), "right"),
        (q3, f"Q3 = {q3:.1f}", (34, 40), "left"),
        (hi, f"whisker = {hi:.1f}\n(Q3 + 1.5 x IQR)", (58, -58), "left"),
    ]
    for xv, label, offset, ha in marks:
        ax.annotate(label, (xv, 1.0), textcoords="offset points", xytext=offset,
                    ha=ha, va="center", fontsize=9, color=INK,
                    arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8,
                                "shrinkA": 2, "shrinkB": 2})
    ax.annotate(f"outliers (> Q3 + 1.5 x IQR): {int((v > hi).sum())} items",
                (v.max(), 1.0), textcoords="offset points", xytext=(-10, 40),
                ha="right", fontsize=9, color=SECOND)
    ax.set_yticks([])
    ax.set_ylim(0.45, 1.6)
    ax.set(xlabel="sugars (g)", title="Box plot anatomy, on the most skewed variable (sugars)")
    ax.grid(axis="y", visible=False)
    return str(save(fig, "05_10_boxplot_anatomy"))


def _box_compare(df: pd.DataFrame) -> str:
    """Slide 21: box plots side by side - compare centre AND spread at a glance."""
    groups, labels = series_by_type(df, "calories")
    fig, ax = plt.subplots(figsize=(8.6, 4.5))
    bp = ax.boxplot(groups, widths=0.55, patch_artist=True,
                    flierprops={"marker": "o", "ms": 4.5, "mfc": MUTED,
                                "mec": MUTED, "alpha": 0.6})
    for box, color in zip(bp["boxes"], CATEGORICAL):
        box.set(facecolor=color, alpha=0.32, edgecolor=color, linewidth=1.5)
    for key in ("whiskers", "caps"):
        for art in bp[key]:
            art.set(color=MUTED, linewidth=1.2)
    for art in bp["medians"]:
        art.set(color=INK, linewidth=2)
    ax.set_xticks(np.arange(1, len(labels) + 1), labels)
    ax.set(ylabel="calories", title="Comparing samples with box plots")
    return str(save(fig, "05_11_box_compare"))


def _box_notched(df: pd.DataFrame) -> str:
    """Slide 22: the notch is a CI on the median - non-overlapping notches ~ different medians."""
    groups, labels = series_by_type(df, "calories")
    fig, ax = plt.subplots(figsize=(8.6, 4.5))
    bp = ax.boxplot(groups, widths=0.55, notch=True, patch_artist=True,
                    flierprops={"marker": "o", "ms": 4.5, "mfc": MUTED,
                                "mec": MUTED, "alpha": 0.6})
    for box, color in zip(bp["boxes"], CATEGORICAL):
        box.set(facecolor=color, alpha=0.32, edgecolor=color, linewidth=1.5)
    for key in ("whiskers", "caps"):
        for art in bp[key]:
            art.set(color=MUTED, linewidth=1.2)
    for art in bp["medians"]:
        art.set(color=INK, linewidth=2)
    ax.set_xticks(np.arange(1, len(labels) + 1), labels)
    ax.set(ylabel="calories",
           title="Notched box plot: notch = 95% CI around the median")
    return str(save(fig, "05_12_box_notched"))


def _bootstrap(df: pd.DataFrame) -> tuple[str, dict[str, tuple[float, float]]]:
    """Slides 25-26: resample with replacement 10,000x; the middle 95% of means is the CI."""
    burger = df.loc[df["type"] == "Burger", "calories"].to_numpy()
    means = np.array([RNG.choice(burger, burger.size, replace=True).mean() for _ in range(10_000)])
    lo, hi = np.percentile(means, [2.5, 97.5])

    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    ax.hist(means, bins=60, color=ACCENT, alpha=0.35, edgecolor="white", linewidth=0.6)
    ax.axvspan(lo, hi, color=ACCENT, alpha=0.12)
    for xv, label in [(lo, f"2.5th pct\n{lo:.1f}"), (hi, f"97.5th pct\n{hi:.1f}")]:
        ax.axvline(xv, color=SECOND, lw=1.8)
        ax.annotate(label, (xv, ax.get_ylim()[1] * 0.92), ha="center",
                    fontsize=9, color=INK)
    ax.axvline(burger.mean(), color=INK, lw=1.8, ls="--")
    ax.annotate(f"sample mean {burger.mean():.1f}", (burger.mean(), ax.get_ylim()[1] * 0.45),
                textcoords="offset points", xytext=(8, 0), fontsize=9, color=INK)
    ax.set(xlabel="bootstrap mean calories", ylabel="count of resamples",
           title="Bootstrap: 10,000 resampled means of Burger calories")
    path = str(save(fig, "05_13_bootstrap_ci"))
    return path, {"Burger": (lo, hi)}


def _ci_compare(df: pd.DataFrame) -> tuple[str, dict[str, tuple[float, float]]]:
    """Slide 27: two bootstrap CIs that do not overlap -> different at p < 0.05."""
    kinds = ["Burger", "Chicken Nuggets"]
    out: dict[str, tuple[float, float]] = {}
    fig, ax = plt.subplots(figsize=(7.6, 2.9))
    for i, (kind, color) in enumerate(zip(kinds, CATEGORICAL)):
        v = df.loc[df["type"] == kind, "calories"].to_numpy()
        means = np.array([RNG.choice(v, v.size, replace=True).mean() for _ in range(10_000)])
        lo, hi = np.percentile(means, [2.5, 97.5])
        out[kind] = (lo, hi)
        ax.hlines(i, lo, hi, color=color, lw=7, alpha=0.55)
        ax.plot([v.mean()], [i], "o", ms=10, color=color, mec="white", mew=1.6)
        ax.annotate(f"mean {v.mean():.0f}   95% CI [{lo:.0f}, {hi:.0f}]",
                    ((lo + hi) / 2, i), textcoords="offset points", xytext=(0, 14),
                    ha="center", fontsize=9, color=INK)
    ax.set_yticks(range(len(kinds)), kinds)
    ax.set_xlim(150, 780)
    ax.set_ylim(-0.55, 1.5)
    ax.set(xlabel="calories", title="Non-overlapping CIs: the two means differ (p < 0.05)")
    ax.grid(axis="y", visible=False)
    return str(save(fig, "05_14_ci_compare")), out


def build_all(df: pd.DataFrame) -> list[str]:
    """Render every deck-5 figure; returns saved paths."""
    paths = [
        _quantiles(df), _histogram_bins(df), _small_multiples(df),
        _density_line(df), _error_bar(df),
        _error_variants(df), _violin(df), _violin_split(df), _barcode(df),
        _boxplot_anatomy(df), _box_compare(df), _box_notched(df),
    ]
    boot_path, _ = _bootstrap(df)
    ci_path, cis = _ci_compare(df)
    paths += [boot_path, ci_path]
    for kind, (lo, hi) in cis.items():
        print(f"  CI  {kind:<18} 95% bootstrap CI of mean calories = [{lo:.2f}, {hi:.2f}]")
    return paths
