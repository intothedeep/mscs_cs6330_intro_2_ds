"""
Problem 1: fit power law, exponential, uniform and normal models to
  A. airport routes (routes per airport)
  B. movie votes (average TMDb rating)
and compare each fitted model against the data by drawing random samples from it.

Run from the repo root:  uv run python hw2/code/p1.py
Writes figures to hw2/figures/ and prints every estimate used in the report.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write PNGs only, no GUI backend needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

HW_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = HW_DIR / "00_doc"
FIG_DIR = HW_DIR / "figures"
SEED = 6330
VOTE_VIEW_MAX = 20.0  # x-limit for movie PDFs; ratings live in [1, 10]

DATA_COLOR = "#2a78d6"
MODEL_COLOR = "#eb6834"
INK = "#52514e"
GRID = "#e4e3df"

plt.rcParams.update({
    "axes.edgecolor": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11, "legend.frameon": False,
})


# ---------------------------------------------------------------- estimators (pure)

def fit_power_law(x: np.ndarray) -> dict[str, float]:
    """Continuous MLE (Clauset et al. 2009) with x_min = min(x): alpha = 1 + n / sum ln(x/x_min)."""
    x_min = float(x.min())
    alpha = 1.0 + len(x) / float(np.log(x / x_min).sum())
    return {"alpha": alpha, "x_min": x_min}


def fit_exponential(x: np.ndarray) -> dict[str, float]:
    """MLE: lambda = 1 / mean."""
    return {"lambda": 1.0 / float(x.mean())}


def fit_uniform(x: np.ndarray) -> dict[str, float]:
    """MLE: [a, b] = [min, max]."""
    return {"a": float(x.min()), "b": float(x.max())}


def fit_normal(x: np.ndarray) -> dict[str, float]:
    """mu = sample mean, sigma = sample standard deviation (ddof = 1)."""
    return {"mu": float(x.mean()), "sigma": float(x.std(ddof=1))}


def sample_models(x: np.ndarray, rng: np.random.Generator) -> dict[str, tuple[str, np.ndarray]]:
    """Draw len(x) values from each fitted model; returns {key: (label, sample)}."""
    n = len(x)
    pl, ex, un, no = fit_power_law(x), fit_exponential(x), fit_uniform(x), fit_normal(x)
    # Inverse-CDF sampling of the Pareto form: x = x_min * (1 - u)^(-1 / (alpha - 1)).
    pl_sample = pl["x_min"] * (1.0 - rng.random(n)) ** (-1.0 / (pl["alpha"] - 1.0))
    return {
        "powerlaw": (f"power law  α={pl['alpha']:.3f}, x_min={pl['x_min']:g}", pl_sample),
        "exponential": (f"exponential  λ={ex['lambda']:.4f}",
                        rng.exponential(1.0 / ex["lambda"], n)),
        "uniform": (f"uniform  [a, b]=[{un['a']:g}, {un['b']:g}]",
                    rng.uniform(un["a"], un["b"], n)),
        "normal": (f"normal  μ={no['mu']:.3f}, σ={no['sigma']:.3f}",
                   rng.normal(no["mu"], no["sigma"], n)),
    }


# ---------------------------------------------------------------- plotting

def ccdf(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """P(X >= x) at each sorted value - a power law is a straight line on log-log."""
    xs = np.sort(x)
    return xs, 1.0 - np.arange(len(xs)) / len(xs)


def ecdf(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xs = np.sort(x)
    return xs, np.arange(1, len(xs) + 1) / len(xs)


def qq_panel(ax: plt.Axes, data: np.ndarray, sample: np.ndarray, log: bool) -> None:
    q = np.linspace(0.005, 0.995, 199)
    dq, sq = np.quantile(data, q), np.quantile(sample, q)
    if log:
        keep = sq > 0  # normal / exponential samples can go <= 0; log axes cannot show them
        dq, sq = dq[keep], sq[keep]
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.scatter(dq, sq, s=12, color=MODEL_COLOR, zorder=3)
    lo, hi = min(dq.min(), sq.min()), max(dq.max(), sq.max())
    ax.plot([lo, hi], [lo, hi], color=INK, lw=1, ls="--", label="y = x (perfect fit)")
    ax.set(xlabel="data quantile", ylabel="model-sample quantile", title="QQ plot")
    ax.legend(loc="upper left")


def save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{name}.png", dpi=160)
    plt.close(fig)


def plot_airport_data(x: np.ndarray) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    # Routes are integers, so plot P(X = k) per value instead of log bins (which leave gaps).
    values, counts = np.unique(x, return_counts=True)
    ax1.scatter(values, counts / len(x), s=12, color=DATA_COLOR)
    ax1.set(xscale="log", yscale="log", xlabel="routes per airport (log)",
            ylabel="P(X = k) (log)", title="PDF (probability mass per value)")
    ax2.plot(*ccdf(x), color=DATA_COLOR, lw=2, label="data")
    ax2.set(xscale="log", yscale="log", xlabel="routes per airport (log)",
            ylabel="P(X ≥ x) (log)", title="CCDF")
    fig.suptitle(f"Airport routes: empirical distribution (n = {len(x)})")
    save(fig, "p1a_0_data")


def plot_airport_model(x: np.ndarray, key: str, label: str, sample: np.ndarray) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    ax1.plot(*ccdf(x), color=DATA_COLOR, lw=2, label="data")
    # CCDF over the FULL sample, then hide x <= 0 (log axis); filtering first would renormalise.
    xs, ps = ccdf(sample)
    ax1.plot(xs[xs > 0], ps[xs > 0], color=MODEL_COLOR, lw=2, label="model sample")
    ax1.set(xscale="log", yscale="log", xlabel="routes per airport (log)",
            ylabel="P(X ≥ x) (log)", title="CCDF: data vs model sample")
    ax1.legend(loc="lower left")
    qq_panel(ax2, x, sample, log=True)
    fig.suptitle(f"Airport routes vs {label}")
    save(fig, f"p1a_{key}")


def plot_movie_data(x: np.ndarray) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    ax1.hist(x, bins=np.arange(1.85, 8.65, 0.1), density=True, color=DATA_COLOR)
    ax1.set(xlabel="average vote", ylabel="density",
            title="PDF, histogram (bin = 0.1, one per vote value)")
    ax2.plot(*ecdf(x), color=DATA_COLOR, lw=2)
    ax2.set(xlabel="average vote", ylabel="P(X ≤ x)", title="CDF")
    fig.suptitle(f"Movie votes: empirical distribution (n = {len(x)})")
    save(fig, "p1b_0_data")


def plot_movie_model(x: np.ndarray, key: str, label: str, sample: np.ndarray) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3.8))
    # A power-law sample reaches the thousands; clip the PDF view and say how much is hidden.
    lo = min(x.min(), np.quantile(sample, 0.001))
    hi = min(max(x.max(), np.quantile(sample, 0.999)), VOTE_VIEW_MAX)
    # Votes sit on a 0.1 grid; bin edges between grid points stop bins catching 1 or 2 values.
    bins = np.arange(np.floor(lo * 10) / 10 - 0.05, hi + 0.2, 0.2)
    width = bins[1] - bins[0]
    # Weights (not density=True) keep each curve normalised to its whole sample, even if clipped.
    ax1.hist(x, bins=bins, weights=np.full(len(x), 1 / (len(x) * width)), color=DATA_COLOR,
             alpha=0.55, label="data")
    ax1.hist(sample, bins=bins, weights=np.full(len(sample), 1 / (len(sample) * width)),
             histtype="step", color=MODEL_COLOR, lw=2, label="model sample")
    hidden = float(np.mean((sample < bins[0]) | (sample > bins[-1])))
    if hidden > 0.01:
        ax1.text(0.98, 0.58, f"{hidden:.0%} of the sample\nlies beyond this axis",
                 transform=ax1.transAxes, ha="right", color=INK)
    ax1.set(xlabel="average vote", ylabel="density", title="PDF: data vs model sample")
    ax1.legend(loc="upper right")
    qq_panel(ax2, x, sample, log=hi == VOTE_VIEW_MAX)
    fig.suptitle(f"Movie votes vs {label}")
    save(fig, f"p1b_{key}")


# ---------------------------------------------------------------- main

def report(name: str, x: np.ndarray) -> None:
    print(f"\n== {name}  (n = {len(x)}, mean = {x.mean():.4f}, median = {np.median(x):.4f})")
    for fit in (fit_power_law, fit_exponential, fit_uniform, fit_normal):
        print(f"  {fit.__name__:16s} {fit(x)}")


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    airports = pd.read_csv(DATA_DIR / "01_airport_routes.csv")["NumberOfRoutes"].to_numpy(float)
    movies = pd.read_csv(DATA_DIR / "01_movie_votes.csv")["AverageVote"].to_numpy(float)

    report("airport routes", airports)
    plot_airport_data(airports)
    for i, (key, (label, sample)) in enumerate(sample_models(airports, rng).items(), start=1):
        print(f"  KS vs {key:12s} D = {ks_2samp(airports, sample).statistic:.3f}")
        plot_airport_model(airports, f"{i}_{key}", label, sample)

    report("movie votes", movies)
    plot_movie_data(movies)
    for i, (key, (label, sample)) in enumerate(sample_models(movies, rng).items(), start=1):
        print(f"  KS vs {key:12s} D = {ks_2samp(movies, sample).statistic:.3f}")
        plot_movie_model(movies, f"{i}_{key}", label, sample)


if __name__ == "__main__":
    main()
