"""Deck 6 (Data and Distributions) -> figures on the fast-food data.

Fitted parameters are printed, not just drawn - the distribution IS its parameters.
Where the dataset has no honest candidate (uniform, power law) the theoretical
curve is drawn as a labelled reference and the text says so plainly.
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
from style import ACCENT, CATEGORICAL, INK, MUTED, SECOND, save

__all__ = ["build_all"]


def _normal_fit(df: pd.DataFrame) -> tuple[str, dict[str, float]]:
    """Slides 4-6, 10: fit a Gaussian to calories; mu centres it, sigma sets its width."""
    v = df["calories"].to_numpy()
    mu, sigma = stats.norm.fit(v)
    grid = np.linspace(v.min() - 150, v.max() + 150, 500)

    fig, ax = plt.subplots(figsize=(7.8, 4.5))
    ax.hist(v, bins=18, density=True, color=ACCENT, alpha=0.30,
            edgecolor="white", linewidth=0.8, label="observed calories")
    ax.plot(grid, stats.norm.pdf(grid, mu, sigma), color=SECOND, lw=2.2,
            label=f"fitted normal  mu={mu:.2f}, sigma={sigma:.2f}")
    ax.axvline(mu, color=INK, lw=1.4, ls="--")
    # left of the line and low, clear of the upper-right legend
    ax.annotate(f"mu = {mu:.2f}", (mu, ax.get_ylim()[1] * 0.25),
                textcoords="offset points", xytext=(-8, 0), ha="right",
                fontsize=9, color=INK)
    ax.set(xlabel="calories", ylabel="probability density",
           title="Normal fit to calories")
    ax.legend(loc="upper right")
    return str(save(fig, "06_01_normal_fit")), {"mu": mu, "sigma": sigma}


def _normal_sigma_bands(df: pd.DataFrame) -> str:
    """Slide 7: 95% of the mass falls within +/- 2 SD - shaded on the fitted curve."""
    v = df["calories"].to_numpy()
    mu, sigma = stats.norm.fit(v)
    grid = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 600)
    pdf = stats.norm.pdf(grid, mu, sigma)

    fig, ax = plt.subplots(figsize=(7.8, 4.5))
    for k, alpha, label in [(2, 0.14, "+/- 2 sigma = 95%"), (1, 0.22, "+/- 1 sigma = 68%")]:
        band = (grid >= mu - k * sigma) & (grid <= mu + k * sigma)
        ax.fill_between(grid[band], pdf[band], color=ACCENT, alpha=alpha, label=label)
    ax.plot(grid, pdf, color=ACCENT, lw=2.2)
    inside = np.mean((v >= mu - 2 * sigma) & (v <= mu + 2 * sigma)) * 100
    for k in (-2, -1, 1, 2):
        ax.axvline(mu + k * sigma, color=MUTED, lw=1, ls=":")
    # the sigma positions ARE the ticks - no second label row to collide with
    ticks = [mu + k * sigma for k in (-2, -1, 0, 1, 2)]
    names = ["-2 sigma", "-1 sigma", "mu", "+1 sigma", "+2 sigma"]
    ax.set_xticks(ticks, [f"{n}\n{t:.0f}" for n, t in zip(names, ticks)])
    ax.set(xlabel="calories", ylabel="probability density",
           title=f"68-95 rule on the fit (actual data inside +/-2 sigma: {inside:.1f}%)")
    ax.legend(loc="upper right")
    return str(save(fig, "06_02_normal_sigma_bands"))


def _cdf(df: pd.DataFrame) -> tuple[str, tuple[float, float]]:
    """CDF view: P(X <= x). Deviation from the fitted line is easier to see than in a histogram."""
    v = np.sort(df["calories"].to_numpy())
    mu, sigma = stats.norm.fit(v)
    emp = np.arange(1, v.size + 1) / v.size

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.step(v, emp, where="post", color=ACCENT, lw=2, label="empirical CDF")
    ax.plot(v, stats.norm.cdf(v, mu, sigma), color=SECOND, lw=2, ls="--",
            label=f"normal CDF  mu={mu:.2f}, sigma={sigma:.2f}")
    # D is the largest vertical gap between the two curves - exactly what the eye
    # is judging here. Shapiro-Wilk carries the normality verdict, because a KS
    # p-value is optimistic when mu/sigma were estimated from this same sample.
    d = np.max(np.abs(emp - stats.norm.cdf(v, mu, sigma)))
    _sw_stat, sw_p = stats.shapiro(v)
    ax.annotate(f"max gap D = {d:.3f}\nShapiro-Wilk p = {sw_p:.5f}",
                (0.03, 0.86), xycoords="axes fraction", fontsize=9, color=MUTED)
    ax.set(xlabel="calories", ylabel="P(X <= x)", title="Empirical vs fitted normal CDF")
    ax.legend(loc="lower right")
    return str(save(fig, "06_03_cdf")), (float(d), float(sw_p))


def _exponential_fit(df: pd.DataFrame) -> tuple[str, dict[str, float]]:
    """Slides 15-18: sugars is the dataset's exponential-shaped variable; lambda = 1/mean."""
    v = df["sugars"].to_numpy()
    lam = 1.0 / v.mean()
    grid = np.linspace(0, v.max(), 400)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.3))
    ax1.hist(v, bins=25, density=True, color=ACCENT, alpha=0.30,
             edgecolor="white", linewidth=0.8, label="observed sugars")
    ax1.plot(grid, stats.expon.pdf(grid, scale=1 / lam), color=SECOND, lw=2.2,
             label=f"exponential PDF  lambda={lam:.4f}")
    ax1.set(xlabel="sugars (g)", ylabel="probability density", title="PDF")
    ax1.legend(loc="upper right")

    sv = np.sort(v)
    ax2.step(sv, np.arange(1, sv.size + 1) / sv.size, where="post", color=ACCENT,
             lw=2, label="empirical CDF")
    ax2.plot(grid, stats.expon.cdf(grid, scale=1 / lam), color=SECOND, lw=2, ls="--",
             label="exponential CDF")
    ax2.set(xlabel="sugars (g)", ylabel="P(X <= x)", title="CDF")
    ax2.legend(loc="lower right")
    fig.suptitle(f"Exponential fit to sugars  (mean = {v.mean():.2f} g, lambda = 1/mean = {lam:.4f})",
                 fontsize=13, fontweight="bold", color=INK, y=1.03)
    return str(save(fig, "06_04_exponential_fit")), {"lambda": lam, "mean": float(v.mean())}


def _uniform_reference() -> str:
    """Slides 11-13: uniform = every outcome equally likely. No fast-food variable is
    uniform, so this is the theoretical reference pair the slides use (die + continuous)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 3.9))
    faces = np.arange(1, 7)
    ax1.bar(faces, np.full(6, 1 / 6), color=ACCENT, width=0.55)
    for f in faces:
        ax1.annotate("1/6", (f, 1 / 6), textcoords="offset points", xytext=(0, 5),
                     ha="center", fontsize=9, color=INK)
    ax1.set(xlabel="die face", ylabel="P(X = x)", ylim=(0, 0.26),
            title="Discrete uniform: fair 6-sided die")
    ax1.set_xticks(faces)

    a, b = 0, 5
    ax2.hlines(1 / (b - a), a, b, color=ACCENT, lw=2.4)
    ax2.fill_between([1, 2], 0, 1 / (b - a), color=SECOND, alpha=0.3)
    ax2.annotate("P(1 < X < 2)\n= 1 x 1/5 = 1/5", (1.5, 0.1), ha="center",
                 fontsize=9, color=INK)
    ax2.set(xlabel="x", ylabel="f(x)", xlim=(-0.6, 5.6), ylim=(0, 0.30),
            title="Continuous uniform on [0, 5]:  f(x) = 1/5")
    fig.suptitle("Uniform distribution - theoretical reference (no uniform variable in this dataset)",
                 fontsize=12, fontweight="bold", color=INK, y=1.04)
    return str(save(fig, "06_05_uniform_reference"))


def _powerlaw_vs_exponential(df: pd.DataFrame) -> str:
    """Slides 20-22: on log-log a power law is a straight line, an exponential curves down.

    The fast-food tail is far too short (126 items) to fit a power law honestly -
    the right panel shows where the real data sits against both reference shapes.
    """
    x = np.linspace(1, 60, 400)
    alpha, xmin = 2.5, 1.0
    pl = (alpha - 1) / xmin * (x / xmin) ** (-alpha)
    ex = stats.expon.pdf(x, scale=8)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.3))
    ax1.loglog(x, pl, color=CATEGORICAL[0], lw=2.2, label=f"power law  alpha={alpha}, x_min={xmin}")
    ax1.loglog(x, ex, color=CATEGORICAL[1], lw=2.2, label="exponential  scale=8")
    # anchored at different x so the two labels never share vertical space
    ax1.annotate("straight on log-log\n= power law", (4, pl[np.searchsorted(x, 4)]),
                 textcoords="offset points", xytext=(-90, -30), fontsize=9,
                 color=CATEGORICAL[0])
    ax1.annotate("bends down\n= exponential", (35, ex[np.searchsorted(x, 35)]),
                 textcoords="offset points", xytext=(-30, 42), fontsize=9,
                 color=CATEGORICAL[1])
    ax1.set(xlabel="x (log)", ylabel="p(x) (log)", title="Power law vs exponential, log-log")
    ax1.legend(loc="lower left")

    v = np.sort(df["sugars"][df["sugars"] > 0].to_numpy())
    ccdf = 1 - np.arange(v.size) / v.size
    ax2.loglog(v, ccdf, color=ACCENT, lw=2, label="sugars, complementary CDF")
    ax2.set(xlabel="sugars, g (log)", ylabel="P(X >= x) (log)",
            title="Real data: too short a tail to call it a power law")
    ax2.legend(loc="lower left")
    return str(save(fig, "06_06_powerlaw_vs_exponential"))


def _bernoulli_binomial(df: pd.DataFrame) -> tuple[str, dict[str, float]]:
    """Slides 23-29: map each item to a binary outcome, then count successes over n trials."""
    high = (df["sodium"] > 1000).to_numpy()
    p = high.mean()
    n = 10

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.2))
    ax1.bar([0, 1], [1 - p, p], color=[CATEGORICAL[0], CATEGORICAL[1]], width=0.5)
    for xi, pv, lab in [(0, 1 - p, "sodium <= 1000 mg"), (1, p, "sodium > 1000 mg")]:
        ax1.annotate(f"{lab}\np = {pv:.4f}", (xi, pv), textcoords="offset points",
                     xytext=(0, 6), ha="center", fontsize=9, color=INK)
    ax1.set(xlabel="outcome", ylabel="probability", ylim=(0, 0.85),
            title=f"Bernoulli PMF  (p = {p:.4f})")
    ax1.set_xticks([0, 1], ["0 = failure", "1 = success"])

    k = np.arange(n + 1)
    ax2.bar(k, stats.binom.pmf(k, n, p), color=ACCENT, width=0.6)
    ax2.axvline(n * p, color=INK, lw=1.4, ls="--")
    ax2.annotate(f"mean = n*p = {n * p:.2f}\nSD = {np.sqrt(n * p * (1 - p)):.2f}",
                 (0.98, 0.95), xycoords="axes fraction", ha="right", va="top",
                 fontsize=9, color=INK)
    ax2.set(xlabel=f"number of high-sodium items in n = {n} picks", ylabel="P(X = k)",
            title=f"Binomial PMF  (n = {n}, p = {p:.4f})")
    ax2.set_xticks(k)
    fig.suptitle("Bernoulli -> Binomial: one trial, then n independent trials",
                 fontsize=13, fontweight="bold", color=INK, y=1.04)
    return str(save(fig, "06_07_bernoulli_binomial")), {
        "p": float(p), "n": n, "mean": float(n * p),
        "sd": float(np.sqrt(n * p * (1 - p))),
    }


def build_all(df: pd.DataFrame) -> list[str]:
    """Render every deck-6 figure and report the fitted parameters."""
    p_norm, norm = _normal_fit(df)
    p_band = _normal_sigma_bands(df)
    p_cdf, (gap_d, sw_p) = _cdf(df)
    p_exp, expo = _exponential_fit(df)
    p_uni = _uniform_reference()
    p_pl = _powerlaw_vs_exponential(df)
    p_bin, bern = _bernoulli_binomial(df)

    print(f"  FIT normal(calories)   mu = {norm['mu']:.2f}, sigma = {norm['sigma']:.2f}")
    print(f"  FIT normal CDF gap     D = {gap_d:.4f}; Shapiro-Wilk p = {sw_p:.5f}"
          f"  -> {'reject' if sw_p < 0.05 else 'cannot reject'} normality at 0.05")
    print(f"  FIT exponential(sugars) lambda = {expo['lambda']:.4f} (mean = {expo['mean']:.2f} g)")
    print(f"  FIT bernoulli(sodium>1000mg) p = {bern['p']:.4f}")
    print(f"  FIT binomial(n=10)     mean = {bern['mean']:.2f}, SD = {bern['sd']:.2f}")
    return [p_norm, p_band, p_cdf, p_exp, p_uni, p_pl, p_bin]
