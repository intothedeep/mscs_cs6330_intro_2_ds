"""Shared chart styling: validated palette, matplotlib defaults, save helper.

Palette is the dataviz reference categorical set (fixed slot order - the order
IS the colorblind-safety mechanism, so never cycle or re-sort it).
Validated light-mode: worst adjacent CVD dE 9.1, normal-vision dE 19.6.
Three slots sit under 3:1 contrast, so every categorical mark here is also
carried by an axis label or direct label ("relief rule") - never color alone.
"""

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")  # file-writing only; no GUI backend in this venv

import matplotlib.pyplot as plt

__all__ = ["ACCENT", "CATEGORICAL", "FIG_DIR", "GRID", "INK", "MUTED", "SECOND",
           "apply_style", "save"]

CATEGORICAL: list[str] = [
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
]
ACCENT = CATEGORICAL[0]
SECOND = CATEGORICAL[1]
INK = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e6e5e1"
SURFACE = "#fcfcfb"

FIG_DIR = Path(__file__).parent / "figures"


def apply_style() -> None:
    """Recessive grid/axes, thin marks, text in ink tokens (never series color)."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "text.color": INK,
        "axes.labelcolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlecolor": INK,
        "axes.titlepad": 10,
        "font.size": 10,
        "legend.frameon": False,
        "lines.linewidth": 2.0,
        "figure.dpi": 130,
    })


def save(fig: plt.Figure, name: str) -> Path:
    """Write one figure to figures/<name>.png and close it."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", dpi=130)
    plt.close(fig)
    return path
