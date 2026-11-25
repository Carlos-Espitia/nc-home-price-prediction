"""Shared matplotlib style and colors so every report figure looks consistent."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"

# Colors (validated categorical slots 1-2, text inks, grid)
BLUE = "#257de0"
ORANGE = "#f47230"
TEXT = "#191919"
TEXT_2 = "#666666"
GRID = "#e6e6e6"

# Sequential (one hue, light -> dark) and diverging (blue <-> gray <-> red)
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#eef5fe", "#9dc8f7", "#338ced", "#1757ac", "#0e3d7e"]
)
DIVERGING = LinearSegmentedColormap.from_list(
    "div_blue_red", ["#1757ac", "#84b9f3", "#efefef", "#f19595", "#c42f30"]
)


def apply_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.edgecolor": TEXT_2,
        "axes.labelcolor": TEXT,
        "axes.titlecolor": TEXT,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "xtick.color": TEXT_2,
        "ytick.color": TEXT_2,
        "legend.frameon": False,
    })


def save(fig, name: str) -> Path:
    """Save a figure to figures/<name>.png and return the path."""
    FIG_DIR.mkdir(exist_ok=True)
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, facecolor="white")
    return path
