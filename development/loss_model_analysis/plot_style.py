"""Shared audit styling and explicit AM reference-chart coverage."""
from functools import lru_cache
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
from turbodash.visualization.matplotlib_style import set_plot_options

ROOT = Path(__file__).resolve().parent
EXTRAPOLATION_COLOR = "#c7bda9"


def colors(count):
    return plt.cm.viridis(np.linspace(.06, .88, count))


def apply_style(fontsize=11):
    set_plot_options(fontsize=fontsize, grid=True, color_order=list(colors(6)),
                     linewidth=1.5, minor_ticks=False)
    plt.rcParams.update({"svg.fonttype": "path", "grid.alpha": .22, "axes.spines.top": True,
                         "axes.spines.right": True, "axes.labelpad": 5,
                         "axes.titlepad": 8})


@lru_cache(None)
def coverage(family):
    path = ROOT / "ainley_mathieson_fitting/data" / f"{family}_points.csv"
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    angles = sorted({float(r["alpha_deg"]) for r in rows})
    bounds = [tuple(f(float(r["pitch_chord"]) for r in rows if float(r["alpha_deg"]) == a)
                    for f in (min, max)) for a in angles]
    return np.array(angles), np.array(bounds)


def in_chart(family, angle, spacing):
    """Envelope between measured trace endpoints, not invented data points."""
    knots, bounds = coverage(family)
    angle, spacing = np.broadcast_arrays(angle, spacing)
    lower = np.interp(angle, knots, bounds[:, 0])
    upper = np.interp(angle, knots, bounds[:, 1])
    return (angle >= knots[0]) & (angle <= knots[-1]) & (spacing >= lower) & (spacing <= upper)


def shade_outside(ax, supported, limits, label=None):
    lo, hi = limits
    a, b = supported
    for left, right in ((lo, min(hi, a)), (max(lo, b), hi)):
        if right > left:
            ax.axvspan(left, right, color=EXTRAPOLATION_COLOR, alpha=.25,
                       zorder=0, linewidth=0, label=label)
            label = None


def shade_mask(ax, x, mask, label=None):
    ax.fill_between(x, 0, 1, where=mask, transform=ax.get_xaxis_transform(),
                    color=EXTRAPOLATION_COLOR, alpha=.25, linewidth=0,
                    zorder=0, label=label)
