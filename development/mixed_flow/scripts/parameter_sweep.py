"""Show independent engineering controls at fixed scale and nominal dimensions."""

import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _common import (
    STUDY,
    BladeEdges,
    BladeMeridional,
    MeanlineDimensions,
    MeridionalChannel,
    WallShape,
    draw_channel,
    line,
    save,
    style,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=STUDY / "output" / "sensitivity"
    )
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    style()
    dimensions = MeanlineDimensions.from_ratios()
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), layout="constrained")
    families = [
        ("Inlet inclination (both walls inherit)", "inlet", (-15, 0, 15)),
        ("Outlet inclination (both walls inherit)", "outlet", (60, 75, 90)),
        ("Shroud inlet wall angle", "angle", (0, 10, 20)),
        ("Shroud inlet / outlet tangent reach", "reach", (0.25, 0.45, 0.65)),
        ("Shroud bow / wall chord", "bow", (-0.035, 0, 0.035)),
    ]
    colors = ["#4d81af", "#253747", "#d37635"]
    for ax, (title, kind, values) in zip(axes.flat, families):
        for color, value in zip(colors, values):
            dims, shape = dimensions, WallShape()
            if kind == "inlet":
                dims = replace(dimensions, inlet_flow_angle_deg=value)
            elif kind == "outlet":
                dims = replace(dimensions, outlet_flow_angle_deg=value)
            elif kind == "angle":
                shape = WallShape(inlet_angle_deg=value)
            elif kind == "reach":
                shape = WallShape(
                    inlet_tangent_fraction=value, outlet_tangent_fraction=value
                )
            else:
                shape = WallShape(bow=value)
            channel = MeridionalChannel(dims, shroud=shape)
            s = np.linspace(0, 1, 301)
            label = (
                f"{value:g}°" if kind in ("inlet", "outlet", "angle") else f"{value:g}"
            )
            line(ax, channel.hub.point_at(s), color=color, lw=1.1, alpha=0.6)
            line(ax, channel.shroud.point_at(s), color=color, lw=2, label=label)
        ax.set(title=title, xlabel="z [mm]", ylabel="r [mm]", aspect="equal")
        ax.legend(fontsize=9)
        ax.grid(True)
    ax = axes[1, 2]
    channel = MeridionalChannel(dimensions)
    draw_channel(ax, channel, stations=False)
    eta = np.linspace(0, 1, 101)
    for color, value in zip(colors, (0, 0.12, 0.24)):
        blade = BladeMeridional(
            channel, BladeEdges(leading_hub=value, leading_shroud=0.0)
        )
        line(ax, blade.leading_edge(eta), color=color, label=f"Hub LE s = {value:g}")
        line(ax, blade.splitter(0.25).leading_edge(eta), color=color, ls="--", lw=1)
    ax.set_title("Blade LE sweep; walls held fixed")
    ax.legend(fontsize=8)
    fig.suptitle(
        "Engineering parameter sensitivities · illustrative geometry", fontsize=16
    )
    save(fig, args.output_dir, "parameter_sweep")
    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
