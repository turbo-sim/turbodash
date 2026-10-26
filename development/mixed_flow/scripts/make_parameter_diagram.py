"""Regenerate the original vector engineering diagram used in the documentation."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Arc
import numpy as np

from _common import (
    BLADE,
    SHROUD,
    SPLITTER,
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


def dimension(ax, first, second, label, offset=(0, 0), text_offset=(0, 0)):
    first, second = np.asarray(first), np.asarray(second)
    a, b = first + offset, second + offset
    for p, q in ((first, a), (second, b)):
        ax.plot([p[0], q[0]], [p[1], q[1]], color="#9aabb7", lw=0.65)
    ax.annotate(
        "", b, a, arrowprops={"arrowstyle": "<->", "color": "#536b7c", "lw": 0.9}
    )
    mid = (a + b) / 2 + text_offset
    ax.text(
        *mid,
        label,
        ha="center",
        va="center",
        fontsize=11,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=STUDY / "assets")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    style()
    dims = MeanlineDimensions.from_ratios()
    channel = MeridionalChannel(dims)
    edges = BladeEdges(
        leading_hub=0.16,
        leading_shroud=0.04,
        trailing_hub=0.98,
        trailing_shroud=1,
        leading_bow=0.035,
    )
    blade = BladeMeridional(channel, edges)
    fig = plt.figure(figsize=(15, 9), layout="constrained")
    grid = fig.add_gridspec(2, 2, width_ratios=(1.25, 1))
    main_ax = fig.add_subplot(grid[:, 0])
    wall_ax = fig.add_subplot(grid[0, 1])
    blade_ax = fig.add_subplot(grid[1, 1])
    draw_channel(main_ax, channel, blade, blade.splitter(0.25))
    main_ax.axhline(0, color="#7b8b96", lw=0.8, ls="-.")
    main_ax.text(22, 2, "Rotation axis", color="#70818d", fontsize=9)
    dimension(
        main_ax, (0, 0), (0, 25), r"$r_{1h}$", offset=(-15, 0), text_offset=(-3, 0)
    )
    dimension(
        main_ax, (0, 0), (0, 65), r"$r_{1s}$", offset=(-25, 0), text_offset=(-3, 0)
    )
    dimension(main_ax, (64, 0), (64, 100), r"$r_2$", offset=(13, 0), text_offset=(3, 0))
    dimension(main_ax, (0, 0), (60, 0), r"$L$", offset=(0, -10), text_offset=(0, -3))
    main_ax.plot([60, 60], [0, 106], color="#9aabb7", ls=":", lw=0.75)
    dimension(
        main_ax, (56, 100), (64, 100), r"$b_2$", offset=(0, 11), text_offset=(0, 3)
    )
    main_ax.annotate(
        "Inlet direction: $\\theta_1=0°$",
        xy=(0, 45),
        xytext=(-26, 79),
        arrowprops={"arrowstyle": "->", "color": "#536b7c"},
        fontsize=10,
    )
    main_ax.annotate(
        "Outlet direction: $\\theta_2=90°$",
        xy=(60, 101),
        xytext=(3, 123),
        arrowprops={"arrowstyle": "->", "color": "#536b7c"},
        fontsize=10,
    )
    main_ax.annotate(
        "",
        (19, 46),
        (5, 46),
        arrowprops={"arrowstyle": "->", "color": "#536b7c", "lw": 1.5},
    )
    main_ax.annotate(
        "",
        (60, 95),
        (60, 81),
        arrowprops={"arrowstyle": "->", "color": "#536b7c", "lw": 1.5},
    )
    main_ax.set(
        xlim=(-34, 85),
        ylim=(-18, 131),
        title="A   Meanline skeleton and independent blade extent",
    )
    main_ax.legend(
        loc="lower center", bbox_to_anchor=(0.56, 0.14), fontsize=9, framealpha=0.95
    )

    # Engineering tangent reaches are derived lengths, never exposed CP inputs.
    shape = WallShape(
        inlet_angle_deg=12,
        outlet_angle_deg=82,
        inlet_tangent_fraction=0.45,
        outlet_tangent_fraction=0.45,
    )
    shaped = MeridionalChannel(dims, shroud=shape).shroud
    bowed = MeridionalChannel(
        dims, shroud=WallShape(inlet_angle_deg=12, outlet_angle_deg=82, bow=0.035)
    ).shroud
    u = np.linspace(0, 1, 301)
    line(wall_ax, shaped.point(u), color=SHROUD, lw=2.4, label="Reference wall")
    line(
        wall_ax,
        bowed.point(u),
        color="#7c8a99",
        lw=1.6,
        ls="--",
        label="With positive bow",
    )
    start, end = shaped.point(0) * 1000, shaped.point(1) * 1000
    chord = end - start
    t0 = np.array([np.cos(np.radians(12)), np.sin(np.radians(12))])
    t1 = np.array([np.cos(np.radians(82)), np.sin(np.radians(82))])
    shoulder0 = start + 0.45 * np.dot(chord, t0) * t0
    shoulder1 = end - 0.45 * np.dot(chord, t1) * t1
    wall_ax.plot(
        [start[0], end[0]],
        [start[1], end[1]],
        color="#bdc6cd",
        ls=":",
        label="Endpoint chord D",
    )
    wall_ax.plot(
        [start[0], start[0] + 29], [start[1], start[1]], color="#aab6bf", lw=0.7
    )
    wall_ax.plot([end[0] - 22, end[0]], [end[1], end[1]], color="#aab6bf", lw=0.7)
    dimension(
        wall_ax,
        start,
        shoulder0,
        r"$\ell_{in}=f_{in}(D\cdot t_{in})$",
        offset=(0, -5),
        text_offset=(0, -3),
    )
    dimension(
        wall_ax, shoulder1, end, r"$\ell_{out}$", offset=(7, 0), text_offset=(4, 0)
    )
    wall_ax.add_patch(Arc(start, 36, 36, theta1=0, theta2=12, color="#536b7c"))
    wall_ax.text(start[0] + 22, start[1] + 2, r"$\alpha_{in}$", fontsize=11)
    wall_ax.add_patch(Arc(end, 20, 20, theta1=180, theta2=262, color="#536b7c"))
    wall_ax.text(end[0] - 17, end[1] - 6, r"$\alpha_{out}$", fontsize=11)
    midpoint, displacement = shaped.point(0.5) * 1000, bowed.point(0.5) * 1000
    wall_ax.annotate(
        "$b|D|$ at $u=0.5$",
        xy=displacement,
        xytext=(6, 97),
        arrowprops={"arrowstyle": "->", "color": "#536b7c"},
    )
    wall_ax.plot(
        [midpoint[0], displacement[0]],
        [midpoint[1], displacement[1]],
        color="#536b7c",
        lw=1,
    )
    wall_ax.set(
        xlim=(-3, 72),
        ylim=(52, 110),
        aspect="equal",
        xlabel="z [mm]",
        ylabel="r [mm]",
        title="B   Wall slopes, tangent reach, and bow (shroud shown)",
    )
    wall_ax.legend(loc="upper left", fontsize=8)
    wall_ax.grid(True)

    eta = np.linspace(0, 1, 201)
    leading, trailing = edges.leading(eta), edges.trailing(eta)
    blade_ax.fill_betweenx(eta, leading, trailing, color=BLADE, alpha=0.15)
    blade_ax.plot(leading, eta, color="#354859", lw=2, label="Main blade LE / TE")
    blade_ax.plot(trailing, eta, color="#354859", lw=2)
    blade_ax.plot(
        blade.splitter(0.25).edges.leading(eta),
        eta,
        color=SPLITTER,
        ls="--",
        lw=2,
        label="Splitter: 25% of local s extent",
    )
    for y, name in ((0, "h"), (1, "s")):
        for x, edge in ((edges.leading(y), "LE"), (edges.trailing(y), "TE")):
            blade_ax.scatter(x, y, color="#354859", s=16, zorder=4)
            blade_ax.annotate(
                f"$s_{{{edge},{name}}}$",
                (x, y),
                xytext=(0, -18 if y == 0 else 9),
                textcoords="offset points",
                ha="center",
                fontsize=10,
            )
    linear_mid = 0.5 * (edges.leading_hub + edges.leading_shroud)
    blade_ax.annotate(
        "",
        (edges.leading(0.5), 0.5),
        (linear_mid, 0.5),
        arrowprops={"arrowstyle": "<->", "color": "#536b7c"},
    )
    blade_ax.annotate(
        "Edge bow",
        (edges.leading(0.5), 0.5),
        (0.29, 0.53),
        arrowprops={"arrowstyle": "->", "color": "#536b7c"},
        fontsize=9,
    )
    blade_ax.set(
        xlim=(-0.04, 1.08),
        ylim=(-0.17, 1.17),
        xlabel="Normalized wall arclength s",
        ylabel="Span fraction η (hub → shroud)",
        title="C   Blade placement in channel coordinates",
    )
    blade_ax.set_yticks([0, 0.5, 1])
    blade_ax.legend(loc="center right", fontsize=8)
    blade_ax.grid(True)
    fig.suptitle(
        "Meridional geometry · engineering controls for channel walls and blades",
        fontsize=17,
    )
    save(fig, args.output_dir, "meridional_parameters")
    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
