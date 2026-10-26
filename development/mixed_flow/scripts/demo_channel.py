"""Generate a baseline/YAML-configured channel, diagnostics, and coordinate exports.

Run from any directory; all example length inputs are in metres.
"""

import argparse
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

from _common import HUB, SHROUD, STUDY, build, draw_channel, load_config, save, style


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Optional engineering-input YAML")
    parser.add_argument(
        "--output-dir", type=Path, default=STUDY / "output" / "baseline"
    )
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    style()
    config = load_config(args.config)
    channel, blade, splitter = build(config)
    d = channel.diagnostics(401)
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), layout="constrained")
    draw_channel(axes[0, 0], channel, blade, splitter)
    axes[0, 0].legend(fontsize=8, loc="upper left")
    axes[0, 0].set_title("Meridional channel and independent blade extent")
    axes[0, 1].plot(d["s"], d["area"] * 1e6, label="Revolved station area")
    axes[0, 1].plot(
        d["s"], d["projected_area"] * 1e6, ls="--", label="Midline-normal projection"
    )
    axes[0, 1].set(
        ylabel="Unblocked geometric area [mm²]",
        title="Area evolution (not blade throat area)",
    )
    axes[0, 1].legend(fontsize=9)
    axes[1, 0].plot(d["s"], d["hub_curvature"] / 1000, color=HUB, label="Hub")
    axes[1, 0].plot(d["s"], d["shroud_curvature"] / 1000, color=SHROUD, label="Shroud")
    axes[1, 0].set(ylabel="Signed wall curvature [1/mm]", title="Turning distribution")
    axes[1, 0].legend()
    axes[1, 1].plot(d["s"], d["gap"] * 1000, color="#537e64")
    axes[1, 1].set(
        ylabel="Station connector length [mm]", title="Geometric passage width"
    )
    for ax in (axes[0, 1], axes[1, 0], axes[1, 1]):
        ax.set_xlabel("Normalized wall arclength s")
        ax.grid(True)
    fig.suptitle("Centrifugal / mixed-flow geometry · engineering inputs", fontsize=16)
    save(fig, args.output_dir, "channel_overview")
    resolved = {
        "dimensions": asdict(channel.dimensions),
        "hub": asdict(channel.hub_shape),
        "shroud": asdict(channel.shroud_shape),
        "blade": asdict(blade.edges),
        "splitter_start": config.get("splitter_start", 0.25),
    }
    (args.output_dir / "parameters.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    np.savetxt(
        args.output_dir / "channel.csv",
        np.column_stack(
            [
                d["s"],
                d["hub"],
                d["shroud"],
                d["area"],
                d["projected_area"],
                d["gap"],
                d["hub_curvature"],
                d["shroud_curvature"],
            ]
        ),
        delimiter=",",
        header="s,hub_z_m,hub_r_m,shroud_z_m,shroud_r_m,area_m2,projected_area_m2,gap_m,hub_curvature_per_m,shroud_curvature_per_m",
        comments="",
    )
    eta = np.linspace(0, 1, 101)
    np.savetxt(
        args.output_dir / "blade_edges.csv",
        np.column_stack(
            [
                eta,
                blade.leading_edge(eta),
                blade.trailing_edge(eta),
                splitter.leading_edge(eta),
            ]
        ),
        delimiter=",",
        header="eta,LE_z_m,LE_r_m,TE_z_m,TE_r_m,splitter_LE_z_m,splitter_LE_r_m",
        comments="",
    )
    summary = {
        "hub_length_m": channel.hub.length,
        "shroud_length_m": channel.shroud.length,
        "inlet_area_m2": channel.dimensions.inlet_area,
        "outlet_area_m2": channel.dimensions.outlet_area,
        "minimum_sampled_area_m2": float(d["area"].min()),
        "minimum_sampled_jacobian_m2": float(
            min(d["jacobian_hub"].min(), d["jacobian_shroud"].min())
        ),
        "blade_LE_hub_zr_m": blade.leading_edge(0).tolist(),
        "blade_LE_shroud_zr_m": blade.leading_edge(1).tolist(),
        "blade_TE_hub_zr_m": blade.trailing_edge(0).tolist(),
        "blade_TE_shroud_zr_m": blade.trailing_edge(1).tolist(),
    }
    (args.output_dir / "summary.yaml").write_text(
        yaml.safe_dump(summary, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    print(yaml.safe_dump(summary, sort_keys=False, allow_unicode=True))
    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
