"""Show independent hub, middle and shroud edits using the production API.

Writes a comparison figure and the fully resolved variant YAML. --export also
writes its closed blade/row and offline HTML. These edits illustrate controls;
they are not aerodynamic optimization results.
"""

import argparse
from copy import deepcopy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from turbodash.geometry.impeller import build_impeller, load_config, save_config
from turbodash.geometry.impeller.output import export_geometry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "output/section_study")
    parser.add_argument("--export", action="store_true")
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    config = load_config(Path(__file__).parent / "examples/reversible_impeller.yaml")
    baseline = build_impeller(config)
    edited = deepcopy(config)
    edited["name"] = "independent_span_controls_demonstration"
    edited["source"]["study_note"] = "Illustrative span, footprint and stacking edits; performance not recomputed."
    edited["blade_edges"].update(leading_hub=0.03, leading_shroud=0.07, leading_bow=0.01,
                                  trailing_hub=0.97, trailing_shroud=0.95)
    shroud = edited["sections"][-1]
    shroud["camberline"].update(outlet_metal_angle_deg=-37, turning_location=0.62)
    shroud["thickness"]["maximum_thickness_m"] *= 0.8
    shroud["stacking_offset_deg"] = 3
    middle = deepcopy(edited["sections"][0])
    middle["v"] = 0.5
    # Use the baseline radius law for the intermediate inlet angle; footprint
    # edits intentionally change the RMS correspondence of this study.
    middle["camberline"].update(inlet_metal_angle_deg=float(np.rad2deg(baseline.section(.5).beta[0])),
                                 outlet_metal_angle_deg=-32, turning_location=.57)
    middle["thickness"]["maximum_thickness_m"] *= 1.15
    middle["stacking_offset_deg"] = 1
    edited["sections"].insert(1, middle)
    variant = build_impeller(edited)
    args.output.mkdir(parents=True, exist_ok=True)
    save_config(variant.config, args.output / "independent_sections.yaml")

    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout="constrained")
    for v, color, label in ((0, "#176b87", "Hub"), (.5, "#d18b20", "Middle"), (1, "#8856a7", "Shroud")):
        for geometry, style, suffix in ((baseline, "--", " baseline"), (variant, "-", " edited")):
            section = geometry.section(v)
            axes[0].plot(geometry.u, np.rad2deg(section.beta), style, color=color, label=label + suffix)
            axes[1].plot(geometry.u, 1000 * section.target_thickness_m, style, color=color)
            axes[2].plot(section.camber[:, 2], np.hypot(section.camber[:, 0], section.camber[:, 1]), style, color=color)
    axes[0].set(xlabel="Conformal u", ylabel="Metal angle [deg]", title="Independent angle laws")
    axes[1].set(xlabel="Conformal u", ylabel="Target thickness [mm]", title="Independent thickness")
    axes[2].set(xlabel="z [m]", ylabel="r [m]", title="Swept and bowed footprints", aspect="equal")
    axes[0].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=.2)
    figure.savefig(args.output / "comparison.png", dpi=180)
    figure.savefig(args.output / "comparison.svg")
    plt.close(figure)
    if args.export:
        export_geometry(variant, args.output / "geometry")
    print(f"Study outputs: {args.output.resolve()}")
    print(f"Edited-case thickness residual: {100 * variant.diagnostics['maximum_relative_thickness_error']:.6f}%")


if __name__ == "__main__":
    main()
