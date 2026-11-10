"""Compare free exit rake, constrained angle laws and trailing-edge stacking.

The curves use the same meanline-derived source geometry. The script writes
comparison figures and each case's resolved YAML. --export also writes offline
3D/STL outputs. No CFD or mesh generation is involved.
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
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "output/rake_study")
    parser.add_argument("--export", action="store_true", help="Also export each closed blade, row and offline 3D viewer.")
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    seed = load_config(Path(__file__).parent / "examples/reversible_impeller.yaml")
    studies = (
        ("free", "Free rake (original)", "free", 0, "#7a8288", "--"),
        ("zero_rake", "0 deg, preserve LE", "preserve_le", 0, "#176b87", "-"),
        ("positive_rake", "+10 deg, preserve LE", "preserve_le", 10, "#d18b20", "-"),
        ("te_stacked", "0 deg, preserve angle laws", "trailing_edge", 0, "#8856a7", "-"),
    )
    args.output.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(12, 9), layout="constrained")
    for name, label, mode, angle, color, style in studies:
        config = deepcopy(seed)
        config["name"] = "reversible_impeller_" + name
        config["rake"] = {"mode": mode, "angle_deg": angle, "reference_azimuth_deg": None}
        config["source"]["rake_study"] = {
            "description": label, "performance_recomputed": False,
            "note": "Geometric constraint comparison; not an aerodynamic optimization.",
        }
        geometry = build_impeller(config)
        case_dir = args.output / name
        case_dir.mkdir(exist_ok=True)
        save_config(geometry.config, case_dir / "geometry.yaml")
        le = np.rad2deg([section.theta[0] for section in geometry.sections])
        te = np.rad2deg([section.theta[-1] for section in geometry.sections])
        axes[0, 0].plot(geometry.v, le, style, color=color, label=label)
        axes[0, 1].plot(geometry.v, te, style, color=color, label=label)
        axes[1, 0].plot(geometry.u, np.rad2deg(geometry.sections[0].beta), style, color=color)
        axes[1, 1].plot(geometry.u, np.rad2deg(geometry.sections[-1].beta), style, color=color)
        if args.export:
            export_geometry(geometry, case_dir)
        diagnostics = geometry.diagnostics["rake"]
        print(f"{label}: hub/shroud TE azimuth {te[0]:.6f}/{te[-1]:.6f} deg")
        if mode != "free":
            print(f"  Maximum rake residual: {diagnostics['max_abs_rake_error_deg']:.3e} deg")
    axes[0, 0].set(xlabel="Span v (hub to shroud)", ylabel="LE azimuth [deg]", title="Leading-edge stacking")
    axes[0, 1].set(xlabel="Span v (hub to shroud)", ylabel="TE azimuth [deg]", title="Exit edge: constant azimuth means zero rake")
    axes[1, 0].set(xlabel="Conformal u", ylabel="Metal angle [deg]", title="Hub metal-angle distribution")
    axes[1, 1].set(xlabel="Conformal u", ylabel="Metal angle [deg]", title="Shroud metal-angle distribution")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].legend(fontsize=8)
    for axis in axes.flat:
        axis.grid(alpha=.2)
    figure.suptitle("Explicit impeller rake: geometric placement and turning-law tradeoffs")
    figure.savefig(args.output / "rake_comparison.png", dpi=180)
    figure.savefig(args.output / "rake_comparison.svg")
    plt.close(figure)
    print(f"Study outputs: {args.output.resolve()}")


if __name__ == "__main__":
    main()
