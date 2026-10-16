"""Reproduce the numerical observations in curvature_method_review.md.

Run from the repository root with the project Python. This reads case.yaml
and writes diagnostic artifacts beside this script; it does not edit the case.
The curvature interval deliberately excludes the rounded tip and trailing edge.
"""

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from blade_geometry import (
    apply_thickness, compute_camberline, compute_denton_thickness,
    create_meridional_path,
)
from demo_asymmetry import signed_surface_curvature


def main():
    config = yaml.safe_load((HERE.parent / "case.yaml").read_text(encoding="utf-8"))
    kind = config["blade_type"]
    parameters = config["thickness"]
    extent = min(2 * parameters["leading_edge_fraction"], 1.0)
    rows = []
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout="constrained")
    colors = {-0.15: "#D55E00", 0.0: "#666666", 0.15: "#0072B2", 0.3: "#009E73", 0.45: "#CC79A7"}
    for count in (10001, 20001, 40001):
        xi = np.linspace(0, 1, count)
        path = create_meridional_path(kind, config[kind + "_geometry"], xi)
        camber = compute_camberline(path, config["camberline"])
        chord = path["meridional_chord"]
        mask = (xi >= 0.05 * extent) & (xi <= extent)
        for alpha, color in colors.items():
            thickness = compute_denton_thickness(xi, dict(parameters, leading_edge_asymmetry=alpha))
            surface = apply_thickness(path, camber, thickness["upper_half_thickness_ratio"],
                                      thickness["lower_half_thickness_ratio"])
            row = {"samples": count, "asymmetry": alpha}
            for name in ("upper", "lower"):
                curvature = chord * signed_surface_curvature(surface[name], xi)
                row[name + "_peak_abs_curvature_times_chord"] = float(np.max(np.abs(curvature[mask])))
                if count == 40001:
                    axis = axes[1 if name == "upper" else 2]
                    axis.plot(xi[mask], curvature[mask], color=color, label=f"alpha = {alpha:g}")
            row["maximum_thickness_ratio"] = float(np.max(thickness["full_thickness_ratio"]))
            row["maximum_thickness_station"] = float(xi[np.argmax(thickness["full_thickness_ratio"])])
            rows.append(row)
            if count == 40001:
                axes[0].plot(*(surface["boundary"] / chord).T, color=color, label=f"alpha = {alpha:g}")
    axes[0].set(title="Supplied axial case: asymmetry sweep", xlabel="Axial distance / chord",
                ylabel="Circumferential distance / chord")
    axes[0].set_aspect("equal")
    for axis, title in zip(axes[1:], ("Plus-side surface", "Minus-side surface")):
        axis.set(title=title, xlabel="Meridional fraction", ylabel="Signed curvature x chord")
        axis.axvline(parameters["leading_edge_fraction"], color="0.6", ls=":", lw=0.8)
        axis.axhline(0, color="0.7", lw=0.6)
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    figure.suptitle("Geometry diagnostics only; curvature shown away from the tip")
    figure.savefig(HERE / "v3_asymmetry_sweep.png", dpi=160)
    figure.savefig(HERE / "v3_asymmetry_sweep.svg")
    plt.close(figure)
    with (HERE / "v3_diagnostics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"curvature_interval": [0.05 * extent, extent],
                      "nominal_maximum": parameters["maximum_thickness_ratio"],
                      "nominal_station": parameters["maximum_thickness_location"],
                      "results": rows}, indent=2))


if __name__ == "__main__":
    main()
