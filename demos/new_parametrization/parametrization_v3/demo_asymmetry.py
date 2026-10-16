"""Compare symmetric and asymmetric nose geometry and surface curvature.

Run with --no-show to save the comparison without opening a window. The case
and asymmetry factor come from the same YAML as the interactive blade demo.
"""

import argparse
from pathlib import Path

import numpy as np
import yaml

if __package__:
    from .blade_geometry import (
        apply_thickness, compute_camberline, compute_denton_thickness,
        create_meridional_path, map_to_physical_plane,
    )
else:
    from blade_geometry import (
        apply_thickness, compute_camberline, compute_denton_thickness,
        create_meridional_path, map_to_physical_plane,
    )

HERE = Path(__file__).resolve().parent


def signed_surface_curvature(points, meridional_fraction):
    """Estimate planar signed curvature from uniformly sampled surface points.

    Parameters
    ----------
    points : ndarray, shape (number_of_points, 2)
        Physical surface coordinates in metres, ordered inlet to outlet.
    meridional_fraction : ndarray
        Uniform reference stations used to generate the surface points.

    Returns
    -------
    ndarray
        Signed curvature in inverse metres. Zero-speed samples return NaN.
        Tip differentiation is less accurate and is excluded from the reported
        comparison interval; these are numerical diagnostics, not constraints.
    """
    first_derivative = np.gradient(points, meridional_fraction, axis=0, edge_order=2)
    second_derivative = np.gradient(first_derivative, meridional_fraction, axis=0, edge_order=2)
    numerator = (first_derivative[:, 0] * second_derivative[:, 1]
                 - first_derivative[:, 1] * second_derivative[:, 0])
    denominator = np.linalg.norm(first_derivative, axis=1)**3
    return np.divide(numerator, denominator, out=np.full_like(numerator, np.nan),
                     where=denominator > 0)


def compare_asymmetry(configuration, output_directory):
    """Save a before/after comparison using the configured signed transfer.

    Parameters
    ----------
    configuration : dict
        Complete v3 case. ``comparison_number_of_points`` in ``numerics``
        sets the uniform differentiation grid. The reference uses asymmetry 0.
    output_directory : pathlib.Path
        Destination for PNG/SVG figures and a curvature CSV.

    Returns
    -------
    matplotlib.figure.Figure
        Three panels: physical blade overlay and curvature on each side.

    Notes
    -----
    Both constructions use exactly the same meridional grid and camberline.
    Curvature peaks are reported over 5%-100% of the asymmetry region, omitting
    the numerically sensitive tip. The total Denton thickness is identical.
    """
    import matplotlib.pyplot as plt

    number_of_points = configuration["numerics"]["comparison_number_of_points"]
    if not isinstance(number_of_points, int) or number_of_points < 21:
        raise ValueError("Use at least 21 comparison points.")
    meridional_fraction = np.linspace(0, 1, number_of_points)
    blade_type = configuration["blade_type"]
    path = create_meridional_path(blade_type, configuration[blade_type + "_geometry"], meridional_fraction)
    camberline = compute_camberline(path, configuration["camberline"])
    chord = path["meridional_chord"]
    extent = min(2 * configuration["thickness"]["leading_edge_fraction"], 1.0)
    if extent == 0:
        raise ValueError("Enable the leading-edge blend to compare its asymmetry.")
    region = (meridional_fraction >= 0.05 * extent) & (meridional_fraction <= extent)
    if np.count_nonzero(region) < 5:
        raise ValueError("Increase comparison_number_of_points to resolve the asymmetry region.")
    selected_asymmetry = configuration["thickness"]["leading_edge_asymmetry"]
    figure, axes = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
    csv_columns = [meridional_fraction]
    csv_names = ["meridional_fraction"]
    for label, asymmetry, color, linestyle in (
        ("Symmetric", 0.0, "0.5", "--"),
        (f"Asymmetry {selected_asymmetry:+g}", selected_asymmetry, "#0072B2", "-"),
    ):
        parameters = dict(configuration["thickness"], leading_edge_asymmetry=asymmetry)
        thickness = compute_denton_thickness(meridional_fraction, parameters)
        surfaces = apply_thickness(path, camberline, thickness["upper_half_thickness_ratio"],
                                    thickness["lower_half_thickness_ratio"])
        axes[0].plot(*(surfaces["boundary"] / chord).T, color=color, ls=linestyle,
                     lw=1.5, label=label)
        for side, axis in zip(("upper", "lower"), axes[1:]):
            curvature = chord * signed_surface_curvature(surfaces[side], meridional_fraction)
            axis.plot(meridional_fraction[region], curvature[region], color=color,
                      ls=linestyle, lw=1.5, label=label)
            peak = np.nanmax(np.abs(curvature[region]))
            print(f"{label:20s} {side:5s}: max |curvature| * chord = {peak:.4f}")
            csv_columns.append(curvature)
            prefix = "symmetric" if asymmetry == 0 and label == "Symmetric" else "adjusted"
            csv_names.append(f"{prefix}_{side}_curvature_times_chord")
    reference_points = map_to_physical_plane(path["meridional_distance"], camberline["azimuth"], path)
    axes[0].plot(*(reference_points / chord).T, ":", color="#6B4C9A", lw=1, label="Reference camberline")
    axes[0].set(title="Same total thickness, different nose split",
                xlabel=r"Horizontal coordinate / $c_{\mathrm{m}}$",
                ylabel=r"Vertical coordinate / $c_{\mathrm{m}}$")
    axes[0].set_aspect("equal", adjustable="box")
    for title, axis in zip(("Plus-side curvature", "Minus-side curvature"), axes[1:]):
        axis.axhline(0, color="0.65", lw=0.7)
        axis.axvline(configuration["thickness"]["leading_edge_fraction"], color="0.65", ls=":", lw=0.8)
        axis.set(title=title, xlabel=r"Meridional fraction $m/c_{\mathrm{m}}$",
                 ylabel=r"Signed curvature $\kappa c_{\mathrm{m}}$",
                 xlim=(0.05 * extent, extent))
    for axis in axes:
        axis.legend(fontsize=8)
        axis.grid(alpha=0.2)
    figure.suptitle(
        f"Leading-edge comparison; curvature interval {0.05*extent:.4f} <= m/chord <= {extent:.4f}",
        fontsize=11,
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_directory / f"{blade_type}_asymmetry_comparison.png", dpi=160)
    figure.savefig(output_directory / f"{blade_type}_asymmetry_comparison.svg")
    np.savetxt(output_directory / f"{blade_type}_curvature_comparison.csv", np.column_stack(csv_columns),
               delimiter=",", header=",".join(csv_names), comments="")
    return figure


def main():
    """Read the case, save the comparison, and optionally show its figure.

    Notes
    -----
    This script diagnoses the current case; it does not optimize or alter the
    asymmetry input and does not overwrite the YAML.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=HERE / "case.yaml")
    parser.add_argument("--output", type=Path, default=HERE / "output")
    parser.add_argument("--no-show", action="store_true")
    arguments = parser.parse_args()
    if arguments.no_show:
        import matplotlib
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    configuration = yaml.safe_load(arguments.config.read_text(encoding="utf-8"))
    figure = compare_asymmetry(configuration, arguments.output)
    if arguments.no_show:
        plt.close(figure)
    else:
        plt.show()


if __name__ == "__main__":
    main()
