"""Plot overview and detailed 5x3 camberline concentration studies.

Run with the repository's Python environment. PNG/SVG figures and a JSON
validation record are written beside this script, independently of the CWD.
Only the camberline is studied; no thickness or cascade model is evaluated.
"""

import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.integrate import cumulative_trapezoid, quad

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
from turbodash import set_plot_options
from turbodash.geometry.blade_parametrization import (
    ParameterState,
    resolve_parameters,
    turning_angle,
)

LOCATIONS = (0.3, 0.4, 0.5, 0.6, 0.7)
STUDIES = (
    {
        "name": "turning_sensitivity",
        "title": "Camberline sensitivity | overview",
        "concentrations": (1.0, 1.5, 2.0, 2.5, 3.0),
    },
    {
        "name": "turning_sensitivity_detail",
        "title": "Camberline sensitivity | concentration detail",
        "concentrations": (0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5),
    },
)
GEOMETRY = {"type": "axial", "meridional_chord": 0.015, "mean_radius": 0.05}
CAMBERLINE = {
    "inlet_metal_angle_deg": 0.0,
    "outlet_metal_angle_deg": 70.0,
    "interpolation": "angle",
}
NUMBER_OF_POINTS = 16001


def angle_gradient(xi, parameters):
    """Analytic d(beta)/d(xi), including infinite edge limits for 0 < q < 1.

    This is the derivative of the production angle-interpolation law. Its
    algebraic form avoids division by xi*(1-xi) at the edges. At q=1 the
    zero exponent correctly gives nonzero endpoint gradients.
    """
    location = parameters["turning_location"]
    concentration = parameters["turning_concentration"]
    a = (xi * (1 - location)) ** concentration
    b = ((1 - xi) * location) ** concentration
    with np.errstate(divide="ignore"):
        weight_gradient = (
            concentration
            * (location * (1 - location)) ** concentration
            * (xi * (1 - xi)) ** (concentration - 1)
            / (a + b) ** 2
        )
    return (
        np.deg2rad(
            parameters["outlet_metal_angle_deg"] - parameters["inlet_metal_angle_deg"]
        )
        * weight_gradient
    )


def sample_case(xi, location, concentration):
    """Use the production turning law, then integrate d(theta)/d(m')=tan(beta)."""
    if not np.isfinite(concentration) or concentration <= 0:
        raise ValueError("Turning concentration must be finite and positive.")
    parameters = resolve_parameters(
        ParameterState(
            {
                "geometry": GEOMETRY,
                "camberline": CAMBERLINE
                | {
                    "turning_location": location,
                    "turning_concentration": max(1.0, concentration),
                },
            }
        )
    ).values
    # The normal section builder requires q >= 1. Explore q < 1 directly with
    # the same angle law, without changing production validation or building
    # a finite-thickness blade with singular endpoint curvature.
    p = parameters["camberline"] | {"turning_concentration": concentration}
    geometry = parameters["geometry"]
    prime_chord = geometry["meridional_chord"] / geometry["mean_radius"]
    beta = turning_angle(xi, p)
    gradient = angle_gradient(xi, p)
    theta = prime_chord * cumulative_trapezoid(np.tan(beta), xi, initial=0)

    # Independent equation, differentiation and adaptive integration checks.
    inlet, outlet = np.deg2rad(
        [p["inlet_metal_angle_deg"], p["outlet_metal_angle_deg"]]
    )
    a = (xi * (1 - location)) ** concentration
    b = ((1 - xi) * location) ** concentration
    np.testing.assert_allclose(
        beta, inlet + (outlet - inlet) * a / (a + b), rtol=0, atol=2e-15
    )
    np.testing.assert_allclose(
        turning_angle([0, location, 1], p),
        [inlet, (inlet + outlet) / 2, outlet],
        rtol=0,
        atol=2e-15,
    )
    assert np.all(np.diff(beta) >= -2e-15)
    assert np.isfinite(theta).all() and np.isfinite(gradient[1:-1]).all()
    # Fractional q can limit endpoint smoothness, reducing finite-difference
    # accuracy there. Check the interior numerically and the edge limits exactly.
    interior = (xi >= 0.01) & (xi <= 0.99)
    gradient_error = float(
        np.max(
            np.abs(
                np.rad2deg(
                    gradient[interior] - np.gradient(beta, xi, edge_order=2)[interior]
                )
            )
        )
    )
    assert gradient_error < 1e-3
    edge_gradient = np.zeros(2)
    if concentration == 1:
        edge_gradient = (outlet - inlet) * np.array(
            [(1 - location) / location, location / (1 - location)]
        )
    elif concentration < 1:
        edge_gradient[:] = np.inf
    np.testing.assert_allclose(gradient[[0, -1]], edge_gradient, rtol=0, atol=2e-14)
    integrated_gradient, _ = quad(
        lambda x: float(angle_gradient(x, p)),
        0,
        1,
        epsabs=1e-11,
        epsrel=1e-11,
    )
    np.testing.assert_allclose(integrated_gradient, outlet - inlet, rtol=0, atol=1e-9)
    endpoint, _ = quad(
        lambda x: float(np.tan(turning_angle(x, p))),
        0,
        1,
        epsabs=1e-12,
        epsrel=1e-12,
    )
    endpoint_error = abs(theta[-1] - prime_chord * endpoint)
    assert endpoint_error < 1e-8
    if concentration == 1 and location == 0.5:
        np.testing.assert_allclose(
            beta, inlet + (outlet - inlet) * xi, rtol=0, atol=2e-15
        )
        # Independent closed-form camberline for this linear-angle case.
        exact_theta = (
            prime_chord / (outlet - inlet) * np.log(np.cos(inlet) / np.cos(beta))
        )
        np.testing.assert_allclose(theta, exact_theta, rtol=0, atol=1e-8)

    return {
        "parameters": p,
        "beta_deg": np.rad2deg(beta),
        "gradient_deg": np.rad2deg(gradient),
        "m_prime": prime_chord * xi,
        "theta": theta,
        "half_turning_theta": float(np.interp(location, xi, theta)),
        "checks": {
            "turning_location": location,
            "turning_concentration": concentration,
            "production_supported": concentration >= 1,
            "endpoint_gradient_behavior": (
                "unbounded" if concentration < 1 else "finite"
            ),
            "trailing_edge_theta_rad": float(theta[-1]),
            "integrated_angle_change_deg": float(np.rad2deg(integrated_gradient)),
            "max_gradient_error_deg": gradient_error,
            "endpoint_integration_error_rad": float(endpoint_error),
        },
    }


def make_figure(xi, cases, study):
    concentrations = study["concentrations"]
    color_positions = np.linspace(0.2, 0.8, len(concentrations))
    colors = plt.get_cmap("plasma")(color_positions)
    exploratory = any(q < 1 for q in concentrations)
    inlet = CAMBERLINE["inlet_metal_angle_deg"]
    outlet = CAMBERLINE["outlet_metal_angle_deg"]
    half_angle = (inlet + outlet) / 2
    header_height = 1.75 if exploratory else 1.3
    fig = plt.figure(
        figsize=(13.5, 3.2 * len(LOCATIONS) + header_height + 0.7),
        layout="constrained",
    )
    header, plot_area = fig.subfigures(
        2, 1, height_ratios=(header_height, 3.2 * len(LOCATIONS))
    )
    axes = plot_area.subplots(
        len(LOCATIONS),
        3,
        sharex="col",
        sharey="col",
        gridspec_kw={"width_ratios": (1, 1, 0.65)},
    )
    prime_chord = GEOMETRY["meridional_chord"] / GEOMETRY["mean_radius"]
    theta_max = max(case["theta"][-1] for group in cases for case in group)
    # Keep regular curves readable. The true q < 1 gradients are clipped by
    # the axes; upward markers identify branches exceeding this finite scale.
    gradient_max = max(
        case["gradient_deg"].max()
        for group in cases
        for case in group
        if case["parameters"]["turning_concentration"] >= 1
    )
    gradient_limit = 1.05 * gradient_max
    for row, (location, group) in enumerate(zip(LOCATIONS, cases)):
        for case, color in zip(group, colors):
            q = case["parameters"]["turning_concentration"]
            linestyle = "--" if q < 1 else "-"
            axes[row, 0].plot(xi, case["beta_deg"], color=color, ls=linestyle)
            finite_gradient = np.where(
                np.isfinite(case["gradient_deg"]), case["gradient_deg"], np.nan
            )
            axes[row, 1].plot(xi, finite_gradient, color=color, ls=linestyle)
            axes[row, 2].plot(case["m_prime"], case["theta"], color=color, ls=linestyle)
            if q < 1:
                gradient = case["gradient_deg"]
                above_limit = gradient > gradient_limit
                crossings = np.flatnonzero(above_limit[:-1] != above_limit[1:])
                for index in crossings:
                    if np.isfinite(gradient[index : index + 2]).all():
                        station = xi[index] + (xi[index + 1] - xi[index]) * (
                            (gradient_limit - gradient[index])
                            / (gradient[index + 1] - gradient[index])
                        )
                    else:
                        # A divergence within the first/last sampling interval
                        # is indicated at the endpoint without interpolating inf.
                        station = (
                            xi[index]
                            if not np.isfinite(gradient[index])
                            else xi[index + 1]
                        )
                    axes[row, 1].plot(
                        station, 0.985 * gradient_limit, "^", color=color, ms=5
                    )
            axes[row, 2].plot(
                prime_chord * location,
                case["half_turning_theta"],
                "o",
                ms=3,
                color=color,
            )
        for col in (0, 1):
            axes[row, col].axvline(location, color="0.6", ls=":", lw=0.8)
            axes[row, col].set_xlim(0, 1)
            axes[row, col].set_xticks(np.linspace(0, 1, 5))
        axes[row, 0].axhline(half_angle, color="0.6", ls=":", lw=0.8)
        axes[row, 0].set_ylim(inlet - 2, outlet + 2)
        axes[row, 0].set_yticks((inlet, half_angle, outlet))
        axes[row, 0].set_ylabel(
            rf"Half-turning location $\ell={location:.1f}$" + "\n" + r"$\beta$ [deg]"
        )
        axes[row, 0].plot(
            location, half_angle, "o", ms=5, mfc="white", mec="0.25", zorder=5
        )
        axes[row, 0].annotate(
            rf"$\beta(\ell)={half_angle:g}^\circ$ for every $q$",
            xy=(location, half_angle),
            xytext=(0.54, 0.17) if location <= 0.5 else (0.07, 0.75),
            textcoords="axes fraction",
            fontsize=9,
            color="0.2",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9},
            arrowprops={"arrowstyle": "->", "color": "0.35", "lw": 0.8},
        )
        axes[row, 1].set_ylim(0, gradient_limit)
        axes[row, 1].set_ylabel(r"$d\beta/d\xi$ [deg]")
        axes[row, 2].set(
            xlim=(-0.02 * prime_chord, 1.02 * prime_chord),
            ylim=(-0.02 * theta_max, 1.04 * theta_max),
            ylabel=r"Tangential angle $\theta$ [rad]",
        )
        # Both conformal coordinates are in radians: preserve blade shape.
        axes[row, 2].set_aspect("equal", adjustable="box")
        axes[row, 2].set_xticks(np.linspace(0, prime_chord, 3))
        for ax in axes[row]:
            ax.grid(alpha=0.18, lw=0.6)

        # The centered row shows the concentration effect without location skew.
        if location == 0.5:
            linear_index = concentrations.index(1.0)
            linear_case = group[linear_index]
            axes[row, 0].annotate(
                r"$q=1.0$: linear $\beta(\xi)$ only" + "\n" + r"when $\ell=0.5$",
                xy=(0.25, np.interp(0.25, xi, linear_case["beta_deg"])),
                xytext=(0.04, 0.82),
                textcoords="axes fraction",
                fontsize=9,
                color=colors[linear_index],
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9},
                arrowprops={
                    "arrowstyle": "->",
                    "color": colors[linear_index],
                    "lw": 0.9,
                },
            )
            for index, text, position, station in (
                (
                    linear_index,
                    "Distributed turning",
                    (0.08, 0.58 if exploratory else 0.44),
                    0.18,
                ),
                (-1, "Concentrated turning", (0.58, 0.87), 0.5),
            ):
                concentration = group[index]["parameters"]["turning_concentration"]
                axes[row, 1].annotate(
                    rf"$q={concentration:.1f}$" + "\n" + text,
                    xy=(station, np.interp(station, xi, group[index]["gradient_deg"])),
                    xytext=position,
                    textcoords="axes fraction",
                    fontsize=9,
                    color=colors[index],
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9},
                    arrowprops={"arrowstyle": "->", "color": colors[index], "lw": 0.9},
                )
    for ax, title in zip(
        axes[0],
        (
            "Metal-angle distribution",
            "Turning gradient",
            "Integrated camberline",
        ),
    ):
        ax.set_title(title, fontsize=12, pad=10)
    for ax in axes[-1, :2]:
        ax.set_xlabel(r"Normalized meridional position $\xi=m'/c'$ [-]")
    axes[-1, 2].set_xlabel(r"Conformal position $m'$ [rad]")
    fig.suptitle(
        study["title"] + " | "
        rf"$\beta_{{\rm in}}={inlet:g}^\circ$, $\beta_{{\rm out}}={outlet:g}^\circ$",
        fontsize=14,
    )
    header.text(
        0.5,
        0.93,
        rf"Half-turning location $\ell$: where $\beta={half_angle:g}^\circ$"
        " (half the total angle change)",
        ha="center",
        va="top",
        fontsize=11.5,
    )
    header.text(
        0.5,
        0.76 if exploratory else 0.71,
        r"Turning concentration $q$: larger $q$ sharpens the angle transition around $\ell$",
        ha="center",
        va="top",
        fontsize=11.5,
    )
    if exploratory:
        header.text(
            0.5,
            0.59,
            r"Dashed $q<1$: unbounded endpoint gradients; upward markers show clipped branches",
            ha="center",
            va="top",
            fontsize=10.5,
            color="0.25",
        )
    handles = [
        Line2D(
            [],
            [],
            color=color,
            lw=2.5,
            ls="--" if q < 1 else "-",
            label=rf"$q={q:.1f}$",
        )
        for q, color in zip(concentrations, colors)
    ]
    handles.append(
        Line2D(
            [],
            [],
            color="0.5",
            marker="o",
            ls="",
            ms=5,
            label=rf"Half-turning station ($\beta={half_angle:g}^\circ$)",
        )
    )
    legend = header.legend(
        handles=handles,
        title="Turning concentration",
        ncol=len(handles),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        fontsize=11,
        title_fontsize=12,
        handlelength=2.0 if exploratory else 2.4,
        columnspacing=1.0 if exploratory else 1.5,
        borderpad=0.8,
        frameon=True,
        framealpha=1,
        facecolor="#f6f5fa",
        edgecolor="#514b61",
    )
    legend.get_title().set_fontweight("bold")
    return fig


def main():
    set_plot_options(fontsize=11, linewidth=1.25)
    plt.rcParams.update(
        {
            "svg.fonttype": "none",
            "path.simplify": False,
            "savefig.facecolor": "white",
        }
    )
    # Cluster stations near both ends to resolve fractional-q behavior without
    # relaxing the integration accuracy or representing infinity as zero.
    xi = 0.5 * (1 - np.cos(np.linspace(0, np.pi, NUMBER_OF_POINTS)))
    (HERE / "figures").mkdir(exist_ok=True)
    (HERE / "results").mkdir(exist_ok=True)
    for study in STUDIES:
        concentrations = study["concentrations"]
        cases = [
            [sample_case(xi, location, q) for q in concentrations]
            for location in LOCATIONS
        ]
        fig = make_figure(xi, cases, study)
        for extension in ("png", "svg"):
            path = HERE / "figures" / f"{study['name']}.{extension}"
            fig.savefig(path, dpi=200)
            print(f"Saved {path}")
        plt.close(fig)
        report = {
            "fixed_camberline": CAMBERLINE,
            "geometry": GEOMETRY,
            "locations": LOCATIONS,
            "concentrations": concentrations,
            "exploratory_concentrations": [q for q in concentrations if q < 1],
            "production_minimum_concentration": 1.0,
            "colormap": "plasma",
            "color_positions": np.linspace(0.2, 0.8, len(concentrations)).tolist(),
            "sample_count": NUMBER_OF_POINTS,
            "sampling": "cosine-spaced",
            "cases": [case["checks"] for group in cases for case in group],
        }
        (HERE / "results" / f"{study['name']}.json").write_text(
            json.dumps(report, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        print(
            f"{study['name']}: all {len(report['cases'])} cases passed angle, derivative "
            "and camberline integration checks."
        )
    print("q=1 is linear in beta only at location 0.5; all camberlines are curved.")
    print(
        "Dashed q<1 curves are exploratory: endpoint gradients and curvature diverge."
    )


if __name__ == "__main__":
    main()
