"""Reproduce the numerical checks and figure in summary_and_way_forward.md.

Run from the TurboDash root with its Python environment. This is a geometry
audit, not a new blade generator or an aerodynamic validation.
"""

import importlib.util
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def load_version(version):
    name = f"summary_blade_{version}"
    spec = importlib.util.spec_from_file_location(
        name, ROOT / f"parametrization_{version}" / "blade_geometry.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def curvature(points, station):
    first = np.gradient(points, station, axis=0, edge_order=2)
    second = np.gradient(first, station, axis=0, edge_order=2)
    return (first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]) / np.linalg.norm(first, axis=1)**3


def main():
    v1, v2, v3 = (load_version(version) for version in ("v1", "v2", "v3"))
    config = yaml.safe_load((ROOT / "parametrization_v3/case.yaml").read_text(encoding="utf-8"))
    cases = yaml.safe_load((ROOT / "parametrization_v1/cases.yaml").read_text(encoding="utf-8"))["cases"]
    v1_checks = []
    for name, case in cases.items():
        camber = v1.make_camberline(case["camberline"])
        for model in v1.MODELS:
            profile = v1.ThicknessProfile(v1.ThicknessParameters(**case["thickness"]), model)
            blade = v1.make_blade(camber, profile)
            assert np.isfinite(blade.boundary).all()
            actual = 2 * np.max(blade.half_thickness) / camber.meridional_chord
            assert np.isclose(actual, profile.parameters.maximum_thickness_ratio)
            v1_checks.append({"case": name, "model": model, "normal_map_margin": blade.normal_map_margin})

    xi = np.linspace(0, 1, 40001)
    path = v3.create_meridional_path("axial", config["axial_geometry"], xi)
    camber = v3.compute_camberline(path, config["camberline"])
    thickness = v3.compute_denton_thickness(xi, config["thickness"])
    redistribution_error = np.max(np.abs(
        thickness["upper_half_thickness_ratio"] + thickness["lower_half_thickness_ratio"]
        - thickness["full_thickness_ratio"]
    ))
    assert redistribution_error < 1e-14
    v2_parameters = {key: value for key, value in config["thickness"].items() if key != "leading_edge_asymmetry"}
    symmetric = v3.compute_denton_thickness(xi, dict(config["thickness"], leading_edge_asymmetry=0.0))
    v2_thickness = v2.compute_denton_thickness(xi, v2_parameters)
    symmetric_surfaces = v3.apply_thickness(path, camber, symmetric["upper_half_thickness_ratio"], symmetric["lower_half_thickness_ratio"])
    v2_surfaces = v2.apply_thickness(path, camber, v2_thickness["full_thickness_ratio"])
    zero_asymmetry_error = np.max(np.abs(symmetric_surfaces["boundary"] - v2_surfaces["boundary"]))
    assert zero_asymmetry_error < 1e-14
    peak = np.argmax(thickness["full_thickness_ratio"])
    extent = min(2 * config["thickness"]["leading_edge_fraction"], 1)
    mask = (xi >= 0.05 * extent) & (xi <= extent)
    peaks = []
    for alpha in (0.0, 0.15):
        t = v3.compute_denton_thickness(xi, dict(config["thickness"], leading_edge_asymmetry=alpha))
        surfaces = v3.apply_thickness(path, camber, t["upper_half_thickness_ratio"], t["lower_half_thickness_ratio"])
        peaks.append({"asymmetry": alpha, **{
            side: float(np.max(np.abs(curvature(surfaces[side], xi)[mask])) * path["meridional_chord"])
            for side in ("upper", "lower")
        }})

    builds = []
    for version, module in (("v2", v2), ("v3", v3)):
        cfg = yaml.safe_load((ROOT / f"parametrization_{version}/case.yaml").read_text(encoding="utf-8"))
        for kind, geometry in (
            ("axial", cfg["axial_geometry"]),
            ("radial_outflow", cfg["radial_geometry"]),
            ("radial_inflow", {"inlet_radius": 0.035, "outlet_radius": 0.020}),
        ):
            blade = module.build_blade("axial" if kind == "axial" else "radial", geometry, cfg["camberline"], cfg["thickness"], 12, 801)
            assert np.isfinite(blade["surfaces"]["boundary"]).all()
            assert np.allclose(np.rad2deg(blade["camberline"]["metal_angle"][[0, -1]]),
                               [cfg["camberline"]["inlet_metal_angle_deg"], cfg["camberline"]["outlet_metal_angle_deg"]])
            builds.append(f"{version}: {kind}")

    # Isolate the coordinate choice on ONE common radial camberline. Constant
    # 2 mm full thickness deliberately exposes the radius-dependent scaling.
    # Flat end faces are used here, without any edge-law or nose comparison.
    rin, rout, full_t = 0.020, 0.035, 0.002
    chord = rout - rin
    prime_chord = np.log(rout / rin)
    radius = rin * np.exp(prime_chord * xi)
    beta = np.deg2rad(30)
    theta = np.tan(beta) * prime_chord * xi
    center = np.column_stack((radius * np.cos(theta), radius * np.sin(theta)))
    normal = np.column_stack((-np.sin(theta + beta), np.cos(theta + beta)))
    half_t = full_t / 2
    exact = (center + half_t * normal, center - half_t * normal)

    def mapped(dm, dtheta):
        rr = radius + dm
        return np.column_stack((rr * np.cos(theta + dtheta), rr * np.sin(theta + dtheta)))

    metric = (mapped(-half_t * np.sin(beta), half_t * np.cos(beta) / radius),
              mapped(half_t * np.sin(beta), -half_t * np.cos(beta) / radius))

    def conformal(a):
        sides = []
        for sign in (1, -1):
            rr = radius * np.exp(-sign * a * np.sin(beta))
            tt = theta + sign * a * np.cos(beta)
            sides.append(np.column_stack((rr * np.cos(tt), rr * np.sin(tt))))
        return sides

    local_conformal = conformal(half_t / radius)
    template_a = 0.5 * (full_t / chord) * prime_chord
    template = conformal(template_a)
    measured_template_t = np.linalg.norm(template[0] - template[1], axis=1)
    expected_template_t = 2 * radius * np.sqrt(np.sinh(template_a * np.sin(beta))**2 + np.sin(template_a * np.cos(beta))**2)
    assert np.allclose(measured_template_t, expected_template_t, rtol=1e-12, atol=1e-15)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), layout="constrained")
    styles = [(exact, "Exact physical normals", "#222222", "-"),
              (metric, "v2/v3 local metric rule", "#E69F00", "--"),
              (local_conformal, "Conformal normals, a = h/r", "#009E73", ":"),
              (template, "One constant-thickness conformal template", "#0072B2", "-")]
    for sides, label, color, linestyle in styles:
        boundary = np.vstack((sides[0], sides[1][::-1], sides[0][:1]))
        axes[0].plot(*(1000 * boundary).T, label=label, color=color, ls=linestyle, lw=1.5)
        axes[1].plot(xi, 1000 * np.linalg.norm(sides[0] - sides[1], axis=1), color=color, ls=linestyle, lw=1.5)
    axes[0].plot(*(1000 * center).T, color="0.65", lw=0.8)
    axes[0].set(xlabel="X [mm]", ylabel="Y [mm]", title="Same radial camberline; different offset rules")
    axes[0].set_aspect("equal")
    axes[1].set(xlabel="Conformal meridional fraction", ylabel="Paired-point physical separation [mm]", title="Radius scaling dominates the finite-offset effect")
    axes[0].legend(fontsize=7, loc="upper left")
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.suptitle("20 to 35 mm radial outflow; 30 degree metal angle; flat end faces")
    for suffix in ("png", "svg"):
        fig.savefig(HERE / f"coordinate_comparison.{suffix}", dpi=170)
    plt.close(fig)

    report = {
        "v1_cases_checked": v1_checks,
        "v2_v3_builds_checked": builds,
        "v3_zero_asymmetry_max_boundary_error_m": float(zero_asymmetry_error),
        "v3_total_thickness_redistribution_max_error": float(redistribution_error),
        "v3_actual_peak_ratio": float(thickness["full_thickness_ratio"][peak]),
        "v3_actual_peak_physical_meridional_fraction": float(xi[peak]),
        "v3_curvature_interval": [0.05 * extent, extent],
        "v3_curvature_peaks_times_meridional_chord": peaks,
        "radial_example": {
            "prime_chord": float(prime_chord),
            "logarithmic_mean_radius_mm": float(1000 * chord / prime_chord),
            "physical_to_prime_midstation": float(np.log((rin + rout) / (2 * rin)) / prime_chord),
            "prime_to_physical_midstation": float((np.sqrt(rin * rout) - rin) / chord),
            "template_paired_separation_at_endpoints_mm": (1000 * measured_template_t[[0, -1]]).tolist(),
            "max_physical_half_thickness_over_radius": float(half_t / rin),
            "local_conformal_max_paired_length_relative_error": float(np.max(np.abs(np.linalg.norm(local_conformal[0] - local_conformal[1], axis=1) / full_t - 1))),
        },
    }
    (HERE / "geometry_checks.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "v1_cases_checked"}, indent=2))
    print(f"Checked {len(v1_checks)} v1 case/model combinations.")


if __name__ == "__main__":
    main()
