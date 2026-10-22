"""Existing angle-interpolation camberline: location/concentration sensitivity.

Hold thickness inputs fixed for each blade family. No application changes.
"""
from copy import deepcopy
from pathlib import Path
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.integrate import quad

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
from turbodash.geometry.blade_parametrization import analysis as geometry
from turbodash.geometry.blade_parametrization.thickness import denton_body
from turbodash.geometry.blade_parametrization.geometry import turning_angle

SWEEPS = {"location": ("turning_location", (.3, .4, .5, .6, .7)),
          "concentration": ("turning_concentration", (1., 1.5, 2., 2.5, 3.))}
REFERENCE = {"turning_location": .5, "turning_concentration": 2.}
FULLNESS = {"impulse": (-.1, -.1), "reaction": (0., -.1)}
COLORS = ("#256995", "#5998ac", "#424b51", "#c28e41", "#b95532")


def blade(family, parameters, count=601):
    base = geometry.baseline(family, count=count, parameters=parameters)
    le, te = FULLNESS[family]
    return geometry.with_fullness(base, le, te)


def audit(b):
    result = geometry.audit(b, count=16001)
    assert result["positive_thickness"] and result["minimum_normal_map_margin"] > 0
    assert result["thickness_slope_sign_changes"] == 1
    assert not result["self_intersection"] and not result["neighbor_intersection"]
    p = b["parameters"]["camberline"]
    loc, q = p["turning_location"], p["turning_concentration"]
    angles = np.deg2rad([p["inlet_metal_angle_deg"], p["outlet_metal_angle_deg"]])
    actual = turning_angle(np.array([0., loc, 1.]), p)
    assert np.max(np.abs(actual-[angles[0], angles.mean(), angles[1]])) < 1e-14
    x = np.linspace(0, 1, 16001)
    # Independently evaluate the algebraic equation used in the note.
    first, second = (x*(1-loc))**q, ((1-x)*loc)**q
    weight = first/(first+second)
    beta = turning_angle(x, p)
    equation_error = float(np.max(np.abs(beta-(angles[0]+np.diff(angles)[0]*weight))))
    assert equation_error < 2e-14
    assert np.min(np.diff(beta)) > -1e-14
    chord = 1000*b["parameters"]["geometry"]["meridional_chord"]
    endpoint, _ = quad(lambda xx: float(np.tan(turning_angle(np.array([xx]), p)[0])),
                       0, 1, epsabs=1e-11, epsrel=1e-11)
    integration_error = abs(chord*endpoint-b["camber"][-1, 1])
    assert integration_error < 2e-5
    result.update(location=loc, concentration=q, endpoint_y_mm=float(b["camber"][-1, 1]),
                  camber_arclength_mm=float(b["s"][-1]), equation_error_radians=equation_error,
                  endpoint_integration_error_mm=float(integration_error))
    if b["family"] == "impulse" and loc == .5:
        assert abs(b["camber"][-1, 1]) < 1e-10
        assert np.max(np.abs(turning_angle(x, p)+turning_angle(1-x, p))) < 1e-13
    return result


def paired_figure(family, sweep, reference, candidates, limits):
    fig, axes = plt.subplots(5, 2, figsize=(12.5, 16.5), layout="constrained")
    fig.get_layout_engine().set(h_pad=.08, hspace=.035)
    for row, (b, color) in enumerate(zip(candidates, COLORS)):
        p = b["parameters"]["camberline"]
        loc, q = p["turning_location"], p["turning_concentration"]
        label = f"Location {loc:.2f} | Concentration {q:.2f}"
        if loc == .5 and q == 2.:
            label += " | Reference"
        axes[row, 0].set_title(label, loc="left", fontsize=11)
        axes[row, 1].set_title("Final blade | fixed thickness inputs", loc="left", fontsize=11)
        for col, ax in enumerate(axes[row]):
            original = reference["camber"] if col == 0 else geometry.outline(reference)
            modified = b["camber"] if col == 0 else geometry.outline(b)
            if col:
                ax.fill(modified[:, 0], modified[:, 1], color=color, alpha=.13)
                ax.plot(b["camber"][:, 0], b["camber"][:, 1], color=color, ls=":", lw=.9)
            ax.plot(original[:, 0], original[:, 1], color="#8d989f", ls="--", lw=1.3)
            ax.plot(modified[:, 0], modified[:, 1], color=color, lw=1.8)
            ax.set_aspect("equal", adjustable="box")
            ax.set(xlim=limits[0], ylim=limits[1], xlabel="Meridional distance [mm]",
                   ylabel="Circumferential distance [mm]")
            ax.grid(alpha=.16)
        half = np.array([np.interp(loc, b["xi"], b["camber"][:, k]) for k in (0, 1)])
        axes[row, 0].plot(*half, "o", color=color, ms=4)
    held = "concentration = 2" if sweep == "location" else "location = 0.50"
    fig.suptitle(f"{family.capitalize()} | Vary {sweep}; hold {held}\nCamberline on the left; final blade on the right", fontsize=13)
    fig.legend(handles=[Line2D([], [], color="#8d989f", ls="--", label="Reference: location 0.50, concentration 2"),
                        Line2D([], [], color=COLORS[0], label="Varied profile (row color)"),
                        Line2D([], [], color=".4", marker="o", ls="", label="Half-turning station")],
               loc="outside lower center", ncol=2, frameon=False, fontsize=9)
    fig.canvas.draw()
    scales = []
    for ax in axes.flat:
        pp = ax.transData.transform([[0, 0], [1, 0], [0, 1]])
        dx, dy = pp[1, 0]-pp[0, 0], pp[2, 1]-pp[0, 1]
        assert abs(dx/dy-1) < 1e-10
        scales.append(float(dx))
    assert np.ptp(scales) < 1e-8
    name = f"{family}_{sweep}"
    for ext in ("png", "svg"):
        fig.savefig(HERE/"figures"/f"{name}.{ext}", dpi=180)
    plt.close(fig)
    return dict(name=name, equal_aspect=True, common_pixels_per_mm=scales[0],
                x_limits_mm=limits[0], y_limits_mm=limits[1])


def turning_figure(parameters):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout="constrained")
    x = np.linspace(0, 1, 1201)
    for ax, (sweep, (key, values)) in zip(axes, SWEEPS.items()):
        for value, color in zip(values, COLORS):
            p = deepcopy(parameters["camberline"])
            p.update(REFERENCE)
            p[key] = value
            angles = np.deg2rad([p["inlet_metal_angle_deg"], p["outlet_metal_angle_deg"]])
            weight = (turning_angle(x, p)-angles[0])/(angles[1]-angles[0])
            ax.plot(x, weight, color=color, lw=1.7, label=f"{value:g}")
        ax.axhline(.5, color=".55", lw=.8, ls=":")
        ax.set(title=f"Vary {sweep}", xlabel="Normalized meridional position x", ylabel="Fraction of total angle change")
        ax.grid(alpha=.16)
        ax.legend(title=sweep.capitalize(), fontsize=9)
    fig.suptitle("The same turning-fraction law applies to both blade families", fontsize=13)
    for ext in ("png", "svg"):
        fig.savefig(HERE/"figures"/f"turning_fraction.{ext}", dpi=180)
    plt.close(fig)


def reaction_figures(reference, groups, limits):
    """Rows are locations; columns/overlaid colors are concentrations."""
    locations = SWEEPS["location"][1]
    concentrations = SWEEPS["concentration"][1]
    fig, axes = plt.subplots(5, 5, figsize=(15, 16.8), layout="constrained")
    original = geometry.outline(reference)
    for row, loc in enumerate(locations):
        for col, (b, color) in enumerate(zip(groups[loc], COLORS)):
            ax = axes[row, col]
            outline = geometry.outline(b)
            ax.fill(outline[:, 0], outline[:, 1], color=color, alpha=.12)
            ax.plot(original[:, 0], original[:, 1], color="#8d989f", ls="--", lw=1)
            ax.plot(outline[:, 0], outline[:, 1], color=color, lw=1.6)
            ax.plot(b["camber"][:, 0], b["camber"][:, 1], color=color, ls=":", lw=.8)
            ax.set_aspect("equal", adjustable="box")
            ax.set(xlim=limits[0], ylim=limits[1])
            ax.grid(alpha=.14)
            ax.tick_params(labelsize=8, labelleft=col == 0, labelbottom=row == 4)
            if row == 0:
                ax.set_title(f"Concentration {concentrations[col]:g}", fontsize=10)
            if col == 0:
                ax.set_ylabel(f"Location {loc:.2f}", fontsize=11)
    fig.supxlabel("Meridional distance [mm]", fontsize=11)
    fig.supylabel("Circumferential distance [mm]", fontsize=11)
    fig.suptitle("Reaction blades | Location by row, concentration by column\nFixed thickness inputs; dashed gray: reference location 0.50, concentration 2", fontsize=13)
    scales = []
    fig.canvas.draw()
    for ax in axes.flat:
        pp = ax.transData.transform([[0, 0], [1, 0], [0, 1]])
        dx, dy = pp[1, 0]-pp[0, 0], pp[2, 1]-pp[0, 1]
        assert abs(dx/dy-1) < 1e-10
        scales.append(float(dx))
    assert np.ptp(scales) < 1e-8
    for ext in ("png", "svg"):
        fig.savefig(HERE/"figures"/f"reaction_grid.{ext}", dpi=180)
    plt.close(fig)
    grid_record = dict(name="reaction_grid", equal_aspect=True, common_pixels_per_mm=scales[0])

    fig, axes = plt.subplots(5, 2, figsize=(12.5, 16.5), layout="constrained")
    fig.get_layout_engine().set(h_pad=.08, hspace=.035)
    for row, loc in enumerate(locations):
        for col, ax in enumerate(axes[row]):
            original = reference["camber"] if col == 0 else geometry.outline(reference)
            ax.plot(original[:, 0], original[:, 1], color="#8d989f", ls="--", lw=1.2)
            for b, color in zip(groups[loc], COLORS):
                curve = b["camber"] if col == 0 else geometry.outline(b)
                ax.plot(curve[:, 0], curve[:, 1], color=color, lw=1.5)
            ax.set_aspect("equal", adjustable="box")
            ax.set(xlim=limits[0], ylim=limits[1], xlabel="Meridional distance [mm]",
                   ylabel="Circumferential distance [mm]")
            ax.grid(alpha=.16)
        axes[row, 0].set_title(f"Location {loc:.2f} | Camberlines", loc="left", fontsize=11)
        axes[row, 1].set_title(f"Location {loc:.2f} | Final blade outlines", loc="left", fontsize=11)
    fig.suptitle("Reaction sensitivity | Each row fixes location; colors vary concentration", fontsize=13)
    handles = [Line2D([], [], color=color, label=f"Concentration {q:g}") for q, color in zip(concentrations, COLORS)]
    handles.append(Line2D([], [], color="#8d989f", ls="--", label="Reference: location 0.50, concentration 2"))
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False, fontsize=9)
    fig.canvas.draw()
    scales = []
    for ax in axes.flat:
        pp = ax.transData.transform([[0, 0], [1, 0], [0, 1]])
        dx, dy = pp[1, 0]-pp[0, 0], pp[2, 1]-pp[0, 1]
        assert abs(dx/dy-1) < 1e-10
        scales.append(float(dx))
    assert np.ptp(scales) < 1e-8
    for ext in ("png", "svg"):
        fig.savefig(HERE/"figures"/f"reaction_overlays.{ext}", dpi=180)
    plt.close(fig)
    return [grid_record, dict(name="reaction_overlays", equal_aspect=True, common_pixels_per_mm=scales[0])]


def main():
    (HERE/"figures").mkdir(exist_ok=True)
    (HERE/"results").mkdir(exist_ok=True)
    plt.rcParams.update({"font.size":10, "axes.spines.top":False, "axes.spines.right":False,
                         "svg.fonttype":"none", "path.simplify":False, "savefig.facecolor":"white"})
    report = dict(interpolation="angle", reference=REFERENCE, sweeps={k:list(v[1]) for k,v in SWEEPS.items()},
                  thickness_fullness=FULLNESS, base_points=601, baselines={}, cases=[], figures=[])
    for family in ("impulse", "reaction"):
        parameters = deepcopy(geometry.baseline(family, count=601)["parameters"])
        parameters["camberline"].update(REFERENCE)
        parameters["camberline"]["interpolation"] = "angle"
        reference = blade(family, parameters)
        report["baselines"][family] = dict(parameters=reference["parameters"], metrics=audit(reference))
        groups = {}
        locations = (.5,) if family == "impulse" else SWEEPS["location"][1]
        for loc in locations:
            candidates = []
            for q in SWEEPS["concentration"][1]:
                p = deepcopy(parameters)
                p["camberline"].update(turning_location=loc, turning_concentration=q)
                b = blade(family, p)
                r = audit(b)
                r["sweep"] = "concentration" if family == "impulse" else "location_and_concentration"
                # Fixed t(x), including the exact rounding/relief factors.
                for side in ("plus_offset", "minus_offset"):
                    assert np.array_equal(b[side], reference[side])
                report["cases"].append(r)
                candidates.append(b)
            groups[loc] = candidates
        curves = [geometry.outline(b) for group in groups.values() for b in group]
        xy = np.vstack(curves)
        chord = parameters["geometry"]["meridional_chord"]*1000
        limits = ((float(xy[:, 0].min()-.06*chord), float(xy[:, 0].max()+.06*chord)),
                  (float(xy[:, 1].min()-.06*chord), float(xy[:, 1].max()+.06*chord)))
        if family == "impulse":
            report["figures"].append(paired_figure(family, "concentration", reference, groups[.5], limits))
        else:
            report["figures"].extend(reaction_figures(reference, groups, limits))
        print(f"Completed {family}: {sum(len(v) for v in groups.values())} cases; fixed thickness, angles, integration and geometry checks passed.", flush=True)
    turning_figure(parameters)
    report["figures"].append(dict(name="turning_fraction"))
    (HERE/"results/sensitivity.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps([dict(family=r["family"], sweep=r["sweep"], location=r["location"], concentration=r["concentration"],
                           endpoint_y_mm=r["endpoint_y_mm"], length_mm=r["camber_arclength_mm"],
                           normal_map_margin=r["minimum_normal_map_margin"]) for r in report["cases"]], indent=2))


if __name__ == "__main__":
    main()
