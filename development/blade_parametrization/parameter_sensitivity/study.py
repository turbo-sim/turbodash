"""Rebuild the GUI parameter gallery using the production blade kernel.

Run from the repository root with the project Python environment. No shape law
is implemented here: this file selects independent inputs, measures production
outputs, and renders them. See README.md for the output contract.
"""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))

from turbodash import set_plot_options
from turbodash.geometry.blade_parametrization.geometry import (
    build_section, map_section_coordinates,
)
from turbodash.geometry.blade_parametrization.parameters import (
    FIELD_SPECS, ParameterState, load_case, resolve_parameters,
)
from turbodash.geometry.blade_parametrization.passage import section_passage
from turbodash.geometry.blade_parametrization import turbomoc_adapter as adapter
from turbodash.geometry.meanline import REACTION_PRESET, IMPULSE_PRESET

ASSETS = REPO / "docs/source/_static/blade_sensitivity"
GALLERY = REPO / "docs/source/guides/_blade_geometry_sweeps.inc"
FAMILIES = ("reaction", "impulse")
COLORMAP = "plasma"
COLORMAP_INTERVAL = (.25, .75)
FIXED_GUI_INPUTS = {"cascade.blade_count": "Held at the reference count by user request; no blade-count sweep."}


@dataclass(frozen=True)
class Sweep:
    path: str
    values: tuple
    meaning: str
    effect: str
    coupling: str
    rationale: str
    context: str = "axial"
    supplement: str = ""
    profile: bool = True


# Scale sweeps use multipliers; location/fullness/angle sweeps use increments.
# A None entry selects that family's exact baseline, not a rounded product.
# Ratios, fractions and exponents are dimensionless. Stored lengths are SI.
SWEEPS = (
    Sweep("geometry.type", ("axial", "radial"),
          "Select a cylinder or a radial plane; Meanline sets this read-only field in the editor.",
          "Changes the physical mapping. The radial outline is in the machine X-Y plane; the axial outline is developed from its cylinder.",
          "Only type changes: radial radii remain 20 and 35 mm, while the axial chord is retained as an inactive input. This is a mapping comparison, not an equal-size blade comparison.",
          "Both supported surfaces, with all other independent inputs fixed.", supplement="channel"),
    Sweep("geometry.mean_radius", (.9, .95, None, 1.1),
          "Radius midway between axial hub and shroud.",
          "The developed 2D profile is unchanged. Pitch grows with radius and the channel moves radially.",
          "The hub must remain positive. A cylinder is an isometry when developed; a chart-coordinate change is not a physical outline change.",
          "90%, 95%, 100% and 110% of the reference radius; count remains fixed.", supplement="neighbors", profile=False),
    Sweep("geometry.blade_height", (.9, .95, None, 1.1),
          "Span between hub and shroud, or separation of radial endwalls.",
          "Changes span; the supported untwisted section has the same developed 2D shape.",
          "Independent of chord and count. Axial hub radius must stay positive; this model does not add spanwise twist.",
          "90%, 95%, 100% and 110% of the 10 mm reference span.", supplement="channel", profile=False),
    Sweep("geometry.meridional_chord", (.9, .95, None, 1.1),
          "Axial separation of camberline endpoints.",
          "Scales both camber and nominal thickness in physical units; rounded edges can protrude beyond the endpoints.",
          "All thickness ratios remain fixed, so dimensional thickness changes naturally. Pitch stays fixed and large chords can collide with neighbors. Inactive for radial geometry.",
          "90%, 95%, 100% and 110% of the reference chord, at fixed count and radius."),
    Sweep("geometry.inlet_radius", (.9, .95, None, 1.1),
          "Radius at the radial camberline inlet.",
          "Changes radial chord, conformal scale and mapped curvature and thickness.",
          "Outlet radius stays 35 mm; equal inlet/outlet radii are rejected. This field is hidden for an axial cylinder.",
          "18, 19, 20 and 22 mm: a local comparison around 20 mm, away from equal radii.", context="radial"),
    Sweep("geometry.outlet_radius", (.9, .95, None, 1.1),
          "Radius at the radial camberline outlet.",
          "Changes radial extent and the radius-dependent physical thickness.",
          "Inlet radius stays 20 mm. The inactive axial chord is not adjusted; equal radii are rejected.",
          "31.5, 33.25, 35 and 38.5 mm: a local comparison around 35 mm.", context="radial"),
    Sweep("camberline.inlet_metal_angle_deg", (-5., -2.5, None, 5.),
          "Signed inlet tangent angle measured from downstream meridional direction.",
          "Rotates the inlet tangent and changes turning, arclength and circumferential displacement.",
          "Outlet angle and all shape inputs stay fixed. Neither preset nor pressure-side selection is manually changed when an angle changes.",
          "Baseline minus 5 and 2.5 degrees, the baseline, and baseline plus 5 degrees."),
    Sweep("camberline.outlet_metal_angle_deg", (-5., -2.5, None, 5.),
          "Signed outlet tangent angle measured from downstream meridional direction.",
          "Moves the trailing endpoint circumferentially and changes total turning and blade length.",
          "The nominal thickness law is unchanged, but its physical orientation and curvature change. No aerodynamic deviation is predicted.",
          "Baseline minus 5 and 2.5 degrees, the baseline, and baseline plus 5 degrees."),
    Sweep("camberline.interpolation", ("angle", "slope"),
          "Blend metal angles directly, or blend their tangents before converting to angles.",
          "Changes the interior camberline while retaining both endpoint angles.",
          "Half-turning location means half the angle change in angle mode and half the tangent change in slope mode.",
          "Both current GUI choices are included."),
    Sweep("camberline.turning_location", (-.05, -.025, None, .05),
          "Conformal meridional station at half the selected turning interpolation.",
          "Moves turning forward or aft and changes the circumferential endpoint position.",
          "This is neither camber arclength fraction nor necessarily the maximum-curvature station. Concentration remains fixed.",
          "Baseline offsets -0.05, -0.025, 0 and +0.05 keep the turning station close to its preset."),
    Sweep("camberline.turning_concentration", (.9, .95, None, 1.1),
          "Concentrate the tangent change around the turning location.",
          "Higher values create straighter end regions and sharper turning in between.",
          "High curvature can fold a thick normal offset. Intersections are rejected rather than repaired by reducing thickness.",
          "90%, 95%, 100% and 110% of the preset concentration, all above the minimum of one."),
    Sweep("thickness.maximum_thickness_ratio", (.9, .95, None, 1.1),
          "Nominal full body thickness divided by conformal meridional chord.",
          "Adds or removes central body thickness without moving the camberline.",
          "The final physical maximum and its station can differ after rounding, relief and radial mapping. Ratios are full thickness, not half-thickness.",
          "90%, 95%, 100% and 110% of the reference body ratio show modest thinning and thickening."),
    Sweep("thickness.maximum_thickness_location", (-.05, -.025, None, .05),
          "Conformal meridional station of the nominal body thickness value.",
          "Moves the thick region forward or aft while holding endpoint body values fixed.",
          "The actual maximum need not occur here, especially with unequal endpoint body thickness. It is not arclength fraction.",
          "Baseline offsets -0.05, -0.025, 0 and +0.05 show a local displacement of the thick region."),
    Sweep("thickness.leading_edge_thickness_ratio", (.9, .95, None, 1.1),
          "Full unrounded body thickness at the leading endpoint, divided by conformal chord.",
          "Changes the nose and the linear contribution to the body thickness.",
          "This is not tip radius. With rounding enabled the final tip closes; the rounding fraction and exponent remain fixed.",
          "90%, 95%, 100% and 110% of the unrounded reference value; no degenerate zero-body tip."),
    Sweep("thickness.trailing_edge_thickness_ratio", (.9, .95, None, 1.1),
          "Full unrounded body thickness at the trailing endpoint, divided by conformal chord.",
          "Changes the trailing shoulder and the body baseline.",
          "A rounded trailing tip still closes to zero width; this is not a finite blunt trailing-edge thickness or an edge-radius prescription.",
          "90%, 95%, 100% and 110% of the unrounded reference trailing value."),
    Sweep("thickness.leading_edge_fraction", (.8, .9, None, 1.2),
          "Conformal chord fraction over which the leading tip closes into the body.",
          "Larger fractions spread rounding downstream. Zero leaves an unrounded endpoint joined by a straight cap.",
          "The leading relief window also depends on this fraction. That dependent change is allowed; relief strength is held fixed. Fraction is not radius.",
          "80%, 90%, 100% and 120% of the reference fraction retain the small preset rounding extent.", supplement="leading_zoom"),
    Sweep("thickness.trailing_edge_fraction", (.8, .9, None, 1.2),
          "Conformal chord fraction measured backward from the trailing tip.",
          "Controls the extent of trailing closure; zero produces a straight closing cap.",
          "Rounding factors can overlap if their fractions sum above one. The current GUI has no trailing-edge relief control.",
          "80%, 90%, 100% and 120% of the reference trailing rounding fraction.", supplement="trailing_zoom"),
    Sweep("thickness.leading_edge_fullness", (-.1, -.05, None, .1),
          "Independent rational modification of the leading reference shoulder.",
          "Negative values narrow and positive values fill the shoulder; zero at both ends recovers the Denton body.",
          "The modification has global support. The trailing fullness is fixed, and some opposed pairs fail the joint rational-denominator check.",
          "Baseline offsets -0.1, -0.05, 0 and +0.1 explore local shoulder changes inside the individual bounds."),
    Sweep("thickness.trailing_edge_fullness", (-.1, -.05, None, .1),
          "Independent rational modification of the trailing reference shoulder.",
          "Changes the trailing shoulder, with some influence elsewhere along the body.",
          "Leading fullness is never linked to this sweep. Symmetry generally breaks in the impulse example when only one shoulder changes.",
          "Baseline offsets -0.1, -0.05, 0 and +0.1; the opposite shoulder is fixed."),
    Sweep("thickness.shape_exponent", (.9, .95, None, 1.1),
          "Broadness of the central Denton body contribution.",
          "Larger exponents broaden the central thick region in its transformed coordinate.",
          "At 1 the peak can have a corner; between 1 and 2 curvature can be singular. Warnings do not imply an intersection rejection.",
          "1.8, 1.9, 2 and 2.2 bracket the preset without reaching the corner limit at one."),
    Sweep("thickness.leading_edge_rounding_exponent", (.9, .95, None, 1.1),
          "Rate at which leading rounding recovers unrounded body thickness.",
          "Higher values recover thickness faster inside the same rounding interval.",
          "Separate from the body exponent. A value of 2 can produce a curvature jump at the rounding shoulder; tip radius also changes.",
          "2.7, 2.85, 3 and 3.3 compare small changes around the preset three.", supplement="leading_zoom"),
    Sweep("thickness.trailing_edge_rounding_exponent", (.9, .95, None, 1.1),
          "Rate at which the trailing tip approaches the unrounded body.",
          "Changes the local trailing closure without changing the camberline.",
          "Neither exponent nor rounding fraction directly prescribes a circular cap or wedge angle.",
          "2.7, 2.85, 3 and 3.3 match the local leading exponent comparison.", supplement="trailing_zoom"),
    Sweep("leading_edge.relief_strength", (-.025, -.0125, None, .025),
          "Remove thickness near the leading edge from the automatically inferred pressure side.",
          "Changes one actual surface and reduces the plotted half of total paired thickness; the camberline remains unchanged.",
          "Turning direction selects the side. Relief is inactive at zero rounding extent and cannot automatically choose a side for nonzero relief on a zero-turning blade.",
          "Reaction: 0.125, 0.1375, 0.15 and 0.175. Impulse: 0, 0.0125, 0.025 and 0.05, since its baseline lies at the lower bound.", supplement="leading_zoom"),
    Sweep("correction.preserve_physical_thickness", (False, True),
          "Rescale chart offsets to approximately retain the intended physical paired thickness as radius changes.",
          "Changes radial outlines and thickness; on an axial cylinder the factor is identically one.",
          "The correction is first order in thickness/radius. Residual error can remain after nonlinear mapping; it does not enforce a minimum distance between complete surfaces.",
          "Both Boolean choices on the documented radial example.", context="radial", supplement="error"),
    Sweep("correction.warning_tolerance", (.75, .9, None, 1.25),
          "Relative mapped-thickness discrepancy at which correction issues a warning.",
          "Changes diagnostics only, not geometry. Thresholds and physical-thickness errors are recorded in the manifest.",
          "Only active with physical correction enabled. This sweep uses a separate, explicitly corrected radial baseline; the axial cylinder has effectively zero error.",
          "0.75%, 0.9%, 1% and 1.25% compare thresholds close to the preset one percent.", context="radial_corrected", supplement="error", profile=False),
    Sweep("flow_domain.inlet_chords", (.8, .9, None, 1.2),
          "Upstream extension requested from the production downstream passage adapter.",
          "Changes the flow-domain boundary while the blade and camberline remain fixed.",
          "Extensions use the downstream adapter's reference chord and automatic passage placement; no replacement rectangular domain is constructed here.",
          "80%, 90%, 100% and 120% of the preset upstream extension.", supplement="domain", profile=False),
    Sweep("flow_domain.outlet_chords", (.8, .9, None, 1.2),
          "Downstream extension requested from the production passage adapter.",
          "Moves downstream domain boundaries without changing the blade outline.",
          "The inlet extension stays fixed. Automatic passage placement can move other domain boundaries as a dependent result.",
          "80%, 90%, 100% and 120% of the preset downstream extension.", supplement="domain", profile=False),
    Sweep("numerics.number_of_points", (501, 551, None, 701),
          "Base stations for geometry sampling and sampled validation.",
          "Changes discretization, not the analytic shape law. The manifest records convergence against a 2401-base-point reference.",
          "Additional edge/shoulder stations make actual sample count larger. Low resolution can miss crossings and distort curvature; it is not a shape-control substitute.",
          "501, 551, 601 and 701 base points compare nearby resolutions around the preset 601.", supplement="sampling", profile=False),
)


def lookup(values, path):
    group, key = path.split(".")
    return values[group][key]


def assign(values, path, value):
    group, key = path.split(".")
    values[group][key] = value


def flat(values):
    return {path: lookup(values, path) for path in FIELD_SPECS}


def hidden_gui_paths():
    """Read literal UI visibility metadata without importing the Dash app."""
    tree = ast.parse((REPO / "turbodash/apps/shared/components/blade_modeller.py").read_text(encoding="utf-8"))
    return next(ast.literal_eval(node.value) for node in tree.body
                if isinstance(node, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "HIDDEN_PATHS" for t in node.targets))


def baselines():
    result, records = {}, {}
    for family, preset in zip(FAMILIES, (REACTION_PRESET, IMPULSE_PRESET)):
        relative = f"turbodash/geometry/blade_parametrization/cases/{family}.yaml"
        reference = resolve_parameters(load_case(REPO / relative)).values
        values = deepcopy(reference)
        for path, value in preset.items():
            assign(values, path, value)
        adjustments = {path: {"reference": lookup(reference, path), "preset": value}
                       for path, value in preset.items() if lookup(reference, path) != value}
        for context in ("axial", "radial", "radial_corrected"):
            p = deepcopy(values)
            p["geometry"]["type"] = "axial" if context == "axial" else "radial"
            p["correction"]["preserve_physical_thickness"] = context == "radial_corrected"
            p = resolve_parameters(ParameterState(p)).values
            section = build_section(p)
            result[family, context] = section
            records[f"{family}/{context}"] = {
                "reference_case": relative, "reference_inputs": reference,
                "preset_adjustments": adjustments, "inputs": p,
                "diagnostics": section.diagnostics, "warnings": section.warnings,
                "coordinate_sha256": coordinate_hash(section),
            }
    return result, records


def coordinate_hash(section):
    digest = hashlib.sha256()
    for name in ("xi", "camber", "plus", "minus"):
        digest.update(np.asarray(getattr(section, name), dtype="<f8").tobytes())
    return digest.hexdigest()


def sweep_values(sweep, baseline):
    base = lookup(baseline.parameters, sweep.path)
    path = sweep.path
    if FIELD_SPECS[path]["type"] in {"enum", "bool"}:
        values = list(sweep.values)
    elif path == "numerics.number_of_points":
        values = [base if v is None else v for v in sweep.values]
    elif path == "leading_edge.relief_strength" and base == 0:
        # A separate, explicit one-sided range at a boundary, never clipping
        # attempted negative values or modifying another input to rescue them.
        values = [0., .0125, .025, .05]
    elif path.endswith("metal_angle_deg") or path in {
            "camberline.turning_location", "thickness.maximum_thickness_location",
            "thickness.leading_edge_fullness", "thickness.trailing_edge_fullness",
            "leading_edge.relief_strength"}:
        values = [base + (v or 0) for v in sweep.values]
    else:
        values = [base if v is None else base*v for v in sweep.values]
    assert base in values, (path, base, values)
    assert len(values) == len(set(values))
    assert len(values) == (len(sweep.values) if FIELD_SPECS[path]["type"] in {"enum", "bool"} else 4)
    return values


def physical(section, name):
    """Only unit conversion/development; radial mapping belongs to the kernel."""
    if section.parameters["geometry"]["type"] == "axial":
        return 1000 * section.radius[0] * np.asarray(getattr(section, name))
    return 1000 * map_section_coordinates(section)[name][:, :2]


def arclength(section):
    points = physical(section, "camber")
    return np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]


def panel_arrays(section, baseline):
    # Each station uses the same baseline arclength coordinate even when angle,
    # chord, radius, or sampling changes. This is not the swept camber's length.
    s_ref = np.interp(section.xi, baseline.xi, arclength(baseline))
    return (physical(section, "camber"),
            np.column_stack((s_ref, 500 * section.thickness["measured_physical"])),
            physical(section, "boundary"))


def production_domain(section):
    """Same passage construction and containment check as the app preview."""
    modules = adapter._modules()
    mapping = adapter._mapping(section, modules)
    contour = adapter._q(section.boundary, mapping.reference_radius)
    blade = {"provider": "parametrization_blades", "pitch": mapping.pitch,
             "blade_curve": {"x": contour[:, 0].tolist(), "y": contour[:, 1].tolist()},
             "trailing_edge": {"x": [], "y": []},
             "metal_angle_in": -section.parameters["camberline"]["inlet_metal_angle_deg"],
             "metal_angle_out": -section.parameters["camberline"]["outlet_metal_angle_deg"]}
    settings = dict(section.parameters["flow_domain"], domain_shift_mode="auto", domain_shift=0., num_points=100)
    curves, diagnostics = adapter._center_passage(modules, blade, settings, mapping.reference_chord * 1e-4)
    result = {name: -np.asarray(curves[name])[:, ::-1]
              for name in ("periodic_left", "periodic_right", "inlet", "outlet")}
    diagnostics.update(reference_chord_mm=float(mapping.reference_chord),
                       shift_mm=float(curves["shift_mm"]),
                       backend="turbomoc_adapter._center_passage (full contour containment checked)")
    return result, diagnostics


def evaluate(sweep, value, baseline):
    requested = deepcopy(baseline.parameters)
    assign(requested, sweep.path, value)
    changed = [path for path in FIELD_SPECS if lookup(requested, path) != lookup(baseline.parameters, path)]
    assert changed == ([] if value == lookup(baseline.parameters, sweep.path) else [sweep.path])
    record = {"value": value, "is_baseline": not changed, "changed_independent_paths": changed,
              "requested_inputs": requested, "resolved_inputs": None,
              "validation": {"status": "rejected", "stage": "parameters", "error": None}}
    section, extra = None, {}
    try:
        resolved = resolve_parameters(ParameterState(baseline.parameters, {sweep.path: value}))
        record["resolved_inputs"] = resolved.values
        assert flat(resolved.values) == flat(requested), "Resolver changed another independent input"
        record["fingerprint"] = resolved.fingerprint
        record["validation"]["stage"] = "sampled_geometry"
        section = build_section(resolved.values)
        record.update(diagnostics=section.diagnostics, warnings=section.warnings,
                      coordinate_sha256=coordinate_hash(section))
        if not changed:
            # Every sweep's baseline must exactly reproduce the direct kernel.
            assert coordinate_hash(section) == coordinate_hash(baseline)
            record["baseline_max_coordinate_error_m"] = 0.
        # Independently check the declared physical measurement against mapped
        # radial points / developed axial samples, without reproducing a law.
        measured = np.linalg.norm(physical(section, "plus") - physical(section, "minus"), axis=1)
        np.testing.assert_allclose(measured, 1000*section.thickness["measured_physical"], atol=2e-11)
        if sweep.supplement == "domain":
            record["validation"]["stage"] = "flow_domain"
            extra["domain"], record["domain_diagnostics"] = production_domain(section)
        if sweep.supplement == "neighbors":
            passage = section_passage(section)
            record["passage"] = {key: passage[key] for key in ("available", "message", "attempted_count", "valid_count")}
            record["passage"]["minimum_valid_cut_width_mm"] = (float(np.min(passage["widths_mm"])) if passage["valid_count"] else None)
        if sweep.supplement == "sampling":
            dense_parameters = deepcopy(section.parameters)
            dense_parameters["numerics"]["number_of_points"] = 2401
            dense = build_section(dense_parameters, check_intersections=False)
            record["sampling_reference"] = {"base_points": 2401, "sample_count": len(dense.xi),
                "maximum_thickness_difference_mm": section.diagnostics["actual_maximum_thickness_mm"] - dense.diagnostics["actual_maximum_thickness_mm"],
                "normal_margin_difference": section.diagnostics["min_normal_map_margin"] - dense.diagnostics["min_normal_map_margin"],
                "intersection_check": "reference only; requested-resolution geometry was checked"}
        record["validation"] = {"status": "accepted", "stage": "complete", "error": None}
    except ValueError as error:
        record["validation"]["error"] = str(error)
        section = None  # Invalid geometry is never drawn as an accepted curve.
    return record, section, extra


def value_label(path, value, *, exact=False):
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, str):
        return value
    units = FIELD_SPECS[path]["units"]
    displayed = value*1000 if units == "m" else value
    number = str(displayed) if exact else f"{displayed:.6g}"
    return number + (" mm" if units == "m" else "°" if units == "deg" else "")


def style(record, color):
    # A modestly heavier dashed baseline remains identifiable with thin curves.
    return dict(color=color, lw=1.15 if record["is_baseline"] else .85,
                ls="--" if record["is_baseline"] else "-", alpha=1 if record["is_baseline"] else .9,
                zorder=4 if record["is_baseline"] else 2)


def fixed_limits(ax, arrays, *, positive=False):
    points = np.vstack(arrays)
    span = np.ptp(points, axis=0)
    pad = np.maximum(span * .08, max(float(span.max()), 1.) * .025)
    ax.set_xlim(points[:, 0].min()-pad[0], points[:, 0].max()+pad[0])
    ax.set_ylim(0 if positive else points[:, 1].min()-pad[1], points[:, 1].max()+pad[1])


def save_figure(fig, stem, directory):
    paths = []
    for extension in ("png", "svg"):
        target = directory / f"{stem}.{extension}"
        fig.savefig(target, dpi=165, metadata={"Date": None} if extension == "svg" else {})
        paths.append(target)
    plt.close(fig)
    return [{"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in paths]


def equalize_limits(ax):
    """Keep identical panel boxes and equal physical scales by expanding limits."""
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    position = ax.get_position()
    width, height = ax.figure.get_size_inches()
    ratio = position.height*height/(position.width*width)
    dx, dy = xlim[1]-xlim[0], ylim[1]-ylim[0]
    dx, dy = max(dx, dy/ratio), max(dy, dx*ratio)
    cx, cy = np.mean(xlim), np.mean(ylim)
    ax.set_xlim(cx-dx/2, cx+dx/2)
    ax.set_ylim(cy-dy/2, cy+dy/2)
    ax.set_aspect("equal", adjustable="box")


def draw_family(sweep, family, baseline, cases, directory, *, detail=False):
    """Every saved image has the same three panel boxes and canvas size."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    fig.subplots_adjust(left=.065, right=.985, bottom=.29, top=.77, wspace=.32)
    subtitle = " | edge detail" if detail else ""
    fig.suptitle(f"{FIELD_SPECS[sweep.path]['label']} | {family.capitalize()}{subtitle}\n"
                 f"{sweep.path} | {sweep.context.replace('_', ' ')} reference", fontsize=13, y=.97)
    accepted = [(r, section, extra) for r, section, extra in cases if section is not None]
    arrays = [panel_arrays(section, baseline) for _, section, _ in accepted]
    for col, (ax, title) in enumerate(zip(axes, ("Camberline", "Half of paired thickness", "Blade profiles"))):
        ax.set_title(title)
        radial = sweep.context != "axial" and col != 1
        ax.set_xlabel(r"Baseline camber arclength $s_0$ [mm]" if col == 1 else "Machine X [mm]" if radial else "Meridional z [mm]")
        ax.set_ylabel("Physical thickness / 2 [mm]" if col == 1 else "Machine Y [mm]" if radial else r"Developed $r\theta$ [mm]")
        if sweep.path == "geometry.type" and col != 1:
            ax.set(xlabel="z (axial) or X (radial) [mm]", ylabel=r"$r\theta$ (axial) or Y (radial) [mm]")
        limits = [a[col] for a in arrays]
        for record, section, extra in sorted(accepted, key=lambda case: case[0]["is_baseline"]):
            points = panel_arrays(section, baseline)[col]
            line = style(record, record["plot_color"])
            ax.plot(*points.T, **line)
            if col == 2 and sweep.supplement == "neighbors":
                neighbor = points + [0., section.diagnostics["physical_pitch_mm"]]
                ax.plot(*neighbor.T, **line)
                limits.append(neighbor)
            if col == 2 and sweep.supplement == "domain":
                for boundary in extra["domain"].values():
                    ax.plot(*boundary.T, color=line["color"], lw=.65, ls=line["ls"], zorder=1)
                    limits.append(boundary)
        fixed_limits(ax, limits, positive=col == 1)
        reference = panel_arrays(baseline, baseline)[col]
        unchanged = all(a[col].shape == reference.shape and np.allclose(a[col], reference, atol=2e-10, rtol=1e-11) for a in arrays)
        if col == 0 and not sweep.path.startswith(("geometry.", "camberline.")):
            unchanged = True
        if sweep.supplement == "sampling":
            unchanged = True
        if unchanged:
            message = "Shape law unchanged" if sweep.supplement == "sampling" else "Unchanged"
            if col == 2 and sweep.supplement == "domain":
                message = "Blade unchanged; domain varies"
            if col == 2 and sweep.supplement == "neighbors":
                message = "Blade unchanged; pitch varies"
            ax.text(.025, .97, message, transform=ax.transAxes, va="top", fontsize=9,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=.85))
        if col == 2 and detail:
            trailing = sweep.supplement == "trailing_zoom"
            endpoint = physical(baseline, "camber")[-1 if trailing else 0]
            end = "trailing" if trailing else "leading"
            fraction = baseline.parameters["thickness"][f"{end}_edge_fraction"]
            window = max(2.2*fraction, .008)*baseline.meridional_chord*1000
            ax.set_xlim(endpoint[0]-window, endpoint[0]+window)
            ax.set_ylim(endpoint[1]-window, endpoint[1]+window)
            ax.set_title("Trailing-edge detail" if trailing else "Leading-edge detail")
        if col != 1:
            equalize_limits(ax)
        for spine in ax.spines.values():
            spine.set_visible(True)
        ax.tick_params(top=True, right=True)
        ax.grid(alpha=.18)
    handles = [Line2D([], [], **style(record, record["plot_color"]),
                      label=value_label(sweep.path, record["value"]) + (" (baseline)" if record["is_baseline"] else "") + (" (rejected)" if section is None else ""))
               for record, section, _ in cases]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .015), ncol=4,
               frameon=False, fontsize=10, handlelength=2.7)
    note = f"Blade count fixed at {baseline.parameters['cascade']['blade_count']}; dashed line: baseline."
    if sweep.path == "geometry.blade_height":
        note += " Span changes; the individual 2D profile remains unchanged."
    if sweep.path == "correction.warning_tolerance":
        note += " Tolerance changes warnings only; error diagnostics are in the manifest."
    if sweep.supplement == "sampling":
        note += " Sampling changes discretization; convergence diagnostics are in the manifest."
    rejected = [value_label(sweep.path, r["value"]) for r, section, _ in cases if section is None]
    if rejected:
        note += " Rejected, not drawn: " + ", ".join(rejected) + "."
    fig.text(.5, .13, note, ha="center", fontsize=9)
    fig.canvas.draw()
    physical_checks, panel_checks = [], []
    for col, ax in enumerate(axes):
        audit = {"panel": col, "xlim_mm": list(ax.get_xlim()), "ylim_mm": list(ax.get_ylim()),
                 "all_spines_visible": all(spine.get_visible() for spine in ax.spines.values()),
                 "box": list(ax.get_position().bounds)}
        assert audit["all_spines_visible"]
        if col != 1:
            points = ax.transData.transform([[0, 0], [1, 0], [0, 1]])
            ratio = (points[1, 0]-points[0, 0])/(points[2, 1]-points[0, 1])
            assert abs(ratio-1) < 1e-9
            audit["physical_scale_ratio"] = ratio
            physical_checks.append(dict(audit, family=family, detail=detail))
        panel_checks.append(audit)
    stem = sweep.path.replace(".", "_")+f"_{family}"+("_detail" if detail else "")
    artifacts = save_figure(fig, stem, directory)
    return artifacts, physical_checks, {"family": family, "detail": detail, "rows": 1, "columns": 3,
                                       "size_inches": [15, 4.8], "panels": panel_checks}


def draw_gallery(sweep, rows, directory):
    artifacts, audits, figures = [], [], []
    for family, (baseline, cases) in zip(FAMILIES, rows):
        positions = np.linspace(*COLORMAP_INTERVAL, len(cases))
        for (record, _, _), position in zip(cases, positions):
            record["colormap_position"] = float(position)
            record["plot_color"] = matplotlib.colors.to_hex(matplotlib.colormaps[COLORMAP](position))
        for detail in ([False, True] if sweep.supplement in {"leading_zoom", "trailing_zoom"} else [False]):
            paths, checks, figure = draw_family(sweep, family, baseline, cases, directory, detail=detail)
            artifacts.extend(paths)
            audits.extend(checks)
            figures.append(figure)
    return artifacts, audits, figures


def section_number(path):
    return 1 if path.startswith(("geometry.", "cascade.")) else 2 if path.startswith("camberline.") else 4 if path.startswith(("flow_domain.", "numerics.")) else 3


def bounds(path):
    spec = FIELD_SPECS[path]
    if spec["type"] == "enum":
        return ", ".join(spec["options"])
    if spec["type"] == "bool":
        return "off, on"
    parts = ["integer"] if spec["type"] == "int" else []
    for key, symbol in (("min", ">"), ("max", "<")):
        if key in spec:
            parts.append(symbol + ("" if spec.get(key+"_exclusive") else "=") + " " + value_label(path, spec[key]))
    return "; ".join(parts) + "; coupled geometry checks also apply"


def write_gallery(report, target):
    lines = ["<!-- Generated by development/blade_parametrization/parameter_sensitivity/study.py. -->", "",
             "## Exact reference inputs", "", "All current GUI baseline inputs, including inactive inputs, are listed below without display rounding. Lengths are displayed in mm; the complete manifest stores metres and also records the fixed hidden kernel inputs.", "",
             "| Independent input | Reaction | Impulse |", "| --- | ---: | ---: |"]
    for path in FIELD_SPECS:
        # Retired inputs are recorded in the manifest, not offered as GUI controls.
        if path in report["coverage"]["hidden_kernel_inputs"]:
            continue
        values = [lookup(report["baselines"][f"{family}/axial"]["inputs"], path) for family in FAMILIES]
        lines.append(f"| `{path}` | {value_label(path, values[0], exact=True)} | {value_label(path, values[1], exact=True)} |")
    lines += ["", "The radial references change only `geometry.type` to `radial`: inlet/outlet radii are **20/35 mm**; the axial chord and mean radius remain stored but inactive. The diagnostic-only tolerance reference additionally enables physical-thickness correction. These context changes occur **before** a sweep and are recorded separately in the manifest.", "",
              "## Parameter index by GUI section", "", "**Profile** identifies a shape input in the main gallery. Other controls are illustrated in the appendix.", ""]
    groups = {1: "01 Meridional channel", 2: "02 Camberline and turning", 3: "03 Thickness distribution", 4: "04 Flow domain, including advanced resolution"}
    for number, title in groups.items():
        lines += [f"### {title}", "", "| Control | Input key | Illustration |", "| --- | --- | --- |"]
        for sweep in SWEEPS:
            if section_number(sweep.path) == number:
                slug = sweep.path.replace(".", "-").replace("_", "-")
                lines.append(f"| [{FIELD_SPECS[sweep.path]['label']}](#{slug}) | `{sweep.path}` | {'Profile' if sweep.profile else 'Appendix'} |")
        if number == 1:
            lines.append("| [Number of blades](#fixed-blade-count) | `cascade.blade_count` | Fixed; no sweep |")
        lines.append("")
    lines += ["(fixed-blade-count)=", "### Fixed blade count", "",
              "Blade count is **24 for reaction and 12 for impulse in every case**, including radial examples. It is not varied. Mean radius can still change physical pitch as a dependent geometric result.", ""]
    for profile, title in ((True, "2D profile controls"), (False, "Appendix: spacing, span, domain and numerical controls")):
        lines += [f"## {title}", ""]
        ordered = ([s for s in SWEEPS if s.profile and s.context == "axial" and s.path != "geometry.type"]
                   + [s for s in SWEEPS if s.profile and (s.context != "axial" or s.path == "geometry.type")]
                   if profile else SWEEPS)
        for sweep in ordered:
            if sweep.profile != profile:
                continue
            entry = next(item for item in report["sweeps"] if item["parameter"] == sweep.path)
            slug = sweep.path.replace(".", "-").replace("_", "-")
            stem = sweep.path.replace(".", "_")
            lines += [f"({slug})=", f"### {FIELD_SPECS[sweep.path]['label']}", "", f"`{sweep.path}` · GUI section {section_number(sweep.path):02} · {sweep.context.replace('_', ' ')} reference", "",
                      sweep.meaning, "", f"**Illustrative range:** {sweep.rationale}", "",
                      f"**Admissible inputs:** {bounds(sweep.path)}. These are mathematical bounds, not aerodynamic design limits or slider endpoints.", "",
                      f"**Visible effect:** {sweep.effect}", "", f"**Coupling and limitations:** {sweep.coupling}", "",
                      "| Reference | Sweep values (baseline in bold) | Rejected values |", "| --- | --- | --- |"]
            for family in FAMILIES:
                cases = entry["families"][family]
                labels = [value_label(sweep.path, r["value"]) for r in cases]
                labels = [f"**{label}**" if r["is_baseline"] else label for label, r in zip(labels, cases)]
                rejected = [value_label(sweep.path, r["value"]) for r in cases if r["validation"]["status"] == "rejected"]
                lines.append(f"| {family.capitalize()} | {', '.join(labels)} | {', '.join(rejected) or 'None'} |")
            for family in FAMILIES:
                lines += ["", f"**{family.capitalize()}** — camberline, half paired thickness, blade profiles.", "",
                          f"![{FIELD_SPECS[sweep.path]['label']}, {family}: one row with camberline, half paired thickness, and physical blade profiles.](../_static/blade_sensitivity/{stem}_{family}.png)", "",
                          f"[Open {family} vector gallery](../_static/blade_sensitivity/{stem}_{family}.svg)", ""]
                if sweep.supplement in {"leading_zoom", "trailing_zoom"}:
                    lines += [f"![{family.capitalize()} edge detail, using the same three-column layout.](../_static/blade_sensitivity/{stem}_{family}_detail.png)", "",
                              f"[Open {family} vector edge detail](../_static/blade_sensitivity/{stem}_{family}_detail.svg)", ""]
            for family in FAMILIES:
                for record in entry["families"][family]:
                    if record["validation"]["status"] == "rejected":
                        lines += [f"- **Rejected {family}, {value_label(sweep.path, record['value'])}:** {record['validation']['error']}", ""]
            warnings = sorted({warning for records in entry["families"].values() for r in records for warning in r.get("warnings", [])})
            if warnings:
                lines += ["**Accepted-case cautions:** " + " ".join(warnings), ""]
    accepted = sum(r["validation"]["status"] == "accepted" for item in report["sweeps"] for records in item["families"].values() for r in records)
    total = sum(len(records) for item in report["sweeps"] for records in item["families"].values())
    lines += ["## Generated validation record", "", f"The recorded run contains **{total} requested cases: {accepted} accepted, {total-accepted} rejected**, sweeping {len(SWEEPS)} GUI fields while holding blade count fixed. Each continuous parameter has four values including the baseline; categorical and Boolean parameters use their available choices. Every image has one row and three columns, thin lines, a full box on each panel, and plasma colors sampled from 0.25 to 0.75. All accepted cases passed independent-input isolation, baseline recovery when applicable, mapped-thickness consistency, and sampled self/neighbor intersection checks. Rejected cases, if any, are reported beside their figures and never drawn as accepted geometry.", "",
              "The manifest contains every resolved input, warning, diagnostic, source hash, coordinate hash, figure hash and physical-aspect check. Intersection checks apply at the requested resolution; they are not proofs between samples. A nonpositive normal-map margin is reported as a caution by the kernel even if the sampled contour passes intersection checks.", ""]
    target.write_text("\n".join(lines), encoding="utf-8")


def remove_superseded_artifacts(previous, current, directory):
    """Remove only obsolete PNG/SVG files named in the previous study manifest."""
    retained = {a["path"] for sweep in current["sweeps"] for a in sweep["artifacts"]}
    for sweep in previous.get("sweeps", []):
        for artifact in sweep["artifacts"]:
            if artifact["path"] in retained:
                continue
            target = (directory / artifact["path"]).resolve()
            if target.parent != directory or target.suffix not in {".png", ".svg"}:
                raise ValueError(f"Unexpected generated artifact path: {target}")
            if target.exists():
                if hashlib.sha256(target.read_bytes()).hexdigest() != artifact["sha256"]:
                    raise ValueError(f"Obsolete artifact was edited outside this study; preserved: {target}")
                target.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="+", choices=[s.path for s in SWEEPS], help="Preview selected parameters; does not replace the full manifest or gallery include.")
    parser.add_argument("--output-dir", type=Path, default=ASSETS)
    args = parser.parse_args()
    directory = args.output_dir.resolve()
    if not args.only and directory != ASSETS.resolve():
        parser.error("A full rebuild uses the documentation asset directory; use --only for a custom preview directory.")
    directory.mkdir(parents=True, exist_ok=True)
    previous_manifest = directory / "manifest.json"
    previous = json.loads(previous_manifest.read_text(encoding="utf-8")) if previous_manifest.exists() and not args.only else {}
    hidden = hidden_gui_paths()
    assert {s.path for s in SWEEPS} | set(FIXED_GUI_INPUTS) == set(FIELD_SPECS)-hidden, "GUI coverage changed; update the sweep catalogue"
    assert not ({s.path for s in SWEEPS} & set(FIXED_GUI_INPUTS))
    assert len({s.path for s in SWEEPS}) == len(SWEEPS)
    base_sections, base_records = baselines()
    report = {"schema_version": 2, "method": "one independent input at a time; production kernel only",
              "plot_options": {"style_function": "turbodash.set_plot_options", "fontsize": 11,
                               "variation_linewidth": .85, "baseline_linewidth": 1.15,
                               "colormap": COLORMAP, "colormap_interval": list(COLORMAP_INTERVAL),
                               "continuous_values_per_sweep": 4, "layout": [1, 3], "box_on": True},
              "units": {"stored_lengths": "m", "plotted_lengths": "mm", "input_angles": "deg", "chart_angles": "rad"},
              "thickness_representation": "half of production measured_physical (total paired intrinsic thickness), versus physical baseline camber polyline arclength at matching xi; not an actual surface",
              "environment": {"python": platform.python_version(), "platform": platform.system(),
                              "packages": {p: importlib.metadata.version(p) for p in ("numpy", "scipy", "matplotlib", "PyYAML", "turbo_moc")}},
              "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
              "source_sha256": {}, "coverage": {"gui_fields": sorted(set(FIELD_SPECS)-hidden), "hidden_kernel_inputs": sorted(hidden),
                "fixed_gui_inputs": FIXED_GUI_INPUTS,
                "not_geometry_inputs": "preview visibility/colors/copy count and passage-crosscut display toggle are display preferences; YAML/reset actions are state operations"},
              "baselines": base_records, "sweeps": []}
    sources = [HERE / "study.py", REPO / "turbodash/geometry/meanline.py", REPO / "turbodash/apps/shared/components/blade_modeller.py",
               REPO / "turbodash/visualization/matplotlib_style.py"]
    sources += list((REPO / "turbodash/geometry/blade_parametrization").glob("*.py"))
    sources += list((REPO / "turbodash/geometry/blade_parametrization/cases").glob("*.yaml"))
    # Include the actual downstream domain implementation, including installed
    # versions whose source might differ from a companion checkout.
    for name, module in adapter._modules().items():
        source = Path(module.__file__)
        report["source_sha256"][f"turbo_moc/{name}"] = hashlib.sha256(source.read_bytes()).hexdigest()
    for source in sources:
        report["source_sha256"][source.relative_to(REPO).as_posix()] = hashlib.sha256(source.read_bytes()).hexdigest()
    # Package imports can install their own global plotting style. Apply the
    # gallery style after loading the optional production passage backend.
    set_plot_options(fontsize=11, linewidth=.85)
    plt.rcParams.update({"axes.spines.top": True, "axes.spines.right": True,
                         "axes.spines.left": True, "axes.spines.bottom": True,
                         "svg.fonttype": "none", "svg.hashsalt": "turbodash-blade-sensitivity",
                         "path.simplify": False, "savefig.facecolor": "white"})
    for sweep in SWEEPS:
        if args.only and sweep.path not in args.only:
            continue
        rows, families = [], {}
        for family in FAMILIES:
            baseline = base_sections[family, sweep.context]
            cases = [evaluate(sweep, value, baseline) for value in sweep_values(sweep, baseline)]
            families[family] = [record for record, _, _ in cases]
            assert any(r["is_baseline"] and s is not None for r, s, _ in cases)
            rows.append((baseline, cases))
        artifacts, audits, figures = draw_gallery(sweep, rows, directory)
        report["sweeps"].append({"parameter": sweep.path, "gui_section": section_number(sweep.path),
            "context": sweep.context, "changes_profile": sweep.profile, "meaning": sweep.meaning,
            "effect": sweep.effect, "coupling": sweep.coupling, "range_rationale": sweep.rationale,
            "field_spec": FIELD_SPECS[sweep.path], "families": families,
            "artifacts": artifacts, "physical_panel_checks": audits, "figures": figures})
        rejected = sum(r["validation"]["status"] == "rejected" for records in families.values() for r in records)
        print(f"{sweep.path}: {sum(map(len, families.values()))} cases, {rejected} rejected", flush=True)
    manifest_name = "preview_manifest.json" if args.only else "manifest.json"
    if not args.only:
        remove_superseded_artifacts(previous, report, directory)
    (directory / manifest_name).write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    if not args.only:
        write_gallery(report, GALLERY)
    print(f"Wrote {directory / manifest_name}", flush=True)


if __name__ == "__main__":
    main()
