"""Isolated conversion from V4 SI geometry to the existing TurboMoC pipeline.

V4 uses (m', mathematical counterclockwise theta). TurboMoC stores dimensional
reference-plane coordinates q=(-Rref*theta, -Rref*m') in millimetres. All unit
and handedness changes happen here. Importing this module does not import
TurboMoC, its thermodynamics dependencies, VTK, or Gmsh.
"""
from __future__ import annotations

from importlib import import_module, util
from pathlib import Path
import hashlib
import json
import os
import sys

import numpy as np


class TurboMOCUnavailable(RuntimeError):
    """The optional downstream geometry/visualization installation is missing."""


def turbo_moc_source():
    """Find its package directory without importing the solver package.

    TURBOMOC_SOURCE may name the checkout, its src directory, or the package
    directory. An installed package wins over automatic sibling discovery;
    the explicit environment variable wins over both. It must be set before
    importing turbo_moc. No source files are copied or changed.
    """
    override = os.environ.get("TURBOMOC_SOURCE")
    if override:
        root = Path(override).expanduser().resolve()
        candidates = (root, root / "turbo_moc", root / "src" / "turbo_moc")
        package = next((item for item in candidates if item.name == "turbo_moc"
                        and (item / "__init__.py").is_file()), None)
        if package is None:
            raise TurboMOCUnavailable("TURBOMOC_SOURCE must point to a TurboMoC checkout, src directory, or turbo_moc package.")
        loaded = sys.modules.get("turbo_moc")
        if loaded and Path(loaded.__file__).resolve().parent != package:
            raise TurboMOCUnavailable("TURBOMOC_SOURCE differs from the already imported TurboMoC. Restart the application after setting it.")
        return package
    specification = util.find_spec("turbo_moc")
    if specification and specification.submodule_search_locations:
        return Path(next(iter(specification.submodule_search_locations))).resolve()
    for ancestor in Path(__file__).resolve().parents:
        package = ancestor / "method_of_characteristics" / "src" / "turbo_moc"
        if (package / "__init__.py").is_file():
            return package
    raise TurboMOCUnavailable(
        "TurboMoC is required for mapping, 3D preview and optional meshing. "
        "Install the companion checkout with `python -m pip install -e /path/to/method_of_characteristics`, "
        "or set TURBOMOC_SOURCE to that checkout and use its prepared Python environment."
    )


def renderer_asset_directory():
    """Return the original packaged renderer assets for the app's safe routes."""
    directory = turbo_moc_source() / "app" / "assets"
    if not all((directory / name).is_file() for name in ("studio.js", "mesh_edges.js")):
        raise TurboMOCUnavailable("This TurboMoC installation lacks the required VTK browser assets; use the supported companion checkout.")
    return directory


def _modules():
    package = turbo_moc_source()
    source = str(package.parent)
    if source not in sys.path:
        sys.path.insert(0, source)
    try:
        return {
            "channel": import_module("turbo_moc.geometry.meridional_channel"),
            "row": import_module("turbo_moc.geometry.blade_row"),
            "passage": import_module("turbo_moc.geometry.passage_2d"),
            "rotor": import_module("turbo_moc.geometry.rotor_blade"),
        }
    except ImportError as error:
        raise TurboMOCUnavailable(
            f"TurboMoC was found at {package}, but a dependency could not be imported ({error}). "
            "Use the companion method_of_characteristics Python environment or install its dependencies."
        ) from error


def _parameters(section):
    parameters = section.parameters
    return parameters.values if hasattr(parameters, "values") and not isinstance(parameters, dict) else parameters


def _mapping(section, modules):
    p = _parameters(section)
    geometry, count = p["geometry"], p["cascade"]["blade_count"]
    if geometry["type"] == "axial":
        radius, height = geometry["mean_radius"], geometry["blade_height"]
        dimensions = {"type": "axial", "hub_radius": (radius - height / 2) * 1000,
                      "blade_height": height * 1000,
                      "hub_chord": section.meridional_chord * 1000,
                      "tip_chord": section.meridional_chord * 1000,
                      "num_blades": count}
    elif geometry["type"] == "radial":
        dimensions = {"type": "radial", "inlet_radius": geometry["inlet_radius"] * 1000,
                      "outlet_radius": geometry["outlet_radius"] * 1000,
                      "blade_height": geometry["blade_height"] * 1000, "num_blades": count}
    else:
        raise ValueError("V4 downstream mapping supports only axial and radial geometry.")
    channel = modules["channel"]
    mapping = channel.MeridionalChannel(channel.geometry_from_machine(dimensions))
    source_eta = section.diagnostics.get("span_fraction", .5)
    if not np.isclose(mapping.prime_chord(source_eta), section.prime_chord, rtol=1e-10, atol=1e-12):
        raise ValueError("V4 and TurboMoC disagree on the camberline endpoint conformal chord.")
    return mapping


def _q(points, radius_mm):
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("Profile coordinates must be finite (m', theta) pairs.")
    return -radius_mm * points[:, ::-1]


def _deduplicate(points):
    points = np.asarray(points, dtype=float)
    tolerance = max(1e-12, np.ptp(points, axis=0).max() * 1e-12)
    return points[np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > tolerance]]


def _curve_distance(points, polyline):
    """Exact point-to-polyline distances using an exhaustive spatial bound.

    A segment within distance d must have its midpoint within d plus its
    half-length. Querying with the largest half-length therefore excludes no
    potentially closer segment, even with strongly nonuniform edge samples.
    """
    from scipy.spatial import cKDTree

    starts, edges = polyline[:-1], np.diff(polyline, axis=0)
    lengths2 = np.sum(edges * edges, axis=1)
    valid = lengths2 > 1e-24
    starts, edges, lengths2 = starts[valid], edges[valid], lengths2[valid]
    if not len(starts):
        raise ValueError("The blade boundary contains no nonzero segments.")
    points = np.asarray(points)
    tree = cKDTree(starts + edges / 2)
    _, nearest = tree.query(points)
    delta = points - starts[nearest]
    fraction = np.clip(np.einsum("ij,ij->i", delta, edges[nearest]) / lengths2[nearest], 0, 1)
    upper = np.linalg.norm(delta - fraction[:, None] * edges[nearest], axis=1)
    radius = upper + .5 * np.sqrt(lengths2.max())
    # Midpoint construction and subtraction each round in coordinate units.
    # One ULP of the much smaller radius is insufficient at a segment endpoint.
    coordinate_scale = (np.linalg.norm(points, axis=1)
                        + np.linalg.norm(starts, axis=1).max()
                        + np.sqrt(lengths2.max()))
    padding = 8 * np.finfo(float).eps * coordinate_scale
    candidates = tree.query_ball_point(points, np.nextafter(radius + padding, np.inf))
    owners = np.repeat(np.arange(len(points)), [len(ids) for ids in candidates])
    indices = np.concatenate(candidates).astype(np.intp, copy=False)
    delta = points[owners] - starts[indices]
    fraction = np.clip(np.einsum("ij,ij->i", delta, edges[indices]) / lengths2[indices], 0, 1)
    distance2 = np.sum((delta - fraction[:, None] * edges[indices]) ** 2, axis=1)
    # The nearest-midpoint segment is already a valid finite upper bound, even
    # if the spatial query happens to lose a boundary candidate to roundoff.
    result = upper**2
    np.minimum.at(result, owners, distance2)
    return np.sqrt(result)


def _placement_contour(points, tolerance):
    """Error-bounded polygon simplification solely for the passage optimizer.

    The full sampled CAD contour and all its final validation checks remain
    authoritative. Split the closed loop before Douglas-Peucker simplification
    so neither end of the blade disappears into a degenerate closing chord.
    """
    def simplify(curve):
        retained = {0, len(curve) - 1}
        stack = [(0, len(curve) - 1)]
        while stack:
            first, last = stack.pop()
            if last - first <= 1:
                continue
            segment = curve[last] - curve[first]
            delta = curve[first + 1:last] - curve[first]
            fraction = np.clip(delta @ segment / (segment @ segment), 0, 1)
            distances = np.sum((delta - fraction[:, None] * segment) ** 2, axis=1)
            index = int(np.argmax(distances))
            if distances[index] > tolerance**2:
                split = first + index + 1
                retained.add(split)
                stack.extend(((first, split), (split, last)))
        return curve[sorted(retained)]

    split = int(np.argmax(np.linalg.norm(points - points[0], axis=1)))
    return np.vstack((simplify(points[:split + 1]), simplify(points[split:])[1:]))


def _center_passage(modules, blade, settings, tolerance):
    """Reuse the same placement objective without scoring redundant CAD points.

    Passage guides are built from the full contour. Only the Euclidean-gap
    placement search uses a simplified contour; full-resolution containment
    is checked immediately. A failed or near-boundary proposal falls back to
    the original dense search. No blade geometry is simplified or refitted.
    """
    planar, row = modules["passage"], modules["row"]
    baseline = modules["rotor"]._flow_domain(blade, settings | {"domain_shift_mode": "manual"})
    contour = planar.blade_contour(blade)
    reduced = _placement_contour(contour, tolerance)
    deviation = max(_curve_distance(contour, reduced).max(), _curve_distance(reduced, contour).max())
    proxy = blade | {"blade_curve": {"x": reduced[:, 0].tolist(), "y": reduced[:, 1].tolist()},
                     "trailing_edge": {"x": [], "y": []}}
    fallback = bool(deviation > tolerance * (1 + 1e-8))
    if not fallback:
        try:
            curves = planar.auto_center_flow_domain(proxy, baseline)
            fallback = row.require_containment(blade, curves) <= 10 * tolerance
        except ValueError:
            fallback = True
    if fallback:
        curves = planar.auto_center_flow_domain(blade, baseline)
        row.require_containment(blade, curves)
    return curves, {"passage_placement_samples": len(contour) if fallback else len(reduced),
                    "passage_full_contour_samples": len(contour),
                    "passage_placement_deviation_mm": float(deviation),
                    "passage_placement_tolerance_mm": float(tolerance),
                    "passage_placement_dense_fallback": fallback}


def _boundary_splines(section, mapping):
    """Transfer paired sides with common rounded-tip tangents and explicit caps.

    Interpolation preserves every supplied side sample. Deviation between
    samples is measured against the original piecewise-linear profile, which
    also contains the input sampling error; it is not an analytic CAD bound.
    """
    from scipy.interpolate import BSpline, make_interp_spline

    # A core section can be sampled at any span. Design's canonical template
    # lives at midspan; the supported section family differs by this scale.
    radius = mapping.reference_radius * mapping.prime_chord(.5) / section.prime_chord
    plus = _deduplicate(_q(section.plus, radius))
    minus = _deduplicate(_q(section.minus, radius))
    if min(len(plus), len(minus)) < 4:
        raise ValueError("At least four distinct samples on each side are needed for a blade CAD spline.")
    scale = mapping.reference_chord
    tolerance = max(1e-10, scale * 1e-10)
    closed = [np.linalg.norm(plus[index] - minus[index]) <= tolerance for index in (0, -1)]
    for index, shared in zip((0, -1), closed):
        if shared:
            minus[index] = plus[index]
    beta = np.asarray(section.beta)
    thickness = _parameters(section)["thickness"]
    rounded = [closed[index] and thickness[f"{end}_edge_fraction"] > 0
               and thickness[f"{end}_edge_thickness_ratio"] > 0
               for index, end in enumerate(("leading", "trailing"))]
    # n=(-sin(beta),cos(beta)) in V4 chart coordinates; q reflects and swaps.
    normals = [np.array([-np.cos(angle), np.sin(angle)]) for angle in (beta[0], beta[-1])]

    def side(points, first, last):
        stations = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
        length = stations[-1]
        # Closed rounded tips have a normal tangent. Open flat caps retain the
        # one-sided profile tangent inferred from the nearest sampled segment.
        a = first if first is not None else (points[1] - points[0]) / (stations[1] - stations[0])
        b = last if last is not None else (points[-1] - points[-2]) / (stations[-1] - stations[-2])
        spline = make_interp_spline(stations / length, points, k=3,
            bc_type=([(1, a * length)], [(1, b * length)]))
        # Includes input samples and interval midpoints, resolving tightly
        # clustered nose samples without requiring uniform huge point arrays.
        locations = stations / length
        samples = np.sort(np.r_[locations, (locations[:-1] + locations[1:]) / 2])
        return spline, spline(samples)

    first, sample_plus = side(plus, normals[0] if rounded[0] else None,
                             -normals[1] if rounded[1] else None)
    second, sample_minus = side(minus[::-1], -normals[1] if rounded[1] else None,
                               normals[0] if rounded[0] else None)
    splines, pieces = [first], [sample_plus]
    if not closed[1]:
        cap = BSpline([0, 0, 1, 1], [plus[-1], minus[-1]], 1)
        splines.append(cap)
        pieces.append(cap(np.linspace(0, 1, 9)))
    splines.append(second)
    pieces.append(sample_minus)
    if not closed[0]:
        cap = BSpline([0, 0, 1, 1], [minus[0], plus[0]], 1)
        splines.append(cap)
        pieces.append(cap(np.linspace(0, 1, 9)))
    sampled = _deduplicate(np.vstack(pieces))
    if np.linalg.norm(sampled[-1] - sampled[0]) > tolerance:
        sampled = np.vstack((sampled, sampled[0]))
    else:
        sampled[-1] = sampled[0]
    original = _deduplicate(_q(section.boundary, radius))
    if np.linalg.norm(original[-1] - original[0]) > tolerance:
        original = np.vstack((original, original[0]))
    # Validate the complete boundary contract, including possible end caps.
    paired = _deduplicate(np.vstack((plus, minus[::-1], plus[:1])))
    coverage = max(_curve_distance(original, paired).max(), _curve_distance(paired, original).max())
    if coverage > max(1e-8, scale * 1e-7):
        raise ValueError("Section.boundary is inconsistent with its paired sides and straight end caps.")
    fit_error = max(_curve_distance(sampled, original).max(), _curve_distance(original, sampled).max())
    if fit_error > max(1e-7, scale * 5e-4):
        raise ValueError(f"CAD interpolation deviates from the sampled profile by {fit_error:.4g} mm; increase geometry sampling or revise the edge parameters.")
    from .geometry import curves_intersect
    if curves_intersect(sampled):
        raise ValueError("The interpolated CAD contour crosses or touches itself; revise the edge parameters or increase sampling.")
    diagnostics = {"cad_polyline_deviation_mm": float(fit_error),
                   "cad_deviation_chord_fraction": float(fit_error / scale),
                   "cad_error_measure": "bidirectional sampled spline-to-input-polyline distance",
                   "cad_boundary_segments": len(splines),
                   "flat_leading_cap": not closed[0], "flat_trailing_cap": not closed[1],
                   "camberline_meridional_chord_mm": float(section.meridional_chord * 1000),
                   "complete_profile_conformal_extent_mm": float(np.ptp(sampled[:, 1]))}
    return splines, sampled, diagnostics


def build_design(section):
    """Return a reusable TurboMoC Design without running a flow or mesh solver.

    The requested chord remains the camberline LE-to-TE separation. Rounded
    edge protrusions are retained; the profile is never translated or scaled
    to force its complete envelope onto the nominal chord planes.
    """
    modules = _modules()
    mapping = _mapping(section, modules)
    splines, sampled, diagnostics = _boundary_splines(section, mapping)
    p = _parameters(section)
    angles = p["camberline"]
    blade = {"provider": "parametrization_blades", "pitch": mapping.pitch,
             "boundary_splines": splines,
             "blade_curve": {"x": sampled[:, 0].tolist(), "y": sampled[:, 1].tolist()},
             "trailing_edge": {"x": [], "y": []},
             "metal_angle_in": -float(angles["inlet_metal_angle_deg"]),
             "metal_angle_out": -float(angles["outlet_metal_angle_deg"]),
             "v4_diagnostics": diagnostics}
    flow_domain = {"inlet_chords": p["flow_domain"]["inlet_chords"],
                   "outlet_chords": p["flow_domain"]["outlet_chords"],
                   "domain_shift_mode": "auto", "domain_shift": 0., "num_points": 100}
    # TurboMoC's arbitrary-contour passage builder is currently private. Keep
    # this single compatibility dependency here, as in its TurboDash adapter.
    curves, placement = _center_passage(modules, blade, flow_domain, mapping.reference_chord * 1e-4)
    diagnostics.update(placement)
    curves.update(blade_curve=sampled, trailing_edge=np.empty((0, 2)))
    design = modules["row"].Design(mapping, blade, curves, flow_domain=flow_domain)
    margins = []
    for eta in (0., .25, .5, .75, 1.):
        margins.append(modules["row"].require_containment(*design.section(eta)))
    diagnostics["minimum_sampled_passage_margin_mm"] = float(min(margins))
    return design


def map_section(section, design=None, span_fraction=.5):
    """Map authoritative side samples using TurboMoC; return XYZ in metres.

    For the supported constant-chord axial row this preserves the same
    developed physical section across span. For a radial row it extrudes the
    same mapped section between parallel endwalls. Preview-only cap overhang
    is never present in these scientific coordinates.
    """
    if not np.isfinite(span_fraction) or not 0 <= span_fraction <= 1:
        raise ValueError("span_fraction must be between zero and one.")
    if design is None:
        mapping = _mapping(section, _modules())
    else:
        mapping = design.meridional_channel
    scale = mapping.prime_chord(float(span_fraction)) / section.prime_chord
    return {name: mapping.map_points(_q(getattr(section, name), mapping.reference_radius) * scale,
                                     float(span_fraction)) / 1000
            for name in ("camber", "plus", "minus", "boundary")}


def make_preview(section, design=None):
    """Return the existing browser renderer's JSON-safe single-row payload."""
    design = build_design(section) if design is None else design
    try:
        preview = import_module("turbo_moc.visualization.geometry_preview").preview_geometry(
            design, span_sections=7, curve_points=220, subdivisions=1)
    except ImportError as error:
        raise TurboMOCUnavailable("The TurboMoC 3D preview requires VTK/PyVista; use the companion environment.") from error
    p = _parameters(section)
    row = {"id": "v4_row", "label": "V4 blade row", "role": "stator",
           "num_blades": int(p["cascade"]["blade_count"]), "is_linear": False,
           "geometry_type": p["geometry"]["type"], "surfaces": preview["surfaces"],
           "placement": {"streamwise_offset_mm": 0., "angular_phase_deg": 0.}}
    encoded = json.dumps(row, sort_keys=True, allow_nan=False, separators=(",", ":"))
    metadata = {"coordinate_units": "mm", "axis_of_rotation": "Z",
                "theta_convention": "V4 mathematical counterclockwise",
                "turbo_moc_version": import_module("turbo_moc").__version__,
                "source_package": str(turbo_moc_source()),
                "cad": design.blade["v4_diagnostics"],
                "preview_is_display_geometry": True,
                "preview_span_padding_fraction": .005}
    return {"key": hashlib.sha256(encoded.encode()).hexdigest(), "rows": [row],
            "copy_limit": row["num_blades"], "metadata": metadata,
            "note": "XYZ in mm. Preview has a display-only 0.5% span overhang at each endwall; measurements use the unpadded geometry."}
