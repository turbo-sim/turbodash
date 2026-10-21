"""Conformal blade sections with a declared physical thickness measurement.

All lengths are metres, angles in arrays are radians, and the chart is
(m-prime, mathematical counterclockwise azimuth). This module has no UI,
TurboDash, TurboMOC, meshing, or thermodynamic dependency.
"""

from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from .parameters import validate_parameters
from .thickness import denton_body


@dataclass
class Section:
    """One sampled stream-surface section, retaining paired side stations."""

    parameters: dict
    xi: np.ndarray
    prime_chord: float
    meridional_chord: float
    radius: np.ndarray
    meridional_distance: np.ndarray
    beta: np.ndarray
    camber: np.ndarray
    plus: np.ndarray
    minus: np.ndarray
    boundary: np.ndarray
    thickness: dict
    diagnostics: dict
    warnings: list
    pressure_side: str


def _integral(values, coordinate):
    return np.r_[0., np.cumsum(.5 * (values[1:] + values[:-1]) * np.diff(coordinate))]


def turning_angle(xi, parameters):
    """Endpoint-preserving angle/slope interpolation on conformal fraction."""
    xi = np.asarray(xi, dtype=float)
    location = parameters["turning_location"]
    concentration = parameters["turning_concentration"]
    fraction = np.zeros_like(xi)
    interior = (xi > 0) & (xi < 1)
    x = xi[interior]
    odds = concentration * (np.log(x) - np.log1p(-x)
                            + np.log1p(-location) - np.log(location))
    fraction[interior] = .5 * (1 + np.tanh(.5 * odds))
    fraction[xi >= 1] = 1.
    incoming, outgoing = np.deg2rad([
        parameters["inlet_metal_angle_deg"], parameters["outlet_metal_angle_deg"]])
    if parameters["interpolation"] == "angle":
        return incoming + (outgoing - incoming) * fraction
    return np.arctan(np.tan(incoming) + (np.tan(outgoing) - np.tan(incoming)) * fraction)


def denton_thickness(xi, parameters):
    """Full thickness / conformal meridional chord, before one-sided relief.

    The input maximum is a nominal body value at a nominal station. The
    unrounded endpoint values and edge blend fractions are independent.
    """
    xi = np.asarray(xi, dtype=float)
    body = denton_body(xi, parameters)
    factors = []
    for fraction, distance, exponent in (
        (parameters["leading_edge_fraction"], xi, parameters["leading_edge_rounding_exponent"]),
        (parameters["trailing_edge_fraction"], 1 - xi, parameters["trailing_edge_rounding_exponent"]),
    ):
        if fraction == 0:
            factors.append(np.ones_like(xi))
        else:
            q = np.minimum(distance / fraction, 1.)
            # Evaluate 1-(1-q)**exponent accurately even close to a tip.
            with np.errstate(divide="ignore"):
                square = -np.expm1(exponent * np.log1p(-q))
            factors.append(np.sqrt(np.maximum(square, 0.)))
    return body, body * factors[0] * factors[1]


def relief_window(xi, leading_edge_fraction):
    """C2 compact window; zero at the tip and beyond twice the nose extent."""
    if leading_edge_fraction == 0:
        return np.zeros_like(xi)
    q = np.clip(np.asarray(xi) / min(2 * leading_edge_fraction, 1.), 0., 1.)
    return np.minimum((256 / 27) * q * (1 - q)**3, 1.)


def physical_pair_distance(radius, beta, plus_offset, minus_offset, *, radial_direction=0):
    """Exact paired intrinsic distance for the supported physical surfaces.

    Axial: Euclidean distance in the unwrapped cylinder. Radial: distance
    in the physical radial plane. The two samples share a camberline station;
    this is not the minimum distance between the complete side curves.
    Offsets are dimensionless distances in the conformal plane.
    """
    radius, beta, a, b = np.broadcast_arrays(radius, beta, plus_offset, minus_offset)
    total = a + b
    if radial_direction == 0:
        return radius * total
    k = -radial_direction * np.sin(beta)
    # Stable exact expression, including asymmetric radii of the paired points.
    return (2 * radius * np.exp(.5 * k * (a - b))
            * np.hypot(np.sinh(.5 * k * total), np.sin(.5 * np.cos(beta) * total)))


def _closed_boundary(plus, minus):
    points = np.vstack((plus, minus[::-1], plus[:1]))
    keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-13]
    points = points[keep]
    if np.linalg.norm(points[-1] - points[0]) > 1e-13:
        points = np.vstack((points, points[0]))
    else:
        points[-1] = points[0]
    return points


def curves_intersect(first, second=None):
    """Detect crossings/touching of sampled polygon segments, in bounded chunks.

    Adjacent segments of the same closed contour are excluded. This checks
    the supplied resolution, not a proof over the continuous analytic curves.
    """
    same = second is None
    second = first if same else second
    first, second = np.asarray(first), np.asarray(second)
    scale = max(float(np.ptp(first, axis=0).max()), float(np.ptp(second, axis=0).max()), 1e-12)
    a, b = first[:-1] / scale, first[1:] / scale
    c, d = second[:-1] / scale, second[1:] / scale
    tolerance = 1e-11
    right_ids = np.arange(len(c))[None, :]
    for begin in range(0, len(a), 64):
        aa, bb = a[begin:begin+64, None], b[begin:begin+64, None]
        cc, dd = c[None], d[None]
        candidate = np.all((np.maximum(aa, bb) + tolerance >= np.minimum(cc, dd))
                           & (np.maximum(cc, dd) + tolerance >= np.minimum(aa, bb)), axis=-1)
        if same:
            left_ids = np.arange(begin, begin + len(aa))[:, None]
            separation = np.abs(left_ids - right_ids)
            candidate &= (right_ids > left_ids) & (separation > 1) & (separation < len(a) - 1)
        rows, cols = np.where(candidate)
        if not len(rows):
            continue
        av, bv, cv, dv = aa[rows, 0], bb[rows, 0], c[cols], d[cols]
        cross = lambda x, y: x[:, 0] * y[:, 1] - x[:, 1] * y[:, 0]
        o1, o2 = cross(bv-av, cv-av), cross(bv-av, dv-av)
        o3, o4 = cross(dv-cv, av-cv), cross(dv-cv, bv-cv)
        opposite_a = ((o1 <= tolerance) & (o2 >= -tolerance)) | ((o2 <= tolerance) & (o1 >= -tolerance))
        opposite_b = ((o3 <= tolerance) & (o4 >= -tolerance)) | ((o4 <= tolerance) & (o3 >= -tolerance))
        if np.any(opposite_a & opposite_b):
            return True
    return False


def build_section(parameters, *, span_fraction=.5, check_intersections=True):
    """Build a complete conformal section from resolved independent parameters.

    Chord refers to camberline endpoints. Edge protrusions are retained.
    Axial sections have equal physical chord and angles across span; radial
    sections are extruded between parallel endwalls. Validation is sampled.
    """
    validate_parameters(parameters)
    if not np.isfinite(span_fraction) or not 0 <= span_fraction <= 1:
        raise ValueError("span_fraction must be between zero and one.")
    parameters = deepcopy(parameters)
    geom, th = parameters["geometry"], parameters["thickness"]
    count = parameters["numerics"]["number_of_points"]
    base = .5 * (1 - np.cos(np.linspace(0, np.pi, count)))
    edge = .5 * (1 - np.cos(np.linspace(0, np.pi, max(31, count // 5))))
    leading_fraction, trailing_fraction = th["leading_edge_fraction"], th["trailing_edge_fraction"]
    extent = min(2 * leading_fraction, 1.)
    trailing_extent = min(2 * trailing_fraction, 1.)
    xi = np.unique(np.r_[base, leading_fraction * edge, 1 - trailing_fraction * edge,
                         th["maximum_thickness_location"], leading_fraction,
                         1-trailing_fraction, extent, extent/4,
                         1-trailing_extent, 1-trailing_extent/4])
    # Collapse near-duplicate stations before differentiating; cosine sampling
    # and explicit shoulders can otherwise differ by only roundoff.
    xi = xi[np.r_[True, np.diff(xi) > 1e-14]]
    if geom["type"] == "axial":
        chord = geom["meridional_chord"]
        radius_value = geom["mean_radius"] + (span_fraction - .5) * geom["blade_height"]
        prime_chord = chord / radius_value
        radius = np.full_like(xi, radius_value)
        meridional_distance = chord * xi
        radial_direction = 0
    else:
        rin, rout = geom["inlet_radius"], geom["outlet_radius"]
        radial_direction = np.sign(rout - rin)
        chord = abs(rout - rin)
        prime_chord = abs(np.log(rout / rin))
        radius = rin * np.exp(radial_direction * prime_chord * xi)
        meridional_distance = radial_direction * (radius - rin)

    # A dense integration grid avoids response changes when edge sampling changes.
    dense = np.unique(np.r_[np.linspace(0., 1., max(4001, 8*count+1)), xi])
    dense_beta = turning_angle(dense, parameters["camberline"])
    theta = np.interp(xi, dense, prime_chord * _integral(np.tan(dense_beta), dense))
    beta = turning_angle(xi, parameters["camberline"])
    camber = np.column_stack((prime_chord * xi, theta))
    normal = np.column_stack((-np.sin(beta), np.cos(beta)))
    body, full = denton_thickness(xi, th)
    warnings = []
    turning = beta[-1] - beta[0]
    window = relief_window(xi, leading_fraction)
    trailing_window = relief_window(1-xi, trailing_fraction)
    plus_ratio, minus_ratio = full/2, full/2
    edge_sides = {}
    for name, fraction, local_window in (("leading_edge", leading_fraction, window),
                                          ("trailing_edge", trailing_fraction, trailing_window)):
        strength = parameters[name]["relief_strength"]
        side = parameters[name]["pressure_side"]
        if side == "auto":
            if abs(turning) < 1e-12:
                if strength > 0 and fraction > 0:
                    raise ValueError("Choose a pressure side explicitly before applying relief to a zero-turning blade.")
                side = "plus"
            else:
                side = "plus" if turning > 0 else "minus"
        edge_sides[name] = side
        if side == "plus":
            plus_ratio = plus_ratio * (1-strength*local_window)
        else:
            minus_ratio = minus_ratio * (1-strength*local_window)
        if strength > 0 and fraction == 0:
            warnings.append(f"{name.replace('_', ' ').capitalize()} relief is inactive because its rounding extent is zero.")
    pressure_side = edge_sides["leading_edge"]
    if abs(turning) < 1e-12:
        warnings.append("Zero turning: plus/minus labels are geometric; no pressure side is inferred.")
    relieved = plus_ratio + minus_ratio
    factor = (chord / (radius * prime_chord)
              if parameters["correction"]["preserve_physical_thickness"] else np.ones_like(xi))
    a, b = prime_chord * plus_ratio * factor, prime_chord * minus_ratio * factor
    plus, minus = camber + a[:, None]*normal, camber - b[:, None]*normal
    boundary = _closed_boundary(plus, minus)
    pitch = 2 * np.pi / parameters["cascade"]["blade_count"]
    if not np.isfinite(boundary).all():
        raise ValueError("The selected parameters produce nonfinite blade coordinates.")
    if check_intersections:
        if curves_intersect(boundary):
            raise ValueError("The blade intersects itself. Reduce thickness/relief or change the turning distribution.")
        if curves_intersect(boundary, boundary + [0., pitch]):
            raise ValueError("Neighboring blades touch or intersect. Reduce blade count or thickness.")

    target = chord * relieved
    measured = physical_pair_distance(radius, beta, a, b, radial_direction=radial_direction)
    peak_target = float(target.max())
    active = target > peak_target * 1e-6
    relative = np.divide(measured-target, target, out=np.zeros_like(target), where=active)
    max_error = float(np.max(np.abs(relative[active]))) if np.any(active) else 0.
    max_error_absolute = float(np.max(np.abs(measured-target)))
    beta_derivative = np.gradient(beta, xi, edge_order=2)
    curvature = beta_derivative * np.cos(beta) / prime_chord
    margin = float(np.min(1 - np.abs(curvature) * np.maximum(a, b)))
    if margin <= 0:
        warnings.append("The local normal-coordinate margin is nonpositive; inspect curvature and increase sampling.")
    if (parameters["correction"]["preserve_physical_thickness"]
            and max_error > parameters["correction"]["warning_tolerance"]):
        warnings.append(f"Physical thickness correction error reaches {100*max_error:.2f}%, above the configured tolerance.")
    if leading_fraction + trailing_fraction > 1:
        warnings.append("The leading and trailing rounding regions overlap.")
    if th["shape_exponent"] < 2:
        warnings.append("Body shape exponent below 2: the nominal body peak can have a corner (1) or singular curvature (between 1 and 2).")
    elif not float(th["shape_exponent"]).is_integer():
        warnings.append("A noninteger Denton exponent can limit higher-derivative smoothness at the body peak.")
    if measured.max() <= 0:
        raise ValueError("The blade must have nonzero thickness.")
    peak_index = int(np.argmax(measured))
    diagnostics = {
        "valid": True,
        "nominal_maximum_ratio": float(th["maximum_thickness_ratio"]),
        "nominal_maximum_station": float(th["maximum_thickness_location"]),
        "denton_maximum_ratio": float(full.max()),
        "denton_maximum_station": float(xi[np.argmax(full)]),
        "relieved_maximum_ratio": float(relieved.max()),
        "actual_maximum_thickness_mm": float(1000 * measured[peak_index]),
        "maximum_thickness_station": float(xi[peak_index]),
        "maximum_correction_error_percent": 100 * max_error,
        "maximum_absolute_thickness_error_mm": 1000 * max_error_absolute,
        "peak_normalized_thickness_error_percent": 100 * max_error_absolute / peak_target,
        "max_half_thickness_over_radius": float(np.max(chord * np.maximum(plus_ratio, minus_ratio) / radius)),
        "min_normal_map_margin": margin,
        "pitch_angle": float(pitch),
        "physical_pitch_mm": float(1000 * radius[np.argmin(np.abs(xi-.5))] * pitch),
        "pressure_side": pressure_side,
        "trailing_edge_pressure_side": edge_sides["trailing_edge"],
        "span_fraction": float(span_fraction),
        "nominal_meridional_chord_mm": 1000 * chord,
        "conformal_envelope_chord": float(np.ptp(boundary[:, 0])),
        "sample_count": len(xi),
        "intersection_check": "sampled" if check_intersections else "not performed",
    }
    thickness = {
        "body_ratio": body, "denton_ratio": full, "relieved_ratio": relieved,
        "plus_ratio": plus_ratio, "minus_ratio": minus_ratio,
        "asymmetry_window": window, "correction_factor": factor,
        "trailing_relief_window": trailing_window,
        "plus_chart_offset": a, "minus_chart_offset": b,
        "target_physical": target, "measured_physical": measured,
        "relative_error": relative,
    }
    return Section(parameters, xi, float(prime_chord), float(chord), radius,
                   meridional_distance, beta, camber, plus, minus, boundary,
                   thickness, diagnostics, warnings, pressure_side)
