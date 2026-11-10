"""Explicit stream-surface impeller construction, independent of UI and meshing.

Positive theta follows rotation about +z; beta is measured from the positive
meridional tangent toward positive theta. XYZ is (r cos theta, r sin theta, z).
No flow-side label is inferred for a reversible machine.
"""

from copy import deepcopy
from dataclasses import dataclass

import numpy as np
from ..blade_parametrization.geometry import curves_intersect, denton_thickness
from .config import config_fingerprint, normalize_config, section_parameters
from .mapping import StreamSurfaceMap
from .meridional import BladeEdges, MeanlineDimensions, MeridionalChannel, WallShape
from .rake import RakeControl, section_camber


def chart_to_xyz(mapping_section, chart):
    """Map dimensionless (m_prime, theta) coordinates onto one curved surface."""
    chart = np.asarray(chart, dtype=float)
    zr = mapping_section.from_prime(chart[..., 0])
    return np.stack((zr[..., 1] * np.cos(chart[..., 1]),
                     zr[..., 1] * np.sin(chart[..., 1]), zr[..., 0]), axis=-1)


@dataclass
class ImpellerSection:
    v: float
    u: np.ndarray
    parameters: dict
    mapping: object
    meridional_u: np.ndarray
    prime: np.ndarray
    radius: np.ndarray
    beta: np.ndarray
    theta: np.ndarray
    target_thickness_m: np.ndarray
    measured_thickness_m: np.ndarray
    plus_chart: np.ndarray
    minus_chart: np.ndarray
    camber: np.ndarray
    plus: np.ndarray
    minus: np.ndarray
    diagnostics: dict

    def from_prime(self, prime):
        return self.mapping.from_prime(prime)


@dataclass
class ImpellerGeometry:
    config: dict
    fingerprint: str
    channel: MeridionalChannel
    mapping: StreamSurfaceMap
    u: np.ndarray
    v: np.ndarray
    sections: list
    plus: np.ndarray
    minus: np.ndarray
    camber: np.ndarray
    blade_count: int
    diagnostics: dict
    rake_control: object = None

    def section(self, v, *, check_intersections=True):
        """Evaluate any span on the same streamwise grid (not an XYZ loft)."""
        if config_fingerprint(normalize_config(self.config)) != self.fingerprint:
            raise ValueError("Geometry inputs changed after evaluation; call build_impeller with the edited configuration.")
        if isinstance(v, bool) or not np.isscalar(v) or not np.isfinite(v) or not 0 <= v <= 1:
            raise ValueError("v must be a finite scalar in [0, 1].")
        control = _interpolate_section(self.config, self.mapping, float(v))
        return _build_section(self.mapping, control, self.u,
                              self.config["numerics"]["integration_points"],
                              self.blade_count, check_intersections, rake=self.rake_control)


def _interpolate_section(config, mapping, v):
    controls = config["sections"]
    spans = np.array([s["v"] for s in controls])
    result = {"v": float(v)}
    for group in ("camberline", "thickness"):
        result[group] = {}
        for key, value in controls[0][group].items():
            result[group][key] = (value if isinstance(value, str) else
                                  float(np.interp(v, spans, [s[group][key] for s in controls])))
    result["stacking_offset_deg"] = float(np.interp(v, spans, [s["stacking_offset_deg"] for s in controls]))
    if config["inlet_interpolation"] == "tangent_radius":
        radii = np.array([mapping.point(0, s)[1] for s in spans])
        index = min(np.searchsorted(spans, v, side="right") - 1, len(spans) - 2)
        r0, r1 = radii[index:index + 2]
        if abs(r1 - r0) < 1e-12 * max(radii):
            raise ValueError("tangent_radius needs distinct LE control radii; use angle_span for this footprint.")
        radius = float(mapping.point(0, v)[1])
        fraction = (radius - r0) / (r1 - r0)
        if fraction < -1e-10 or fraction > 1 + 1e-10:
            raise ValueError("LE radius leaves its control interval; use angle_span or add span controls.")
        angles = [controls[i]["camberline"]["inlet_metal_angle_deg"] for i in (index, index + 1)]
        slopes = np.tan(np.deg2rad(angles))
        result["camberline"]["inlet_metal_angle_deg"] = float(np.rad2deg(np.arctan(slopes[0] + fraction * (slopes[1] - slopes[0]))))
    return result


def _build_section(mapping, control, u, integration_points, blade_count, check_intersections, *, rake=None):
    section_map = mapping.section(control["v"])
    parameters = section_parameters(control, section_map.length_m)
    beta, theta, rake_details = section_camber(section_map, control, u, integration_points, rake=rake)
    prime = section_map.prime_length * u
    camber_chart = np.column_stack((prime, theta))
    radius = section_map.from_prime(prime)[:, 1]
    target = section_map.length_m * denton_thickness(u, parameters["thickness"])[1]
    # Explicit local metric correction: ds^2 = r^2 (dm_prime^2 + dtheta^2).
    # This targets paired Euclidean thickness to first order; measure the finite
    # 3-D distance afterwards, without a root solve or remapping iteration.
    offset = (target / (2 * radius))[:, None] * np.column_stack((-np.sin(beta), np.cos(beta)))
    plus_chart, minus_chart = camber_chart + offset, camber_chart - offset
    plus, minus = chart_to_xyz(section_map, plus_chart), chart_to_xyz(section_map, minus_chart)
    measured = np.linalg.norm(plus - minus, axis=-1)
    active = target > np.max(target) * 1e-8
    relative_error = np.abs(measured[active] / target[active] - 1)
    boundary = np.concatenate((plus_chart, minus_chart[-2::-1]))
    self_intersection = neighbor_intersection = None
    if check_intersections:
        self_intersection = curves_intersect(boundary)
        neighbor_intersection = curves_intersect(boundary, boundary + (0, 2 * np.pi / blade_count))
        if self_intersection or neighbor_intersection:
            raise ValueError(f"Blade section v={control['v']:.6g} intersects itself or its periodic neighbor; adjust turning/thickness/count.")
    side_prime = np.r_[plus_chart[:, 0], minus_chart[:, 0]]
    side_u = section_map.u_from_prime(side_prime)
    diagnostics = {
        "rake": rake_details,
        "v": float(control["v"]), "meridional_length_m": section_map.length_m,
        "conformal_length": section_map.prime_length,
        "wrap_angle_deg": float(np.rad2deg(theta[-1] - theta[0])),
        "maximum_target_thickness_m": float(np.max(target)),
        "maximum_measured_thickness_m": float(np.max(measured)),
        "maximum_relative_thickness_error": float(np.max(relative_error)),
        "maximum_absolute_thickness_error_m": float(np.max(np.abs(measured - target))),
        "minimum_side_meridional_u": float(side_u.min()),
        "maximum_side_meridional_u": float(side_u.max()),
        "self_intersection_sampled": self_intersection,
        "neighbor_intersection_sampled": neighbor_intersection,
    }
    return ImpellerSection(control["v"], u, deepcopy(control), section_map,
                           section_map.u_from_prime(prime), prime, radius, beta, theta,
                           target, measured, plus_chart, minus_chart,
                           chart_to_xyz(section_map, camber_chart), plus, minus, diagnostics)


def build_impeller(config, *, check_intersections=True):
    """Build one blade over curved stream surfaces and sampled diagnostics.

    Span controls are interpolated as scalar parameters, then each section is
    constructed on its own surface. Result u is normalized conformal distance;
    mapping.point(u,v) instead uses normalized meridional footprint position.
    Rounded endpoints coincide exactly. Caps and periodic copies are exports.
    """
    config = normalize_config(config)
    channel = MeridionalChannel(MeanlineDimensions(**config["dimensions"]),
                                WallShape(**config["hub"]), WallShape(**config["shroud"]),
                                control_point_displacements=config["control_point_displacements"])
    numerics = config["numerics"]
    mapping = StreamSurfaceMap(channel, BladeEdges(**config["blade_edges"]),
                               integration_points=numerics["integration_points"])
    rake = RakeControl(mapping, config, lambda span: _interpolate_section(config, mapping, span))
    v = np.unique(np.r_[np.linspace(0, 1, numerics["spanwise_points"]),
                        [s["v"] for s in config["sections"]]])
    controls = [_interpolate_section(config, mapping, span) for span in v]
    base = 0.5 * (1 - np.cos(np.linspace(0, np.pi, numerics["streamwise_points"])))
    edge = 0.5 * (1 - np.cos(np.linspace(0, np.pi, 25)))
    extras = []
    for control in controls:
        t = control["thickness"]
        extras.extend([t["maximum_thickness_location"], t["leading_edge_fraction"], 1 - t["trailing_edge_fraction"]])
        extras.extend(t["leading_edge_fraction"] * edge)
        extras.extend(1 - t["trailing_edge_fraction"] * edge)
    u = np.unique(np.r_[base, extras])
    u = u[np.r_[True, np.diff(u) > 1e-13]]
    # Deduplication can retain a shoulder immediately before 1 instead of the
    # exact endpoint. Preserve both rounded seams even for extreme controls.
    u[0], u[-1] = 0.0, 1.0
    sections = [_build_section(mapping, control, u, numerics["integration_points"],
                                config["blade_count"], check_intersections, rake=rake) for control in controls]
    plus, minus, camber = [np.stack([getattr(s, name) for s in sections]) for name in ("plus", "minus", "camber")]
    if not np.all(np.isfinite(np.stack((plus, minus, camber)))):
        raise ValueError("Impeller coordinates must be finite.")
    max_error = max(s.diagnostics["maximum_relative_thickness_error"] for s in sections)
    warnings = []
    if max_error > 0.01:
        warnings.append("Explicit local-metric thickness differs from measured paired 3-D distance by more than 1%; inspect thickness/curvature.")
    if not check_intersections:
        warnings.append("Section intersection checks were disabled.")
    diagnostics = {
        "rake": rake.diagnostics(sections),
        "coordinate_system": "XYZ=(r*cos(theta), r*sin(theta), z); metres; beta from +meridional toward +theta",
        "thickness_method": "explicit symmetric local metric, measured paired Euclidean residual; no iterative correction",
        "maximum_relative_thickness_error": max_error,
        "maximum_absolute_thickness_error_m": max(s.diagnostics["maximum_absolute_thickness_error_m"] for s in sections),
        "intersections_checked": bool(check_intersections),
        "validation_scope": "Sampled section contours and adjacent periodic copies; not a continuous/global 3-D intersection proof.",
        "shrouded": True, "tip_clearance_m": 0.0, "splitters": False,
        "streamwise_samples": len(u), "spanwise_samples": len(v),
        "sections": [s.diagnostics for s in sections], "warnings": warnings,
    }
    return ImpellerGeometry(config, config_fingerprint(config), channel, mapping, u, v, sections,
                             plus, minus, camber, config["blade_count"], diagnostics, rake)
