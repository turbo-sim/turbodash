"""Row-section construction shared by physical geometry and exporters."""
import numpy as np
from turbodash.geometry.blade_parametrization.geometry import build_section
from turbodash.geometry.blade_parametrization.parameters import resolve_parameters
from turbodash.geometry.meanline import state_for_stage_row


def _positive(value, name):
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive.")
    return value


def _station_geometry(station):
    radius = _positive(station["r"], "Station radius")
    height = _positive(station["H"], "Station height")
    if radius <= height / 2.0:
        raise ValueError("The hub radius must be positive.")
    _positive(station["v_m"], "Station meridional velocity")
    return radius, height


def _make_section(stage, row, span_fraction, profile_points, camberline_type, thickness_model):
    """The same untwisted cylindrical blade seen in the modeller's preview."""
    values = resolve_parameters(state_for_stage_row(stage, row)).values
    if values["geometry"]["type"] != "axial":
        raise ValueError("TurboGrid span sections support axial cylindrical blades only.")
    values["numerics"]["number_of_points"] = max(601, int(profile_points))
    section = build_section(values, span_fraction=span_fraction, check_intersections=False)
    radius = values["geometry"]["mean_radius"] + (span_fraction-.5)*values["geometry"]["blade_height"]
    axial, theta = section.boundary[:, 0]*radius, section.boundary[:, 1]
    points = np.column_stack((axial, radius * np.cos(theta), radius * np.sin(theta)))
    if not np.all(np.isfinite(points)):
        raise ValueError("Generated blade coordinates must be finite.")
    if np.linalg.norm(points[0] - points[-1]) <= 1.0e-10:
        points = points[:-1]
    points = np.roll(points, -int(np.argmin(points[:, 0])), axis=0)
    points = np.vstack((points, points[:1]))
    return {
        "span_fraction": float(span_fraction),
        "angle_in_deg": values["camberline"]["inlet_metal_angle_deg"],
        "angle_out_deg": values["camberline"]["outlet_metal_angle_deg"],
        "points": points,
    }

