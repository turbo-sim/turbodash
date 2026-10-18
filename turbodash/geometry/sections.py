"""Row-section construction shared by physical geometry and exporters."""
import numpy as np
from turbodash.geometry.blade import compute_blade_coordinates_cartesian
from turbodash.core.turbine import get_blade_row_geometry


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
    inlet_index = 0 if row == "stator" else 2
    inlet, outlet = stage["flow_stations"][inlet_index:inlet_index + 2]
    geometry = get_blade_row_geometry(stage, row, turbine_type="axial", discrete=True)
    mean_in, height_in = _station_geometry(inlet)
    mean_out, height_out = _station_geometry(outlet)
    radius_in = mean_in + (span_fraction - 0.5) * height_in
    radius_out = mean_out + (span_fraction - 0.5) * height_out
    omega = 0.0
    if row == "rotor":
        omega = 0.5 * (float(inlet["u"]) / mean_in + float(outlet["u"]) / mean_out)

    angle_in = np.arctan2(
        float(inlet["v_t"]) * mean_in / radius_in - omega * radius_in,
        float(inlet["v_m"]),
    )
    angle_out = np.arctan2(
        float(outlet["v_t"]) * mean_out / radius_out - omega * radius_out,
        float(outlet["v_m"]),
    )
    chord = _positive(geometry["chord_meridional"], "Meridional chord")
    axial, tangential, *_ = compute_blade_coordinates_cartesian(
        camberline_type=camberline_type,
        x1=0.0,
        y1=0.0,
        beta1=angle_in,
        beta2=angle_out,
        chord_ax=chord,
        loc_max=float(geometry["maximum_thickness_location"]),
        thickness_max=float(geometry["maximum_thickness"]),
        thickness_trailing=float(geometry["trailing_edge_thickness"]),
        wedge_trailing=np.deg2rad(float(geometry["trailing_edge_wedge_angle"])),
        radius_leading=float(geometry["leading_edge_radius"]),
        N_points=profile_points,
        thickness_model=thickness_model,
    )
    axial = np.asarray(axial, dtype=float)
    tangential = np.asarray(tangential, dtype=float)
    radius = radius_in + np.clip(axial / chord, 0.0, 1.0) * (radius_out - radius_in)
    theta = tangential / radius
    points = np.column_stack((axial, radius * np.cos(theta), radius * np.sin(theta)))
    if not np.all(np.isfinite(points)):
        raise ValueError("Generated blade coordinates must be finite.")
    if np.linalg.norm(points[0] - points[-1]) <= 1.0e-10:
        points = points[:-1]
    points = np.roll(points, -int(np.argmin(points[:, 0])), axis=0)
    points = np.vstack((points, points[:1]))
    return {
        "span_fraction": float(span_fraction),
        "angle_in_deg": float(np.rad2deg(angle_in)),
        "angle_out_deg": float(np.rad2deg(angle_out)),
        "points": points,
    }

