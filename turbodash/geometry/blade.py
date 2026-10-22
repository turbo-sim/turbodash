"""Coordinate compatibility adapters for the production conformal blade kernel.

Old callable signatures remain usable by plotting/export integrations. Legacy
model names and wedge arguments do not select geometry laws: every call uses
the same conformal camberline and rational Denton implementation. The retired
implementation is archived in ``geometry/legacy_parametrization/blade.py``.
"""
import numpy as np

from .blade_parametrization.parameters import ParameterState, state_from_dict
from .meanline import parameters_from_meanline, section_from_state


def _parameter_state(geometry, turbine_type, parameters, points, thickness_leading, exponent):
    if parameters is not None:
        return parameters if isinstance(parameters, ParameterState) else state_from_dict(parameters)
    state = parameters_from_meanline(geometry, turbine_type)[0]
    source = state.source
    source["numerics"]["number_of_points"] = max(601, int(points))
    source["thickness"]["shape_exponent"] = float(exponent)
    if thickness_leading > 0:
        source["thickness"]["leading_edge_thickness_ratio"] = float(thickness_leading / geometry["chord_meridional"])
    return ParameterState(source)


def section_coordinates(section, points=None):
    """Physical 2D coordinates; axial means developed (axial,tangential)."""
    points = section.boundary if points is None else np.asarray(points)
    geometry = section.parameters["geometry"]
    if geometry["type"] == "axial":
        return points * float(section.radius[0])
    direction = np.sign(geometry["outlet_radius"]-geometry["inlet_radius"])
    radius = geometry["inlet_radius"] * np.exp(direction*points[:, 0])
    return np.column_stack((radius*np.cos(points[:, 1]), radius*np.sin(points[:, 1])))


def compute_blade_coordinates_cartesian(
    camberline_type, x1, y1, beta1, beta2, chord_ax, loc_max,
    thickness_max, thickness_trailing, wedge_trailing, radius_leading, N_points,
    thickness_model="rational_denton", curvature_cp=None, thickness_control_points=None,
    thickness_leading=0., thickness_shape_exponent=2., *, parameters=None,
):
    """Return the new closed contour, stagger and endpoint chord (SI/radians)."""
    if curvature_cp is not None or thickness_control_points is not None:
        raise ValueError("Legacy spline control points are retired; use conformal turning/fullness parameters.")
    geometry = dict(radius_in=1., radius_out=1., height=.01, chord_meridional=float(chord_ax),
                    metal_angle_in=float(np.rad2deg(beta1)), metal_angle_out=float(np.rad2deg(beta2)),
                    maximum_thickness=float(thickness_max), maximum_thickness_location=float(loc_max),
                    trailing_edge_thickness=float(thickness_trailing), leading_edge_radius=float(radius_leading),
                    blade_count=2)
    state = _parameter_state(geometry, "axial", parameters, N_points, thickness_leading, thickness_shape_exponent)
    section = section_from_state(state)
    if section.parameters["geometry"]["type"] != "axial":
        raise ValueError("Cartesian blade coordinates require an axial stream surface.")
    points = section_coordinates(section) + [x1, y1]
    endpoints = section_coordinates(section, section.camber[[0, -1]])
    vector = endpoints[-1]-endpoints[0]
    return points[:, 0], points[:, 1], np.arctan2(vector[1], vector[0]), float(np.linalg.norm(vector))


def compute_blade_coordinates_radial(
    camberline_type, r1, r2, metal_angle1, metal_angle2, theta0, loc_max,
    thickness_max, thickness_trailing, wedge_trailing, radius_leading, N_points,
    thickness_model="rational_denton", curvature_cp=None, thickness_control_points=None,
    thickness_leading=0., thickness_shape_exponent=2., *, parameters=None,
):
    if curvature_cp is not None or thickness_control_points is not None:
        raise ValueError("Legacy spline control points are retired; use conformal turning/fullness parameters.")
    geometry = dict(radius_in=float(r1), radius_out=float(r2), height=.01,
                    chord_meridional=float(abs(r2-r1)), metal_angle_in=float(np.rad2deg(metal_angle1)),
                    metal_angle_out=float(np.rad2deg(metal_angle2)), maximum_thickness=float(thickness_max),
                    maximum_thickness_location=float(loc_max), trailing_edge_thickness=float(thickness_trailing),
                    leading_edge_radius=float(radius_leading), blade_count=2)
    state = _parameter_state(geometry, "radial", parameters, N_points, thickness_leading, thickness_shape_exponent)
    section = section_from_state(state)
    if section.parameters["geometry"]["type"] != "radial":
        raise ValueError("Radial blade coordinates require a radial stream surface.")
    points = section_coordinates(section)
    rotation = np.array([[np.cos(theta0), -np.sin(theta0)], [np.sin(theta0), np.cos(theta0)]])
    points = points @ rotation.T
    ends = section_coordinates(section, section.camber[[0, -1]])
    vector = ends[-1]-ends[0]
    return points[:, 0], points[:, 1], np.arctan2(vector[1], vector[0]), float(np.linalg.norm(vector))
