"""Geometry invariants against the real reversible-example seed."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from turbodash.geometry.impeller import build_impeller, load_config, normalize_config
from turbodash.geometry.impeller.config import config_fingerprint
from turbodash.geometry.impeller.geometry import _build_section
from turbodash.geometry.impeller.mapping import StreamSurfaceMap


CASE = Path(__file__).resolve().parents[1] / "development/impeller/examples/reversible_impeller.yaml"


@pytest.fixture(scope="module")
def baseline():
    return build_impeller(load_config(CASE))


def test_real_geometry_closed_round_ends_and_endpoint_angles(baseline):
    np.testing.assert_array_equal(baseline.plus[:, [0, -1]], baseline.minus[:, [0, -1]])
    for section in baseline.sections:
        expected = [section.parameters["camberline"][key] for key in
                    ("inlet_metal_angle_deg", "outlet_metal_angle_deg")]
        np.testing.assert_allclose(np.rad2deg(section.beta[[0, -1]]), expected, atol=1e-12)
        assert not section.diagnostics["self_intersection_sampled"]
        assert not section.diagnostics["neighbor_intersection_sampled"]
    assert baseline.diagnostics["maximum_relative_thickness_error"] < 1e-4


def test_rms_inlet_angle_uses_actual_radius_not_midspan(baseline):
    d = baseline.config["dimensions"]
    rh, rs = d["inlet_hub_radius"], d["inlet_shroud_radius"]
    rms_radius = np.sqrt((rh**2 + rs**2) / 2)
    v_rms = (rms_radius - rh) / (rs - rh)
    rms = baseline.section(v_rms)
    assert np.rad2deg(rms.beta[0]) == pytest.approx(-47.28, abs=1e-10)
    assert abs(np.rad2deg(baseline.section(0.5).beta[0]) + 47.28) > 1


def test_interior_section_follows_channel_not_xyz_interpolation(baseline):
    middle = baseline.section(0.5)
    zr = baseline.mapping.point(middle.meridional_u, 0.5)
    np.testing.assert_allclose(np.hypot(middle.camber[:, 0], middle.camber[:, 1]), zr[:, 1], atol=1e-14)
    np.testing.assert_allclose(middle.camber[:, 2], zr[:, 0], atol=1e-14)
    assert np.max(np.linalg.norm(middle.camber - (baseline.camber[0] + baseline.camber[-1]) / 2, axis=-1)) > 1e-3


def test_independent_sections_middle_control_and_stacking(baseline):
    config = deepcopy(baseline.config)
    config["sections"][-1]["thickness"]["maximum_thickness_m"] *= 1.25
    config["sections"][-1]["camberline"]["outlet_metal_angle_deg"] = -38
    config["sections"][-1]["stacking_offset_deg"] = 4
    middle = deepcopy(config["sections"][0])
    middle["v"] = 0.43
    middle["camberline"]["outlet_metal_angle_deg"] = -31
    config["sections"].insert(1, middle)
    changed = build_impeller(config)
    np.testing.assert_allclose(changed.plus[0], baseline.plus[0])
    assert np.rad2deg(changed.sections[-1].beta[-1]) == pytest.approx(-38)
    assert np.rad2deg(changed.sections[-1].theta[0]) == pytest.approx(4)
    assert changed.section(0.43).parameters["camberline"]["outlet_metal_angle_deg"] == -31
    assert changed.sections[-1].diagnostics["maximum_target_thickness_m"] > baseline.sections[-1].diagnostics["maximum_target_thickness_m"]


def test_uniform_dimensional_scaling(baseline):
    config = deepcopy(baseline.config)
    factor = 3.7
    for key in ("inlet_hub_radius", "inlet_shroud_radius", "outlet_radius", "outlet_width", "axial_length"):
        config["dimensions"][key] *= factor
    for section in config["sections"]:
        for key in ("maximum_thickness_m", "leading_edge_thickness_m", "trailing_edge_thickness_m"):
            section["thickness"][key] *= factor
    scaled = build_impeller(config)
    np.testing.assert_allclose(scaled.plus, baseline.plus * factor, rtol=2e-12, atol=1e-13)
    assert scaled.diagnostics["maximum_relative_thickness_error"] == pytest.approx(baseline.diagnostics["maximum_relative_thickness_error"], abs=1e-11)


def test_explicit_metric_has_analytic_cylinder_paired_distance(baseline):
    class Cylinder:
        def point(self, u, v):
            u, v = np.broadcast_arrays(u, v)
            return np.stack((0.2 * u, np.full_like(u, 0.1)), axis=-1)

        def derivatives(self, u, v):
            u, v = np.broadcast_arrays(u, v)
            return np.stack((np.full_like(u, 0.2), np.zeros_like(u)), axis=-1), np.zeros(u.shape + (2,))

    control = deepcopy(baseline.config["sections"][0])
    control["camberline"]["inlet_metal_angle_deg"] = 45
    control["camberline"]["outlet_metal_angle_deg"] = 45
    section = _build_section(StreamSurfaceMap(Cylinder()), control, baseline.u, 2001, 20, True)
    t, beta, r = section.target_thickness_m, section.beta, section.radius
    exact = np.sqrt((t * np.sin(beta))**2 + (2 * r * np.sin(t * np.cos(beta) / (2 * r)))**2)
    np.testing.assert_allclose(section.measured_thickness_m, exact, rtol=1e-11, atol=1e-15)


def test_invalid_geometry_and_periodic_overlap_fail(baseline):
    config = deepcopy(baseline.config)
    config["blade_count"] = 1000
    with pytest.raises(ValueError, match="periodic neighbor"):
        build_impeller(config)
    config = deepcopy(baseline.config)
    config["hub"]["bow"] = 10
    with pytest.raises(ValueError):
        build_impeller(config)


@pytest.mark.parametrize("change", [
    {"unknown": 1}, {"blade_count": 20.0}, {"construction": {"shrouded": False}},
    {"construction": {"splitters": True}}, {"numerics": {"spanwise_points": 2}},
    {"inlet_interpolation": "arbitrary"},
])
def test_config_rejects_unsupported_inputs(baseline, change):
    with pytest.raises(ValueError):
        normalize_config(baseline.config | change)


def test_config_is_defensive_and_metadata_does_not_change_geometry_identity(baseline):
    config = normalize_config(baseline.config)
    config["source"]["note"] = "updated description"
    config["name"] = "renamed"
    assert config_fingerprint(config) == baseline.fingerprint
    assert "note" not in baseline.config["source"]
    assert normalize_config(config) == config


def test_changed_result_config_requires_rebuild(baseline):
    changed = deepcopy(baseline)
    changed.config["sections"][0]["camberline"]["outlet_metal_angle_deg"] += 1
    with pytest.raises(ValueError, match="build_impeller"):
        changed.section(0.2)


def test_near_endpoint_shape_station_keeps_exact_rounded_seam(baseline):
    config = deepcopy(baseline.config)
    for section in config["sections"]:
        section["thickness"]["maximum_thickness_location"] = 1 - 5e-14
    geometry = build_impeller(config, check_intersections=False)
    assert geometry.u[-1] == 1.0
    np.testing.assert_array_equal(geometry.plus[:, -1], geometry.minus[:, -1])
