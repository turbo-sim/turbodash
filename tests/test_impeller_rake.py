"""Independent geometry checks for explicit exit-rake control."""

from copy import deepcopy
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.integrate import quad

from turbodash.geometry.impeller import build_impeller, load_config, normalize_config
from turbodash.geometry.impeller.output import blade_surface


CASE = Path(__file__).parents[1] / "development/impeller/examples/reversible_impeller.yaml"


@pytest.fixture(scope="module")
def config():
    result = load_config(CASE)
    result["numerics"].update(streamwise_points=81, spanwise_points=7, integration_points=1001)
    result["rake"] = {"mode": "free", "angle_deg": 0.0, "reference_azimuth_deg": None}
    return result


def with_rake(config, angle=0, mode="preserve_le", reference=None):
    result = deepcopy(config)
    result["rake"] = {"mode": mode, "angle_deg": angle, "reference_azimuth_deg": reference}
    return result


@pytest.fixture(scope="module")
def legacy(config):
    return build_impeller(config)


@pytest.fixture(scope="module")
def zero(config):
    return build_impeller(with_rake(config))


def xyz_rake(geometry, v, step=2e-5):
    """Recover signed local rake from the actual Cartesian TE curve."""
    point = geometry.section(v).camber[-1]
    derivative = (geometry.section(v + step).camber[-1]
                  - geometry.section(v - step).camber[-1]) / (2 * step)
    radius = np.linalg.norm(point[:2])
    radial_velocity = np.dot(point[:2], derivative[:2]) / radius
    circumferential_velocity = np.dot([-point[1], point[0]], derivative[:2]) / radius
    meridional_velocity = np.hypot(radial_velocity, derivative[2])
    return np.rad2deg(np.arctan2(circumferential_velocity, meridional_velocity))


def test_missing_rake_retains_legacy_geometry(config, legacy):
    old = deepcopy(config)
    old.pop("rake")
    loaded = build_impeller(old)
    assert_allclose(loaded.plus, legacy.plus, rtol=0, atol=0)
    assert_allclose(loaded.minus, legacy.minus, rtol=0, atol=0)


def test_zero_rake_preserves_le_endpoint_angles_and_monotonicity(zero, legacy):
    azimuth = zero.sections[0].theta[-1]
    for original, section in zip(legacy.sections, zero.sections):
        assert section.theta[0] == pytest.approx(original.theta[0], abs=1e-14)
        assert section.theta[-1] == pytest.approx(azimuth, abs=2e-13)
        assert_allclose(section.beta[[0, -1]], original.beta[[0, -1]], atol=2e-14)
        direction = np.sign(section.beta[-1] - section.beta[0])
        assert np.min(direction * np.diff(section.beta)) >= -1e-13
        assert section.beta.min() >= min(original.beta[[0, -1]]) - 1e-13
        assert section.beta.max() <= max(original.beta[[0, -1]]) + 1e-13
    # The new rake constraint actually changes the old nonzero-rake geometry.
    assert abs(legacy.sections[-1].theta[-1] - legacy.sections[0].theta[-1]) > 0.3
    assert np.max(np.abs(zero.sections[-1].beta - legacy.sections[-1].beta)) > 0.1
    for v in (0.137, 0.493, 0.817):
        assert xyz_rake(zero, v) == pytest.approx(0, abs=2e-6)
        assert zero.section(v).theta[-1] == pytest.approx(azimuth, abs=2e-13)


@pytest.mark.parametrize("angle", [-10.0, 10.0])
def test_signed_rake_recovered_from_cartesian_derivatives(config, angle):
    geometry = build_impeller(with_rake(config, angle))
    for v in (0.137, 0.493, 0.817):
        assert xyz_rake(geometry, v) == pytest.approx(angle, abs=2e-5)
    assert geometry.diagnostics["rake"]["target_angle_deg"] == angle
    assert geometry.diagnostics["rake"]["max_abs_rake_error_deg"] < 1e-7


def test_te_stack_preserves_angle_and_thickness_laws(config, legacy):
    geometry = build_impeller(with_rake(config, mode="trailing_edge"))
    for original, section in zip(legacy.sections, geometry.sections):
        assert_allclose(section.beta, original.beta, atol=1e-14)
        assert_allclose(section.target_thickness_m, original.target_thickness_m, atol=1e-15)
        assert_allclose(section.measured_thickness_m, original.measured_thickness_m, atol=2e-16)
        assert section.theta[-1] == pytest.approx(legacy.sections[0].theta[-1], abs=1e-13)
    assert abs(geometry.sections[-1].theta[0] - legacy.sections[-1].theta[0]) > 0.3
    # Evaluate away from control/sample spans: endpoint-only offset interpolation
    # would leave a bowed trailing edge and fail this check.
    assert xyz_rake(geometry, 0.413) == pytest.approx(0, abs=2e-6)


def test_default_example_keeps_gradual_turning_and_allows_le_lean(config, legacy):
    """The editable default must retain the seed laws, not the constrained law."""
    selected = load_config(CASE.with_name("reversible_impeller_zero_rake.yaml"))
    selected["numerics"] = deepcopy(config["numerics"])
    assert selected["rake"]["mode"] == "trailing_edge"
    assert selected["rake"]["angle_deg"] == 0.0
    geometry = build_impeller(selected)
    reference = legacy.sections[0].theta[-1]
    for v in (0.0, 0.137, 0.493, 0.817, 1.0):
        original, section = legacy.section(v), geometry.section(v)
        assert_allclose(section.beta, original.beta, rtol=0, atol=1e-14)
        assert_allclose(section.target_thickness_m, original.target_thickness_m,
                        rtol=0, atol=1e-15)
        assert_allclose(section.theta - section.theta[0],
                        original.theta - original.theta[0], rtol=0, atol=2e-14)
        assert section.theta[-1] == pytest.approx(reference, abs=2e-13)
        expected_le = reference - (original.theta[-1] - original.theta[0])
        assert section.theta[0] == pytest.approx(expected_le, abs=2e-13)
    assert_allclose(geometry.section(0).camber, legacy.section(0).camber, atol=2e-15)
    assert abs(geometry.section(1).theta[0] - geometry.section(0).theta[0]) > 0.3
    for v in (0.137, 0.493, 0.817):
        assert xyz_rake(geometry, v) == pytest.approx(0, abs=2e-6)
    archived = load_config(CASE.with_name("reversible_impeller_zero_rake_preserve_le.yaml"))
    assert archived["rake"]["mode"] == "preserve_le"


def test_reference_azimuth_and_auto_choice_do_not_depend_on_output_spans(config, zero):
    reference = np.rad2deg(zero.sections[0].theta[-1])
    explicit = build_impeller(with_rake(config, reference=reference))
    assert_allclose(explicit.plus, zero.plus, atol=1e-13)
    changed = with_rake(config)
    changed["numerics"]["spanwise_points"] = 13
    refined = build_impeller(changed)
    assert refined.sections[0].theta[-1] == pytest.approx(zero.sections[0].theta[-1], abs=1e-13)
    assert_allclose(refined.section(0.413).plus, zero.section(0.413).plus, atol=1e-13)


def test_optional_middle_control_and_nonzero_le_offsets(config):
    changed = with_rake(config)
    changed["sections"][0]["stacking_offset_deg"] = 0.2
    changed["sections"][-1]["stacking_offset_deg"] = 0.3
    middle = deepcopy(changed["sections"][0])
    middle["v"] = 0.43
    middle["stacking_offset_deg"] = 0.1
    middle["camberline"]["inlet_metal_angle_deg"] = -44
    middle["camberline"]["outlet_metal_angle_deg"] = -28
    middle["thickness"]["maximum_thickness_m"] *= 0.95
    changed["sections"].insert(1, middle)
    geometry = build_impeller(changed)
    for v in (0, 0.23, 0.43, 0.71, 1):
        section = geometry.section(v)
        expected_le = np.interp(v, [0, .43, 1], [.2, .1, .3])
        assert np.rad2deg(section.theta[0]) == pytest.approx(expected_le, abs=1e-12)
        assert section.theta[-1] == pytest.approx(geometry.sections[0].theta[-1], abs=2e-13)


def test_swept_inclined_te_matches_independent_meridional_edge_integral(config):
    changed = with_rake(config, angle=12, mode="trailing_edge", reference=-42)
    changed["dimensions"]["outlet_flow_angle_deg"] = 80
    changed["hub"]["outlet_angle_deg"] = 80
    changed["shroud"]["outlet_angle_deg"] = 80
    changed["blade_edges"].update(trailing_hub=.93, trailing_shroud=.98, trailing_bow=.01)
    geometry = build_impeller(changed)
    for v in (0.37, 0.83, 1.0):
        integral, _ = quad(
            lambda v: np.linalg.norm(geometry.mapping.derivatives(1, v)[1])
            / geometry.mapping.point(1, v)[1], 0, v, epsabs=1e-11, epsrel=1e-11,
        )
        expected = np.deg2rad(-42) + np.tan(np.deg2rad(12)) * integral
        assert geometry.section(v).theta[-1] == pytest.approx(expected, abs=2e-9)
    assert xyz_rake(geometry, .413) == pytest.approx(12, abs=2e-4)


def test_impossible_preserved_le_constraint_is_rejected(config):
    with pytest.raises(ValueError, match="(?i)(feasib|rake|wrap|mean)"):
        build_impeller(with_rake(config, reference=0))
    with pytest.raises(ValueError, match="(?i)(feasib|rake|wrap|mean)"):
        build_impeller(with_rake(config, angle=-60))


def test_constant_zero_angle_has_valid_collapsed_constraint(config):
    changed = with_rake(config)
    for section in changed["sections"]:
        section["camberline"].update(inlet_metal_angle_deg=0, outlet_metal_angle_deg=0)
    geometry = build_impeller(changed)
    assert_allclose(geometry.camber[..., 1], 0, atol=1e-14)
    for section in geometry.sections:
        assert_allclose(section.beta, 0, atol=1e-14)
        assert_allclose(section.theta, 0, atol=1e-14)


def test_rake_geometry_is_scale_invariant_and_closed(config, zero):
    changed = with_rake(config)
    factor = 2.3
    for key in ("inlet_hub_radius", "inlet_shroud_radius", "outlet_radius", "outlet_width", "axial_length"):
        changed["dimensions"][key] *= factor
    for section in changed["sections"]:
        for key in ("maximum_thickness_m", "leading_edge_thickness_m", "trailing_edge_thickness_m"):
            section["thickness"][key] *= factor
    scaled = build_impeller(changed)
    assert_allclose(scaled.plus, factor * zero.plus, rtol=2e-11, atol=1e-13)
    quality = blade_surface(zero).validate(closed=True, positive_volume=True)
    assert quality["closed"]
    assert quality["boundary_edges"] == quality["nonmanifold_edges"] == quality["degenerate_triangles"] == 0


@pytest.mark.parametrize("settings", [
    {"mode": "unknown"}, {"mode": "preserve_le", "angle_deg": 90},
    {"mode": "preserve_le", "angle_deg": True},
    {"mode": "preserve_le", "reference_azimuth_deg": "auto"},
    {"mode": "preserve_le", "misspelled": 0},
])
def test_invalid_rake_settings_fail(config, settings):
    with pytest.raises(ValueError):
        normalize_config(config | {"rake": settings})
