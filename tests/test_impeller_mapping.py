"""Independent geometric limits and accuracy checks for the impeller mapper."""

from types import SimpleNamespace
from dataclasses import asdict

import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.integrate import quad

from turbodash.geometry.impeller.mapping import StreamSurfaceMap
from turbodash.geometry.impeller.meridional import (
    BladeEdges,
    MeanlineDimensions,
    MeridionalChannel,
    WallShape,
)


class LinearChannel:
    """An independent affine channel, covering axial and radial exact limits."""

    def __init__(self, origin, streamwise, span):
        self.origin = np.asarray(origin, dtype=float)
        self.streamwise = np.asarray(streamwise, dtype=float)
        self.span = np.asarray(span, dtype=float)
        self.dimensions = SimpleNamespace(outlet_radius=1.0)

    def point(self, u, v):
        u, v = np.broadcast_arrays(u, v)
        return self.origin + u[..., None] * self.streamwise + v[..., None] * self.span

    def derivatives(self, u, v):
        shape = np.broadcast_arrays(u, v)[0].shape + (2,)
        return np.broadcast_to(self.streamwise, shape), np.broadcast_to(self.span, shape)


def curved_mapping(scale=1.0, integration_points=2001):
    dimensions = MeanlineDimensions.from_ratios(0.1 * scale)
    channel = MeridionalChannel(
        dimensions,
        hub=WallShape(bow=-0.012),
        shroud=WallShape(bow=0.018),
    )
    edges = BladeEdges(0.08, 0.17, 0.91, 0.98, 0.027, -0.018)
    return StreamSurfaceMap(channel, edges, integration_points=integration_points)


def test_cylinder_exact_distances_and_inverse_with_extensions():
    mapping = StreamSurfaceMap(LinearChannel((0, 0.07), (0.12, 0), (0, 0.03)))
    section = mapping.section(0.4)
    radius = 0.082
    u = np.array([[-0.3, 0, 0.12345], [0.7, 1, 1.4]])
    assert section.length_m == pytest.approx(0.12, rel=1e-13)
    assert section.prime_length == pytest.approx(0.12 / radius, rel=1e-13)
    assert_allclose(section.physical_distance(u), 0.12 * u, atol=1e-14)
    assert_allclose(section.prime(u), 0.12 * u / radius, atol=1e-13)
    assert_allclose(section.u_from_prime(section.prime(u)), u, atol=1e-13)
    assert_allclose(section.from_prime(section.prime(u)), section.point(u), atol=1e-14)
    assert section.point(0.5).shape == (2,)
    assert section.prime(0.5).shape == ()
    assert mapping.section(0.4) is section


@pytest.mark.parametrize("radial_direction", [1, -1])
def test_radial_plane_exact_logarithm_with_extensions(radial_direction):
    origin = 0.09 if radial_direction == 1 else 0.21
    radial_change = radial_direction * 0.12
    mapping = StreamSurfaceMap(LinearChannel((0.02, origin), (0, radial_change), (-0.01, 0)))
    section = mapping.section(0.7)
    u = np.array([-0.25, 0, 0.11375, 0.4567, 0.91234, 1, 1.25])
    radius = origin + radial_change * u
    expected = radial_direction * np.log(radius / origin)
    assert section.length_m == pytest.approx(0.12, rel=1e-13)
    assert section.prime_length == pytest.approx(expected[-2], rel=1e-12)
    assert_allclose(section.prime(u), expected, rtol=1e-10, atol=2e-11)
    assert_allclose(section.u_from_prime(expected), u, rtol=1e-10, atol=2e-11)
    assert_allclose(section.from_prime(expected)[..., 1], radius, atol=2e-12)
    assert_allclose(section.physical_distance(u), 0.12 * u, atol=1e-14)
    zero_radius_u = -origin / radial_change
    for operation in (section.point, section.prime, section.physical_distance, section.derivative):
        with pytest.raises(ValueError, match="radius"):
            operation(zero_radius_u + (-0.01 if radial_direction == 1 else 0.01))


def test_bowed_footprint_derivatives_and_jacobian_chain_rule():
    mapping = curved_mapping()
    u = np.linspace(0.08, 0.92, 7)[:, None]
    v = np.linspace(0.1, 0.9, 5)[None, :]
    step = 2e-6
    du, dv = mapping.derivatives(u, v)
    numerical_du = (mapping.point(u + step, v) - mapping.point(u - step, v)) / (2 * step)
    numerical_dv = (mapping.point(u, v + step) - mapping.point(u, v - step)) / (2 * step)
    assert_allclose(du, numerical_du, rtol=2e-6, atol=2e-8)
    assert_allclose(dv, numerical_dv, rtol=2e-6, atol=2e-8)
    extent = mapping.edges.trailing(v) - mapping.edges.leading(v)
    channel_u = mapping.edges.leading(v) + u * extent
    determinant = du[..., 0] * dv[..., 1] - du[..., 1] * dv[..., 0]
    assert_allclose(determinant, extent * mapping.channel.jacobian(channel_u, v), atol=1e-18)
    assert np.all(determinant > 0)


def test_curved_section_integrals_against_independent_adaptive_quadrature():
    mapping = curved_mapping()
    for v in (0.0, 0.43, 1.0):
        section = mapping.section(v)
        for u in (0.137, 0.5143, 1.0):
            physical = quad(
                lambda u: np.linalg.norm(mapping.derivatives(u, v)[0]),
                0, u, epsabs=1e-11, epsrel=1e-10,
            )[0]
            conformal = quad(
                lambda u: np.linalg.norm(mapping.derivatives(u, v)[0]) / mapping.point(u, v)[1],
                0, u, epsabs=1e-10, epsrel=1e-9,
            )[0]
            assert section.physical_distance(u) == pytest.approx(physical, rel=1e-8, abs=1e-10)
            assert section.prime(u) == pytest.approx(conformal, rel=1e-8, abs=1e-9)
            assert section.u_from_prime(conformal) == pytest.approx(u, abs=2e-9)
        assert section.prime(0) == 0
        assert_allclose(section.point([0, 1]), mapping.point([0, 1], v), atol=1e-15)


def test_curved_extension_roundtrip_and_tangent_continuity():
    section = curved_mapping().section(0.36)
    lower, upper = section.channel_u_bounds
    u = np.array([lower - 0.04, -0.001, 0, 0.32167, 1, 1.001, upper + 0.04])
    assert_allclose(section.u_from_prime(section.prime(u)), u, atol=2e-10)
    for end, sign in ((lower, -1), (upper, 1)):
        delta = sign * 1e-7
        tangent = section.derivative(end)
        assert_allclose((section.point(end + delta) - section.point(end)) / delta, tangent, atol=1e-10)
        assert_allclose(section.derivative(end + delta), tangent, atol=1e-15)
        slope = np.linalg.norm(tangent) / section.point(end)[1]
        assert (section.prime(end + delta) - section.prime(end)) / delta == pytest.approx(slope, rel=2e-7)


def test_outside_blade_footprint_follows_true_channel_before_tangent_extension():
    mapping = curved_mapping()
    section = mapping.section(0.31)
    lower, upper = section.channel_u_bounds
    assert lower < 0 < 1 < upper
    assert section.channel_prime_bounds[0] < 0
    assert section.channel_prime_bounds[1] > section.prime_length
    local_u = np.array([lower, lower / 2, 0, 1, (1 + upper) / 2, upper])
    channel_u = mapping.edges.leading(section.v) + local_u * (
        mapping.edges.trailing(section.v) - mapping.edges.leading(section.v)
    )
    assert_allclose(section.point(local_u), mapping.channel.point(channel_u, section.v), atol=1e-15)
    assert_allclose(section.from_prime(section.channel_prime_bounds), mapping.channel.point([0, 1], section.v), atol=1e-14)


def test_uniform_scaling_preserves_conformal_coordinates():
    small, large = curved_mapping(), curved_mapping(17)
    u = np.linspace(-0.03, 1.04, 23)
    for v in (0.0, 0.41, 1.0):
        a, b = small.section(v), large.section(v)
        assert_allclose(b.point(u), 17 * a.point(u), rtol=2e-12, atol=1e-12)
        assert_allclose(b.physical_distance(u), 17 * a.physical_distance(u), rtol=2e-12, atol=1e-12)
        assert_allclose(b.prime(u), a.prime(u), rtol=2e-12, atol=1e-12)
        assert b.length_m == pytest.approx(17 * a.length_m, rel=2e-12)
        assert b.prime_length == pytest.approx(a.prime_length, rel=2e-12)


def test_legacy_imports_share_the_canonical_implementation():
    from development.mixed_flow.meridional import MeridionalChannel as OldChannel
    from development.mixed_flow.meridional.curves import _bezier as old_bezier
    from turbodash.geometry.impeller.meridional.curves import _bezier

    assert OldChannel is MeridionalChannel
    assert old_bezier is _bezier


def test_nearly_coincident_edge_and_integration_knots_do_not_change_geometry():
    channel = MeridionalChannel(MeanlineDimensions.from_ratios())
    a = StreamSurfaceMap(channel, BladeEdges(0.15, 0.15, 0.9, 0.9)).section(0.37)
    b = StreamSurfaceMap(channel, BladeEdges(0.15000000000000002, 0.15000000000000002, 0.9, 0.9)).section(0.37)
    assert a.length_m == pytest.approx(b.length_m, rel=1e-12)
    assert a.prime_length == pytest.approx(b.prime_length, rel=1e-12)
    assert a.prime(0) == b.prime(0) == 0
    assert_allclose(a.u_from_prime([0, a.prime_length]), [0, 1], atol=0)


@pytest.mark.parametrize("points", [True, 4, 8.1])
def test_reject_invalid_integration_resolution(points):
    with pytest.raises(ValueError, match="integration_points"):
        StreamSurfaceMap(LinearChannel((0, 1), (1, 0), (0, 1)), integration_points=points)


def test_reject_invalid_coordinates_and_sections():
    mapping = curved_mapping()
    for invalid in (-0.1, 1.1, np.nan, np.inf):
        with pytest.raises(ValueError):
            mapping.point(invalid, 0.5)
        with pytest.raises(ValueError):
            mapping.section(invalid)
    with pytest.raises(ValueError, match="scalar"):
        mapping.section([0.3, 0.7])
    section = mapping.section(0.5)
    with pytest.raises(ValueError, match="finite"):
        section.u_from_prime(np.nan)
    bad = StreamSurfaceMap(LinearChannel((0, -1), (1, 0), (0, 0)))
    with pytest.raises(ValueError, match="radii"):
        bad.section(0.5)
    collapsed = StreamSurfaceMap(LinearChannel((0, 1), (0, 0), (0, 1)))
    with pytest.raises(ValueError, match="tangent"):
        collapsed.section(0.5)


def test_camber_integration_matches_independent_angle_integral():
    from turbodash.geometry.blade_parametrization.geometry import turning_angle
    from turbodash.geometry.impeller.config import normalize_config, section_parameters
    from turbodash.geometry.impeller.geometry import _build_section

    controls = [
        {
            "v": v,
            "camberline": {
                "inlet_metal_angle_deg": -57.0,
                "outlet_metal_angle_deg": -21.0,
                "turning_location": 0.31,
                "turning_concentration": 2.8,
            },
            "thickness": {
                "maximum_thickness_m": 0.001,
                "leading_edge_thickness_m": 0.0004,
                "trailing_edge_thickness_m": 0.0003,
            },
            "stacking_offset_deg": 7.5,
        }
        for v in (0, 1)
    ]
    config = normalize_config({
        "dimensions": asdict(MeanlineDimensions.from_ratios()),
        "blade_count": 12,
        "sections": controls,
    })
    mapping = curved_mapping()
    control = config["sections"][0]
    u = np.array([0, 0.15, 0.15000000000000002, 0.31, 0.54, 0.83, 1])
    section = _build_section(mapping, control, u, 2001, 12, False)
    camber_parameters = section_parameters(control, section.mapping.length_m)["camberline"]
    expected = np.deg2rad(7.5) + section.mapping.prime_length * np.array([
        quad(lambda u: np.tan(turning_angle(np.asarray(u), camber_parameters)),
             0, stop, epsabs=1e-12, epsrel=1e-12)[0]
        for stop in u
    ])
    assert_allclose(section.theta, expected, rtol=2e-9, atol=2e-10)
    assert_allclose(section.prime, section.mapping.prime_length * u, atol=1e-15)
    for angle in (-52.0, 0.0, 33.0):
        control["camberline"]["inlet_metal_angle_deg"] = angle
        control["camberline"]["outlet_metal_angle_deg"] = angle
        constant = _build_section(mapping, control, u, 2001, 12, False)
        assert_allclose(constant.theta, np.deg2rad(7.5) + constant.prime * np.tan(np.deg2rad(angle)), atol=2e-13)
