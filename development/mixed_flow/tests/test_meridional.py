"""Geometry invariants independent of the chosen polynomial construction."""

from dataclasses import replace

import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.integrate import quad

from development.mixed_flow.meridional import (
    BladeEdges,
    BladeMeridional,
    MeanlineDimensions,
    MeridionalChannel,
    WallShape,
)


@pytest.mark.parametrize("inlet,outlet", [(0, 90), (15, 65), (-20, 80)])
def test_station_geometry_area_and_inclination(inlet, outlet):
    dims = MeanlineDimensions.from_ratios(
        inlet_flow_angle_deg=inlet, outlet_flow_angle_deg=outlet
    )
    channel = MeridionalChannel(dims)
    for index, angle, expected_area in (
        (0, inlet, dims.inlet_area),
        (1, outlet, dims.outlet_area),
    ):
        points = channel.point(index, [0, 1])
        tangent = np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))])
        assert np.dot(points[1] - points[0], tangent) == pytest.approx(0, abs=1e-15)
        for wall in (channel.hub, channel.shroud):
            direction = wall.derivative_at(index)
            assert_allclose(direction / np.linalg.norm(direction), tangent, atol=2e-14)
        # Independent surface-of-revolution integral along the station.
        gap = np.linalg.norm(points[1] - points[0])
        integral = quad(
            lambda eta: 2
            * np.pi
            * ((1 - eta) * points[0, 1] + eta * points[1, 1])
            * gap,
            0,
            1,
        )[0]
        assert integral == pytest.approx(expected_area, rel=1e-13)
        d = channel.diagnostics()
        i = 0 if index == 0 else -1
        assert d["area"][i] == pytest.approx(integral, rel=1e-13)
        assert d["projected_area"][i] == pytest.approx(integral, rel=1e-13)
    if inlet == 0:
        assert dims.inlet_area == pytest.approx(np.pi * (0.065**2 - 0.025**2))
    if outlet == 90:
        assert_allclose(channel.point(1, [0, 1])[:, 1], 0.1)
        assert dims.outlet_area == pytest.approx(2 * np.pi * 0.1 * 0.008)


def test_meanline_area_adapter_and_default_length():
    original = MeanlineDimensions.from_ratios(
        inlet_flow_angle_deg=22, outlet_flow_angle_deg=68
    )
    recovered = MeanlineDimensions.from_flow_areas(
        inlet_area=original.inlet_area,
        outlet_area=original.outlet_area,
        outlet_radius=original.outlet_radius,
        inlet_hub_to_shroud_ratio=original.inlet_hub_radius
        / original.inlet_shroud_radius,
        inlet_flow_angle_deg=22,
        outlet_flow_angle_deg=68,
    )
    assert_allclose(recovered.inlet_points, original.inlet_points)
    assert_allclose(recovered.outlet_points, original.outlet_points)
    assert recovered.axial_length == pytest.approx(0.6 * original.outlet_radius)


def test_independent_wall_angles_bow_and_exact_bspline():
    dims = MeanlineDimensions.from_ratios()
    reference = MeridionalChannel(dims)
    shape = WallShape(inlet_angle_deg=12, outlet_angle_deg=84, bow=0.03)
    modified = MeridionalChannel(dims, shroud=shape)
    s = np.linspace(0, 1, 99)
    assert_allclose(modified.hub.point_at(s), reference.hub.point_at(s))
    assert not np.allclose(modified.shroud.point_at(s), reference.shroud.point_at(s))
    wall = modified.shroud
    for u, angle in ((0, 12), (1, 84)):
        tangent = wall.derivative(u)
        assert np.degrees(np.arctan2(tangent[1], tangent[0])) == pytest.approx(angle)
        assert wall.curvature(u) == pytest.approx(0, abs=1e-11)
    spline = wall.as_bspline()
    assert_allclose(spline(s), wall.point(s), atol=1e-15)
    for order in (1, 2):
        assert_allclose(
            spline.derivative(order)(s), wall.derivative(s, order), atol=2e-14
        )
    no_bow = MeridionalChannel(dims, shroud=replace(shape, bow=0)).shroud
    displacement = wall.point(0.5) - no_bow.point(0.5)
    chord = wall.end - wall.start
    assert np.linalg.norm(displacement) == pytest.approx(0.03 * np.linalg.norm(chord))
    assert np.dot(displacement, chord) == pytest.approx(0, abs=1e-16)


def test_arclength_inversion_and_analytic_derivatives():
    channel = MeridionalChannel(
        MeanlineDimensions.from_ratios(), shroud=WallShape(bow=0.025)
    )
    for wall in (channel.hub, channel.shroud):
        length = quad(lambda u: np.linalg.norm(wall.derivative(u)), 0, 1, epsabs=1e-12)[
            0
        ]
        assert wall.length == pytest.approx(length, rel=1e-10)
        for s in (0.1, 0.5, 0.9):
            u = wall.polynomial_coordinate(s)
            length_to_s = quad(
                lambda v: np.linalg.norm(wall.derivative(v)), 0, u, epsabs=1e-12
            )[0]
            assert length_to_s / length == pytest.approx(s, abs=2e-9)
            h = 2e-6
            numerical = (wall.point_at(s + h) - wall.point_at(s - h)) / (2 * h)
            assert_allclose(wall.derivative_at(s), numerical, rtol=2e-6, atol=1e-8)
        u = np.linspace(0.05, 0.95, 17)
        h = 1e-5
        numerical_second = (wall.derivative(u + h) - wall.derivative(u - h)) / (2 * h)
        assert_allclose(wall.derivative(u, 2), numerical_second, rtol=1e-6, atol=1e-8)


def test_blade_edges_splitter_and_channel_independence():
    channel = MeridionalChannel(MeanlineDimensions.from_ratios())
    eta = np.linspace(0, 1, 51)
    before = channel.point(0.4, eta).copy()
    edges = BladeEdges(0.2, 0.05, 0.95, 1, 0.03, -0.01)
    blade = BladeMeridional(channel, edges)
    splitter = blade.splitter(0.25)
    assert splitter.channel is channel
    assert_allclose(splitter.leading_edge(eta), blade.point(0.25, eta), atol=1e-15)
    assert_allclose(splitter.trailing_edge(eta), blade.trailing_edge(eta))
    assert_allclose(channel.point(0.4, eta), before)
    for span, leading, trailing, wall in (
        (0, 0.2, 0.95, channel.hub),
        (1, 0.05, 1, channel.shroud),
    ):
        assert_allclose(blade.leading_edge(span), wall.point_at(leading))
        assert_allclose(blade.trailing_edge(span), wall.point_at(trailing))
    grid = blade.point(np.linspace(0, 1, 7)[:, None], eta[None, :])
    assert grid.shape == (7, 51, 2)
    assert blade.point(0.5, 0.5).shape == (2,)


def test_geometric_scaling():
    small = MeridionalChannel(MeanlineDimensions.from_ratios(0.1))
    large = MeridionalChannel(MeanlineDimensions.from_ratios(1.7))
    s = np.linspace(0, 1, 51)
    assert_allclose(
        large.point(s, 0.3), 17 * small.point(s, 0.3), rtol=1e-12, atol=1e-13
    )
    assert_allclose(
        large.diagnostics()["area"], 17**2 * small.diagnostics()["area"], rtol=1e-12
    )
    assert_allclose(large.hub.curvature(s), small.hub.curvature(s) / 17, atol=1e-12)


@pytest.mark.parametrize(
    "edges",
    [
        {"leading_hub": -0.1},
        {"leading_hub": 0.5, "trailing_hub": 0.4},
        {"leading_hub": 0.3, "leading_shroud": 0.3, "leading_bow": -0.4},
        {"trailing_bow": 0.1},
        {
            "leading_hub": 0.1,
            "leading_shroud": 0.1,
            "leading_bow": 0.6,
            "trailing_hub": 0.9,
            "trailing_shroud": 0.9,
            "trailing_bow": -0.3,
        },
    ],
)
def test_invalid_edges_including_interior_extrema(edges):
    with pytest.raises(ValueError):
        BladeEdges(**edges)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"inlet_hub_radius": 0},
        {"inlet_shroud_radius": 0.2},
        {"outlet_width": -1},
        {"inlet_flow_angle_deg": 90},
        {"outlet_flow_angle_deg": 0},
        {"axial_length": np.nan},
    ],
)
def test_invalid_dimensions(kwargs):
    with pytest.raises(ValueError):
        replace(MeanlineDimensions.from_ratios(), **kwargs)


def test_invalid_mapping_coordinates_and_near_degenerate_handles():
    dims = MeanlineDimensions.from_ratios()
    with pytest.raises(ValueError, match="folds|radius|tangent"):
        MeridionalChannel(dims, shroud=WallShape(bow=-0.5))
    with pytest.raises(ValueError):
        WallShape(inlet_tangent_fraction=0.009)
    lower_bound = MeridionalChannel(
        dims, hub=WallShape(inlet_tangent_fraction=0.01, outlet_tangent_fraction=0.01)
    )
    assert np.isfinite(lower_bound.point([0, 1e-10, 0.5, 1], 0.5)).all()
    for value in (-0.001, 1.001, np.nan, np.inf):
        with pytest.raises(ValueError):
            lower_bound.point(value, 0.5)
        with pytest.raises(ValueError):
            BladeEdges().leading(value)
