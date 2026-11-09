"""Deterministic wall edits share the original channel and arclength machinery."""

from copy import deepcopy

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from scipy.integrate import quad
import yaml

from development.mixed_flow.meridional import (
    EDITABLE_CONTROL_POINTS,
    MeanlineDimensions,
    MeridionalChannel,
    WallShape,
    apply_displacements,
    normalize_displacements,
    wall_scale,
)

EDITS = {"hub": {2: [0.004, -0.006], 3: [0.01, 0.005]}, "shroud": {4: [-0.006, 0.004]}}


def test_sparse_edits_are_reproducible_and_never_accumulate():
    dims = MeanlineDimensions.from_ratios()
    original = MeridionalChannel(dims, shroud=WallShape(bow=0.02))
    saved = yaml.safe_load(yaml.safe_dump(EDITS))
    first = MeridionalChannel(
        dims, shroud=WallShape(bow=0.02), control_point_displacements=EDITS
    )
    restored = MeridionalChannel(
        dims, shroud=WallShape(bow=0.02), control_point_displacements=saved
    )
    u = np.linspace(0, 1, 101)
    assert first.control_point_displacements == EDITS
    for name in ("hub", "shroud"):
        baseline, moved = getattr(original, name), getattr(first, name)
        expected = baseline.as_bspline().c.copy()
        for index, delta in EDITS[name].items():
            expected[index] += wall_scale(baseline) * np.array(delta)
        assert_array_equal(moved.as_bspline().c, expected)
        assert_array_equal(moved.point(u), getattr(restored, name).point(u))
        assert_array_equal(
            apply_displacements(moved, EDITS[name]).point(u), moved.point(u)
        )
        assert_array_equal(
            baseline.as_bspline().c, getattr(original, name).as_bspline().c
        )
        assert apply_displacements(moved, {}) is moved.baseline


def test_zero_inputs_preserve_original_polynomial_evaluation():
    dims = MeanlineDimensions.from_ratios()
    baseline = MeridionalChannel(dims, shroud=WallShape(bow=0.025))
    empty = MeridionalChannel(
        dims,
        shroud=WallShape(bow=0.025),
        control_point_displacements={"hub": {2: [0, -0.0]}, "shroud": {}},
    )
    assert empty.control_point_displacements == {}
    u = np.linspace(0, 1, 73)
    for name in ("hub", "shroud"):
        a, b = getattr(baseline, name), getattr(empty, name)
        assert apply_displacements(a, {}) is a
        assert_array_equal(a.point(u), b.point(u))
        assert_array_equal(a.point_at(u), b.point_at(u))
        assert_array_equal(a.derivative(u, 2), b.derivative(u, 2))


def test_edited_wall_scaling_and_protected_endpoint_tangents():
    small = MeridionalChannel(
        MeanlineDimensions.from_ratios(), control_point_displacements=EDITS
    )
    large = MeridionalChannel(
        MeanlineDimensions.from_ratios(1.7), control_point_displacements=EDITS
    )
    baseline = MeridionalChannel(MeanlineDimensions.from_ratios())
    assert EDITABLE_CONTROL_POINTS == (2, 3, 4)
    u = np.linspace(0, 1, 53)
    for name in ("hub", "shroud"):
        a, b, original = (
            getattr(small, name),
            getattr(large, name),
            getattr(baseline, name),
        )
        assert_allclose(b.point_at(u), 17 * a.point_at(u), rtol=1e-12, atol=1e-14)
        assert_allclose(b.curvature(u), a.curvature(u) / 17, atol=1e-12)
        assert_array_equal(a.point([0, 1]), original.point([0, 1]))
        assert_allclose(
            a.derivative([0, 1]), original.derivative([0, 1]), rtol=1e-14, atol=1e-16
        )
        assert_array_equal(
            a.as_bspline().c[[0, 1, 5, 6]], original.as_bspline().c[[0, 1, 5, 6]]
        )


def test_edited_arclength_derivatives_and_exact_spline_export():
    channel = MeridionalChannel(
        MeanlineDimensions.from_ratios(), control_point_displacements=EDITS
    )
    for wall in (channel.hub, channel.shroud):
        length = quad(lambda u: np.linalg.norm(wall.derivative(u)), 0, 1, epsabs=1e-12)[
            0
        ]
        assert wall.length == pytest.approx(length, rel=1e-10)
        spline = wall.as_bspline()
        u = np.linspace(0, 1, 37)
        assert_array_equal(wall.point(u), spline(u))
        for order in (1, 2):
            assert_array_equal(wall.derivative(u, order), spline.derivative(order)(u))
        for s in (0.1, 0.5, 0.9):
            u_s = wall.polynomial_coordinate(s)
            integral = quad(
                lambda u: np.linalg.norm(wall.derivative(u)), 0, u_s, epsabs=1e-12
            )[0]
            assert integral / length == pytest.approx(s, abs=2e-9)
            h = 2e-6
            finite = (wall.point_at(s + h) - wall.point_at(s - h)) / (2 * h)
            assert_allclose(wall.derivative_at(s), finite, rtol=2e-6, atol=1e-8)
        # Exported objects cannot mutate the current geometry.
        before = wall.point(0.4).copy()
        spline.c[3] += 1
        assert_array_equal(wall.point(0.4), before)


@pytest.mark.parametrize(
    "invalid",
    [
        [],
        1,
        {"outer": {}},
        {"hub": None},
        {"hub": []},
        {"hub": {True: [0, 0]}},
        {"hub": {2.0: [0, 0]}},
        {"hub": {"2.0": [0, 0]}},
        {"hub": {2: [0, 0], "2": [0, 0]}},
        *[{"hub": {index: [0, 0]}} for index in (-1, 0, 1, 5, 6, 7)],
        *[
            {"shroud": {3: vector}}
            for vector in (
                1,
                "12",
                None,
                [0],
                [0, 1, 2],
                [[0], [1]],
                [True, 0],
                [0, np.nan],
                [np.inf, 0],
                [0, "1"],
                [0, 1j],
                np.array(1),
                [10**400, 0],
            )
        ],
    ],
)
def test_malformed_or_protected_edits_are_rejected(invalid):
    with pytest.raises(ValueError):
        normalize_displacements(invalid)
    with pytest.raises(ValueError):
        MeridionalChannel(
            MeanlineDimensions.from_ratios(), control_point_displacements=invalid
        )


def test_normalization_does_not_mutate_inputs_and_rejects_folded_channel():
    entries = {"shroud": {"4": [0, 0]}, "hub": {"3": [0.01, -0.02]}}
    before = deepcopy(entries)
    assert normalize_displacements(entries) == {"hub": {3: [0.01, -0.02]}}
    assert entries == before
    with pytest.raises(ValueError, match="folds|radius|tangent"):
        MeridionalChannel(
            MeanlineDimensions.from_ratios(),
            control_point_displacements={"hub": {3: [0, 2]}},
        )
