"""Shared smoothing contracts and interpolation independent of loss models."""

import numpy as np
import pytest
from numpy.polynomial import Polynomial
from scipy.interpolate import PchipInterpolator

from turbodash import math


@pytest.mark.parametrize("name,method,expected", [
    ("smooth_maximum", "boltzmann", [.3379948962255225, .5, 1.1760627057808204]),
    ("smooth_maximum", "logsumexp", [.4927751664869444, .6732867951399863, 1.2067392732520519]),
    ("smooth_maximum", "p-norm", [.4061086369737862, .5946035575013605, 1.2011701622873112]),
    ("smooth_minimum", "boltzmann", [.2620051037744775, .5, .32393729421917916]),
    ("smooth_minimum", "logsumexp", [.10722483351305559, .32671320486001365, .29326072674794795]),
    ("smooth_minimum", "p-norm", [.19699162420218094, .42044820762685725, .2997077444168902]),
])
def test_existing_extrema_methods_keep_reference_values(name, method, expected):
    result = getattr(math, name)([.2, .5, 1.2], [.4, .5, .3], method, 4)
    np.testing.assert_allclose(result, expected, rtol=2e-15, atol=0)


@pytest.mark.parametrize("function", [math.smooth_maximum, math.smooth_minimum])
def test_extrema_keep_boltzmann_default_and_require_explicit_compact_width(function):
    np.testing.assert_array_equal(function([0., 1.], .4), function([0., 1.], .4, "boltzmann", 10))
    with pytest.raises(ValueError, match="half_width is required"):
        function(0., 1., method="compact")
    with pytest.raises(ValueError, match="only supported"):
        function(0., 1., half_width=.1)
    with pytest.raises(ValueError, match="Unsupported method"):
        function(0., 1., method="unknown")


@pytest.mark.parametrize("function,exact,sign", [
    (math.smooth_maximum, np.maximum, 1),
    (math.smooth_minimum, np.minimum, -1),
])
def test_compact_extrema_bounds_exact_outer_branches_and_broadcasting(function, exact, sign):
    x = np.linspace(-1., 1., 2001)[:, None]
    y = np.array([-.3, 0., .4])
    width = np.array([0., .05, .2])
    with np.errstate(all="raise"):
        result = function(x, y, method="compact", half_width=width)
    assert result.shape == (len(x), len(y))
    error = sign * (result - exact(x, y))
    assert np.all(error >= -2e-16)
    assert np.all(error <= 3 * width / 16 + 2e-16)
    outer = np.abs(x - y) >= width
    np.testing.assert_array_equal(result[outer], exact(x, y)[outer])
    np.testing.assert_allclose(result, function(y, x, method="compact", half_width=width), atol=2e-16)


def test_positive_part_is_monotone_bounded_and_accepts_lists():
    x = np.linspace(-1., 1., 2001)
    result = math.positive_part(x.tolist(), .1)
    error = result - np.maximum(x, 0.)
    assert np.all(error >= -1e-16)
    assert np.max(error) == pytest.approx(3 * .1 / 16)
    assert np.all(np.diff(result) >= -1e-16)
    np.testing.assert_array_equal(math.positive_part(x, 0.), np.maximum(x, 0.))


@pytest.mark.parametrize("function,expected_slopes", [
    (lambda x: math.positive_part(x, .1), (0., 1.)),
    (lambda x: math.smooth_maximum(x, 0., method="compact", half_width=.1), (0., 1.)),
    (lambda x: math.smooth_minimum(x, 0., method="compact", half_width=.1), (1., 0.)),
])
def test_compact_transitions_match_outer_derivatives_through_second_order(function, expected_slopes):
    step = 1e-7
    for endpoint, expected in zip((-.1, .1), expected_slopes):
        # Differentiate the analytic first derivative independently on each side.
        points = endpoint + step * np.arange(-2, 3)
        slopes = np.imag(function(points.astype(complex) + 1e-25j)) / 1e-25
        np.testing.assert_allclose(slopes, expected, atol=4e-11, rtol=0)
        np.testing.assert_allclose(np.diff(slopes) / step, 0., atol=1e-4, rtol=0)


def test_smootherstep_and_branch_blend_are_flat_outside_transition():
    np.testing.assert_array_equal(math.sigmoid_smootherstep([-1., 0., .5, 1., 2.]), [0., 0., .5, 1., 1.])
    x = np.array([-1., -.5, 0., .5, 1.])
    np.testing.assert_array_equal(math.blend_branches(x, 2., 4., 0., .5), [2., 2., 3., 4., 4.])
    for endpoint in (0., 1.):
        points = endpoint + 1e-7 * np.arange(-2, 3)
        slopes = np.imag(math.sigmoid_smootherstep(points.astype(complex) + 1e-25j)) / 1e-25
        np.testing.assert_allclose(slopes, 0., atol=2e-12)
        np.testing.assert_allclose(np.diff(slopes) / 1e-7, 0., atol=1e-5)


def test_quintic_hermite_reproduces_degree_five_polynomials_and_clamps_values():
    # Independent polynomial evaluation checks the interval scaling of derivatives.
    polynomial = Polynomial([.3, -.2, .7, .1, -.4, .05])
    lower, upper = -.7, 1.3
    x = np.linspace(lower - .2, upper + .2, 101)
    first, second = polynomial.deriv(), polynomial.deriv(2)
    result = math.quintic_hermite(
        x, lower, upper, polynomial(lower), polynomial(upper),
        first(lower), first(upper), second(lower), second(upper),
    )
    np.testing.assert_allclose(result, polynomial(np.clip(x, lower, upper)), rtol=2e-13, atol=2e-15)


@pytest.mark.parametrize("values", [[0., .4, .6, 1.], [1., .7, .2, 0.], [0., 1., .5, .8]])
def test_cubic_table_matches_scipy_shape_preserving_interpolator(values):
    knots = np.array([-1., -.2, .5, 1.3])
    x = np.linspace(knots[0], knots[-1], 251)
    result = math.monotone_cubic_table(x, knots, values, flat_left=False, flat_right=False)
    np.testing.assert_allclose(result, PchipInterpolator(knots, values)(x), atol=5e-16, rtol=2e-14)


def test_cubic_table_default_bounds_knots_and_flat_constant_extension():
    knots = np.array([-1., -.2, .5, 1.3])
    values = np.array([0., .4, .6, 1.])
    np.testing.assert_array_equal(math.monotone_cubic_table(knots, knots, values), values)
    np.testing.assert_array_equal(math.monotone_cubic_table([-2., 2.], knots, values), [0., 1.])
    x = np.linspace(-2., 2., 1001)
    result = math.monotone_cubic_table(x, knots, values)
    assert np.all(np.diff(result) >= -1e-15)
    assert np.all((result >= values[0]) & (result <= values[-1]))
    for endpoint in knots[[0, -1]]:
        slope = (math.monotone_cubic_table(endpoint + 1e-7, knots, values)
                 - math.monotone_cubic_table(endpoint - 1e-7, knots, values)) / 2e-7
        assert slope == pytest.approx(0., abs=5e-7)
