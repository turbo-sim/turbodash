"""Independent checks of prescribed-wrap monotone turning and its primitive."""

import numpy as np
import pytest
from scipy.integrate import quad

from turbodash.geometry.blade_parametrization.turning import constrained_turning


@pytest.mark.parametrize("incoming,outgoing,fraction", [
    (-55, -27, .04), (-55, -27, .4), (-55, -27, .96),
    (65, -25, .13), (65, -25, .87), (-35, 45, .5),
])
def test_analytic_primitive_matches_independent_quadrature(incoming, outgoing, fraction):
    a, b = np.tan(np.deg2rad([incoming, outgoing]))
    mean = a + (b - a) * fraction
    u = np.array([0, .17, .43, .81, 1])
    _, integral, metadata = constrained_turning(u, incoming, outgoing, mean, concentration=1.3)
    for coordinate, actual in zip(u, integral):
        expected = quad(lambda x: np.tan(constrained_turning(x, incoming, outgoing, mean, concentration=1.3)[0]),
                        0, coordinate, epsabs=1e-12, epsrel=1e-12)[0]
        assert actual == pytest.approx(expected, abs=2e-12)
    assert integral[-1] == mean
    assert metadata["mean_slope"] == mean


@pytest.mark.parametrize("fraction", [.001, .1, .5, .9, .999])
@pytest.mark.parametrize("incoming,outgoing", [(-55, -27), (45, -40)])
def test_turning_remains_monotone_and_preserves_angles(fraction, incoming, outgoing):
    a, b = np.tan(np.deg2rad([incoming, outgoing]))
    mean = a + (b - a) * fraction
    u = np.linspace(0, 1, 2001)
    beta, integral, metadata = constrained_turning(u, incoming, outgoing, mean)
    assert np.all(np.diff(beta) * np.sign(outgoing - incoming) >= -1e-14)
    np.testing.assert_array_equal(beta[[0, -1]], np.deg2rad([incoming, outgoing]))
    assert integral[0] == 0 and integral[-1] == mean
    assert metadata["shape_a"] >= 2 - 1e-12
    assert metadata["shape_b"] >= 2 - 1e-12


def test_primitive_derivative_matches_local_camber_slope():
    incoming, outgoing, mean = -52, -20, -.7
    u = np.linspace(.03, .97, 71)
    h = 1e-6
    beta = constrained_turning(u, incoming, outgoing, mean)[0]
    derivative = (constrained_turning(u + h, incoming, outgoing, mean)[1]
                  - constrained_turning(u - h, incoming, outgoing, mean)[1]) / (2 * h)
    np.testing.assert_allclose(derivative, np.tan(beta), atol=2e-9, rtol=0)


def test_concentration_changes_shape_but_not_total_wrap():
    u = np.linspace(0, 1, 501)
    broad = constrained_turning(u, -55, -27, -.9, concentration=1)
    narrow = constrained_turning(u, -55, -27, -.9, concentration=4)
    assert np.max(abs(broad[0] - narrow[0])) > .02
    assert broad[1][-1] == narrow[1][-1] == -.9
    np.testing.assert_array_equal(broad[0][[0, -1]], narrow[0][[0, -1]])


def test_constant_angle_has_constant_primitive_and_rejects_other_mean():
    u = np.array([[0, .2], [.7, 1]])
    slope = np.tan(np.deg2rad(-32))
    beta, integral, metadata = constrained_turning(u, -32, -32, slope)
    np.testing.assert_array_equal(beta, np.full_like(u, np.deg2rad(-32)))
    np.testing.assert_array_equal(integral, slope * u)
    assert metadata["constant_angle"]
    with pytest.raises(ValueError, match="constant mean"):
        constrained_turning(u, -32, -32, slope + .01)


@pytest.mark.parametrize("fraction", [1e-7, 1 - 1e-7])
def test_extreme_feasible_mean_stays_finite_without_iteration(fraction):
    a, b = np.tan(np.deg2rad([-55, -27]))
    u = np.unique(np.r_[np.linspace(0, 1, 101), np.geomspace(1e-12, .01, 100),
                        1 - np.geomspace(1e-12, .01, 100)])
    beta, integral, _ = constrained_turning(u, -55, -27, a + (b - a) * fraction)
    assert np.all(np.isfinite(beta)) and np.all(np.isfinite(integral))
    assert np.all(np.diff(beta) >= -1e-13)
    assert np.all(np.diff(integral) <= 1e-13)  # Both endpoint slopes are negative.


@pytest.mark.parametrize("arguments", [
    ([0, 1], -90, -27, -.8), ([0, 1], -55, 90, -.8),
    ([0, 1], -55, -27, -2), ([0, 1], -55, -27, 0),
    ([0, np.nan], -55, -27, -.8), ([-.1, 1], -55, -27, -.8),
    ([0, 1.1], -55, -27, -.8), ([0, 1], -55, -27, np.inf),
    ([0, 1], True, -27, -.8),
])
def test_invalid_domains_are_rejected(arguments):
    with pytest.raises(ValueError):
        constrained_turning(*arguments)


def test_boundary_mean_and_invalid_concentration_are_rejected():
    for mean in np.tan(np.deg2rad([-55, -27])):
        with pytest.raises(ValueError, match="strictly between"):
            constrained_turning([0, 1], -55, -27, mean)
    for concentration in (0, .9, np.inf, True):
        with pytest.raises(ValueError):
            constrained_turning([0, 1], -55, -27, -.8, concentration=concentration)
