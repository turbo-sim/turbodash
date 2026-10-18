"""Independent velocity-vector checks for the high-reaction loading repair."""
import numpy as np
from turbodash.loss_models import _blade_loading as loading


def slope(function, value):
    return np.imag(function(np.asarray(value, dtype=complex)+1e-25j))/1e-25


def test_loading_retains_physical_turning_as_throughflow_vanishes():
    vm = np.array([1., .3, .1, .03, .01, .001, .0001])
    # Finite inlet/outlet tangential speeds, equal vm, genuine finite turning.
    wt_in, wt_out = -1., -2.
    bi, bo = np.rad2deg(np.arctan2(wt_in, vm)), np.rad2deg(np.arctan2(wt_out, vm))
    physical_secant = np.sqrt(vm**2+1.5**2)/vm
    start, limit = 1/np.cos(np.deg2rad([80.,87.5]))
    t = np.maximum(physical_secant-start, 0.)/(limit-start)
    secant = np.where(physical_secant <= start, physical_secant,
                       limit-(limit-start)/(1+t+t*t))
    expected = 4*(wt_in-wt_out)**2/(vm**2+wt_out**2)*secant
    actual = loading.loading(bi, bo)
    np.testing.assert_allclose(actual, expected, rtol=1e-10)
    assert actual[-1] > 22.9
    np.testing.assert_allclose(loading.physical_cosine_ratio(bo, bi),
                               np.sqrt(vm**2+wt_in**2)/np.sqrt(vm**2+wt_out**2), rtol=1e-11)


def test_mean_secant_c2_join_scale_invariance_and_zero_meridional_limit():
    start, limit = 1/np.cos(np.deg2rad([80.,87.5]))
    function = lambda sec: loading.mean_secant(1., np.sqrt(sec*sec-1))
    h = 1e-5
    positions = start+np.array([-h, 0., h])
    np.testing.assert_allclose(function(positions), positions, atol=2e-14)
    np.testing.assert_allclose(slope(function, positions), 1., atol=2e-11)
    np.testing.assert_allclose(np.diff(slope(function, positions))/h, 0., atol=3e-6)
    for scale in (1e-200, 1., 1e200):
        np.testing.assert_allclose(loading.mean_secant(scale, 20.*scale), function(np.sqrt(401.)), rtol=3e-15)
    np.testing.assert_allclose(loading.mean_secant(0., 1.), limit, rtol=1e-14)
    assert np.isnan(loading.mean_secant(0., 0.))


def test_no_individual_angle_saturation_before_mean_join():
    # Both angles exceed the old tangent join, but their mean tangent cancels.
    beta = np.array([80., 85., 89., 89.9])
    np.testing.assert_allclose(loading.loading(-beta, beta), 16*np.sin(np.deg2rad(beta))**2, rtol=3e-14)
    # A grazing outlet with a moderate inlet has a finite unsuppressed turning.
    for angle in (85., 88., 89.5):
        h = 1e-4
        fn = lambda outlet: loading.loading(-40., outlet)
        finite = (fn(angle-2*h)-8*fn(angle-h)+8*fn(angle+h)-fn(angle+2*h))/(12*h)
        np.testing.assert_allclose(slope(fn, angle), finite, rtol=2e-7, atol=1e-8)
