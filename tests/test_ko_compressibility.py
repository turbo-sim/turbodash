"""Compressibility contracts: bounded factors, smooth joins and zero-Mach limits."""
import numpy as np
import pytest
from turbodash.loss_models import loss_model_kacker_okapuu as ko


def derivative(function, x):
    return np.imag(function(np.asarray(x)+1e-25j))/1e-25


@pytest.mark.parametrize("factor,join,value,slope,curvature", [
    ("K1", .15, .9971875, -.0375, -.25), ("K1", .25, .9375, -1.25, 0.),
    ("K1", .9, .125, -1.25, 0.),
    ("K2", .8, .64, 1.6, 2.),
])
def test_join_matches_outer_value_slope_and_curvature(factor, join, value, slope, curvature):
    f = (lambda m: ko.get_compressible_correction_factors(.3, m)[2]) if factor == "K1" else (
        lambda r: ko.get_compressible_correction_factors(r, 1.)[1])
    np.testing.assert_allclose(f(join), value, atol=2e-15)
    np.testing.assert_allclose(derivative(f, join), slope, atol=2e-12)
    for h in (4e-6, 2e-6):
        for side in (-1, 1):
            d0, d1, d2 = [derivative(f, join+side*k*h) for k in (0, 1, 2)]
            estimate = side*(-3*d0+4*d1-d2)/(2*h)
            np.testing.assert_allclose(estimate, curvature, atol=8e-5)


def test_factors_are_bounded_monotone_and_retain_outer_laws():
    x = np.linspace(0, 2., 20001)
    _, _, k1 = ko.get_compressible_correction_factors(.3, x)
    _, k2, _ = ko.get_compressible_correction_factors(x, 1.)
    assert np.all((0 <= k1) & (k1 <= 1))
    assert np.all((0 <= k2) & (k2 <= 1))
    assert np.max(np.diff(k1)) < 0
    assert np.min(np.diff(k2)) > 0
    np.testing.assert_array_equal(k1[x <= .15], 1-.005*(x[x <= .15]/.2)**2)
    np.testing.assert_allclose(k1[(x >= .25) & (x <= .9)], 1.25*(1-x[(x >= .25) & (x <= .9)]), atol=3e-16)
    join_slopes = derivative(lambda m: ko.get_compressible_correction_factors(.3, m)[2],
                             np.linspace(.15, .25, 1001))
    assert np.all(np.diff(join_slopes) <= 1e-14)
    assert np.all((join_slopes >= -1.25-1e-14) & (join_slopes <= -.0375+1e-14))
    np.testing.assert_allclose(ko.get_compressible_correction_factors(.3, .2)[2], .98625, atol=2e-15)
    assert np.all(k1 > 0)
    np.testing.assert_array_equal(k2[x <= .8], x[x <= .8]**2)
    assert np.all(k2 < 1)
    assert np.all(k2 <= x*x+2e-15)
    kp, _, _ = ko.get_compressible_correction_factors(x[::10, None], x[None, ::10])
    assert np.isfinite(kp).all() and np.all((kp >= .4+.2/13-1e-15) & (kp <= 1))


def test_zero_mach_and_extreme_ratios_do_not_divide_by_zero():
    # A vanishing asymptotic correction may legitimately underflow to zero.
    with np.errstate(all="raise", under="ignore"):
        kp, k2, k1 = ko.get_compressible_correction_factors([0., .2, 1.5], 0.)
        np.testing.assert_array_equal(kp, 1.)
        np.testing.assert_array_equal(k1, 1.)
        np.testing.assert_array_equal(k2, [0., 1., 1.])
        np.testing.assert_array_equal(ko.get_compressible_correction_factors(1e100, 1e-100), [1., 1., 1.])
    for ratio in (0., .2, .8, 2.):
        np.testing.assert_array_equal(ko.get_compressible_correction_factors(ratio*1e-12, 1e-12)[0], 1.)
    np.testing.assert_array_equal(ko.get_compressible_correction_factors(0., [0., .2, 1., 1.5])[0], 1.)


def test_asymptotic_tails_have_useful_gradients_in_the_diagnostic_range():
    k1 = lambda x: ko.get_compressible_correction_factors(.3, x)[2]
    k2 = lambda x: ko.get_compressible_correction_factors(x, 1.)[1]
    assert derivative(k1, 0.) == 0
    assert np.all(derivative(k1, [.05, .1, .15, .25, .9, 1., 1.5, 2.]) < 0)
    assert np.all(derivative(k2, [.8, .9, 1., 1.5, 2.]) > 0)
    np.testing.assert_allclose(k1(1.), 1/24, rtol=2e-15)
    np.testing.assert_allclose(k2(1.), .64+.36/np.sqrt(2), rtol=2e-15)
    assert k1(100.) < k1(2.) and k2(100.) > k2(2.)
    # The former floor hid this nonzero combined-factor sensitivity.
    kp = lambda x: ko.get_compressible_correction_factors(2., x)[0]
    assert .4 < kp(1.2) < .5 and abs(derivative(kp, 1.2)) > 1e-6


def kp_at_unbounded_factor(raw):
    # At M2=1, K1=1/24. For raw>=0.4, the ratio stays below 0.8,
    # where K2=r**2, giving an independent inverse of the raw factor.
    m2 = 1.
    m1 = m2*np.sqrt((1-np.asarray(raw))/(23/24))
    return ko.get_compressible_correction_factors(m1, m2)[0]


def test_rational_continuation_preserves_positive_sensitivity_and_outer_identity():
    raw = np.linspace(0, 1, 10001)
    kp = ko._continue_profile_compressibility(raw)
    slope = derivative(ko._continue_profile_compressibility, raw)
    assert np.isfinite(kp).all() and np.all(np.diff(kp) > 0)
    assert np.all(kp >= raw-1e-15) and np.all((kp >= .4+.2/13-1e-15) & (kp <= 1))
    assert np.all((slope >= 7/169-1e-15) & (slope <= 1+1e-15))
    np.testing.assert_array_equal(kp[raw >= .6], raw[raw >= .6])
    # Independent values obtained from t=3, 1 and 1/2 respectively.
    np.testing.assert_allclose(ko._continue_profile_compressibility([0.,.4,.5]),
                               [.4+.2/13,7/15,18/35], atol=1e-15)
    # Check integration through both K1 and K2, not just the scalar helper.
    raw = np.linspace(.4, .99, 591)
    np.testing.assert_allclose(kp_at_unbounded_factor(raw),
                               ko._continue_profile_compressibility(raw), atol=3e-16)


@pytest.mark.parametrize("function", [ko._continue_profile_compressibility,kp_at_unbounded_factor])
def test_rational_continuation_join_is_c2(function):
    np.testing.assert_allclose(function(.6), .6, atol=2e-16)
    np.testing.assert_allclose(derivative(function, .6), 1., atol=2e-14)
    for h in (2e-6,1e-6):
        for side in (-1, 1):
            d0, d1, d2 = [derivative(function, .6+side*k*h) for k in (0,1,2)]
            curvature = side*(-3*d0+4*d1-d2)/(2*h)
            np.testing.assert_allclose(curvature, 0., atol=1e-6)


@pytest.mark.parametrize("m1,m2", [(-.1,.5),(.1,-.5),(np.inf,.5),(.1,np.nan)])
def test_invalid_mach_is_not_made_finite(m1,m2):
    assert np.isnan(ko.get_compressible_correction_factors(m1,m2)).all()


def test_kp_gradient_agrees_with_independent_real_differences():
    for m1, m2 in ((.2,.2),(.6,.64),(.95,.98),(1.2,1.3),(0.,0.)):
        for index in (0,1):
            f = (lambda x: ko.get_compressible_correction_factors(x,m2)[0]) if index == 0 else (
                lambda x: ko.get_compressible_correction_factors(m1,x)[0])
            x=(m1,m2)[index]
            if x == 0:
                np.testing.assert_allclose(derivative(f,x),0,atol=1e-15)
                continue
            h=1e-5
            real=(f(x-2*h)-8*f(x-h)+8*f(x+h)-f(x+2*h))/(12*h)
            np.testing.assert_allclose(derivative(f,x),real,atol=2e-9,rtol=2e-8)
