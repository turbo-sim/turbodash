"""Secondary aspect-factor continuation, physical limits and assembly."""

import numpy as np
import pytest

from turbodash.loss_models_new import loss_model_kacker_okapuu as ko


def inputs(ar, alpha1=40., alpha2=60., mach_in=.6, mach_out=.8, row=0):
    """Opposite signed row orientations with the same physical turning."""
    direction = 1 if row == 0 else -1
    flow = dict(Ma_rel_in=mach_in, Ma_rel_out=mach_out,
                beta_in=-direction*np.asarray(alpha1),
                beta_out=direction*np.asarray(alpha2))
    geometry = dict(chord=.05, height=.05*np.asarray(ar),
                    meridional_chord=.04, leading_edge_angle=flow["beta_in"])
    return flow, geometry


def reference_continuation(raw):
    """Independent mathematical expression for the aspect-factor cap."""
    raw = np.asarray(raw)
    t = np.maximum(raw-.8, 0)/.2
    return np.where(raw <= .8, raw, 1.-.2/(1+t+t*t))


def independent_reference(flow, geometry):
    """Assemble secondary loss independently on the outer aspect branches.

    The shared compressibility helper has its own tests. Secondary loading,
    both factor continuations and outer multipliers are independent
    here. Test inputs avoid the numerical 1.5--2.5 aspect-ratio patch.
    """
    ar = np.asarray(geometry["height"])/geometry["chord"]
    assert np.all((ar <= 1.5) | (ar >= 2.5))
    far_raw = np.empty_like(ar, dtype=float)
    low = ar <= 1.5
    far_raw[low] = (1-.25*np.sqrt(2-ar[low]))/ar[low]
    far_raw[~low] = 1/ar[~low]
    far = reference_continuation(far_raw)
    inlet, outlet = np.deg2rad(flow["beta_in"]), np.deg2rad(flow["beta_out"])
    mean = np.arctan((np.tan(inlet)+np.tan(outlet))/2)
    loading = 4*(np.tan(inlet)-np.tan(outlet))**2*np.cos(outlet)**2/np.cos(mean)
    geometry_loss = (1.2*.0334*far*loading*np.cos(outlet)
                     / np.cos(np.deg2rad(geometry["leading_edge_angle"])))
    kp = ko.get_compressible_correction_factors(flow["Ma_rel_in"], flow["Ma_rel_out"])[0]
    influence = reference_continuation((geometry["meridional_chord"]/geometry["height"])**2)
    ks = (1-influence)+influence*kp
    return geometry_loss*ks, geometry_loss, ks


def complex_slope(function, x):
    return np.imag(function(np.asarray(x, dtype=complex)+1e-25j))/1e-25


def test_aspect_continuation_identity_bounds_and_no_practical_plateau():
    assert ko.SECONDARY_FACTOR_CONTINUATION_START == .8
    assert ko.SECONDARY_FACTOR_ASYMPTOTE == 1.
    fn = ko.continue_secondary_aspect_ratio_factor
    unchanged = np.linspace(0, .8, 101)
    np.testing.assert_array_equal(fn(unchanged), unchanged)
    raw = np.r_[unchanged, np.geomspace(.800001, 1e3, 501)]
    values = fn(raw)
    assert np.all(np.isfinite(values))
    assert np.all(np.diff(values) > 0)
    assert np.all((values >= 0) & (values < 1.))
    assert np.all(values <= raw)
    assert np.all(complex_slope(fn, raw) > 0)
    np.testing.assert_allclose(values, reference_continuation(raw), atol=1e-16, rtol=2e-15)


def test_aspect_continuation_c2_at_onset():
    h = 1e-7
    points = .8+np.array([-h, 0., h])
    fn = ko.continue_secondary_aspect_ratio_factor
    values = fn(points)
    np.testing.assert_allclose(values, points, rtol=0, atol=2e-16)
    slopes = complex_slope(fn, points)
    np.testing.assert_allclose(slopes, 1., rtol=0, atol=1e-11)
    # One-sided derivatives of the first derivative test curvature continuity.
    np.testing.assert_allclose(np.diff(slopes)/h, 0., rtol=0, atol=2e-5)


def test_aspect_continuation_gradients_against_real_differences():
    raw = np.array([0., .1, .8, .81, .9, 1., 2., 10.])
    h = 1e-5
    fn = ko.continue_secondary_aspect_ratio_factor
    finite = (fn(raw-2*h)-8*fn(raw-h)+8*fn(raw+h)-fn(raw+2*h))/(12*h)
    exact = complex_slope(fn, raw)
    np.testing.assert_allclose(exact, finite, rtol=2e-7, atol=2e-10)
    t = np.maximum(raw-.8, 0)/.2
    expected = np.where(raw <= .8, 1., (1+2*t)/(1+t+t*t)**2)
    np.testing.assert_allclose(exact, expected, rtol=3e-15, atol=1e-16)


def test_aspect_continuation_large_finite_inputs_do_not_overflow():
    raw = np.array([1., 1e6, 1e100, 1e200, np.finfo(float).max])
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        values = ko.continue_secondary_aspect_ratio_factor(raw)
        slopes = complex_slope(ko.continue_secondary_aspect_ratio_factor, raw)
    assert np.all(np.isfinite(values)) and np.all(np.isfinite(slopes))
    assert np.all((values > 0) & (values <= 1.))
    assert np.all(np.diff(values) >= 0)
    assert np.all(slopes >= 0)


def test_production_aspect_factor_bounded_decreasing_and_c2():
    # Include the plotted lower bound and the smaller diagnostic AR explicitly.
    ar = np.unique(np.r_[np.geomspace(.01, 5., 2001), .2])
    fn = ko.get_secondary_aspect_ratio_factor
    values = fn(ar)
    assert np.all(np.isfinite(values))
    assert np.all((values > 0) & (values < 1.))
    assert np.all(np.diff(values) < 0)
    assert np.all(complex_slope(fn, ar) < 0)

    # Solve the original low-AR branch's f_AR = 0.8 crossing independently.
    onset_ar = min(np.roots([.8**2, -2*.8+.25**2, 1.-2*.25**2]))
    for join in (onset_ar, 1.5, 2.5):
        h = 1e-6
        points = join + np.array([-h, 0., h])
        slopes = complex_slope(fn, points)
        np.testing.assert_allclose(slopes, slopes[1], atol=2e-6, rtol=0)
        curvatures = np.diff(slopes)/h
        np.testing.assert_allclose(curvatures[0], curvatures[1], atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("row", [0, 1])
@pytest.mark.parametrize("mach_in,mach_out", [(0., .8), (.2, .5), (.6, .8), (1.2, .4), (.6, 1.5)])
def test_production_matches_aspect_continuation_down_to_ar_point_zero_one(row, mach_in, mach_out):
    ar = np.array([.01, .03, .1, .2, .3, .8, 1.2, 1.4, 2.6, 5.])[:, None, None]
    alpha1 = np.array([0., 20., 40., 60., 80.])[None, :, None]
    alpha2 = np.arange(20., 81., 10.)[None, None, :]
    flow, geometry = inputs(ar, alpha1, alpha2, mach_in, mach_out, row)
    expected, _, _ = independent_reference(flow, geometry)
    actual = ko.get_secondary_loss(flow, geometry, row)
    np.testing.assert_allclose(actual, expected, rtol=3e-14, atol=2e-16)
    assert np.all(np.isfinite(actual))
    assert np.all(actual > 0)
    assert np.max(actual) > .5, "High-turning losses must retain no final 0.5 cap."


@pytest.mark.parametrize("row", [0, 1])
def test_compressibility_is_multiplicative_with_no_final_loss_cap(row):
    flow, geometry = inputs(.2, alpha1=80., alpha2=20., row=row)
    expected, uncompressed, ks = independent_reference(flow, geometry)
    kp = ko.get_compressible_correction_factors(flow["Ma_rel_in"], flow["Ma_rel_out"])[0]
    assert kp < ks < kp+.001
    actual = ko.get_secondary_loss(flow, geometry, row)
    flow["Ma_rel_in"] = 0.
    zero_mach_correction = ko.get_secondary_loss(flow, geometry, row)
    np.testing.assert_allclose(zero_mach_correction, uncompressed, rtol=3e-14)
    np.testing.assert_allclose(actual, expected, rtol=3e-14)
    np.testing.assert_allclose(actual, zero_mach_correction*ks, rtol=3e-14)
    assert actual > .5 and zero_mach_correction > .5


@pytest.mark.parametrize("row", [0, 1])
def test_zero_turning_remains_exactly_zero(row):
    flow, geometry = inputs(np.geomspace(.01, 5., 101), row=row)
    flow["beta_in"] = flow["beta_out"]
    geometry["leading_edge_angle"] = flow["beta_in"]
    np.testing.assert_array_equal(ko.get_secondary_loss(flow, geometry, row), 0.)


@pytest.mark.parametrize("row", [0, 1])
def test_complete_secondary_gradients_include_aspect_continuation(row):
    # Mach-zero correction isolates the aspect-ratio trend, including active
    # continuation, untouched factors and the retained 1.5--2.5 smooth patch.
    def loss(ar):
        flow, geometry = inputs(ar, alpha1=40., alpha2=20., mach_in=0., row=row)
        return ko.get_secondary_loss(flow, geometry, row)
    ar = np.array([.01, .05, .2, .5, 1., 1.5, 2., 2.5, 5.])
    h = 1e-5
    finite = (loss(ar-2*h)-8*loss(ar-h)+8*loss(ar+h)-loss(ar+2*h))/(12*h)
    exact = complex_slope(loss, ar)
    np.testing.assert_allclose(exact, finite, rtol=4e-7, atol=2e-10)
    assert np.all(exact < 0)


@pytest.mark.parametrize("kp", [0., .4153846153846154, .6, 1.])
def test_compressibility_influence_bounds_outer_identity_and_sensitivity(kp):
    ratio = np.r_[0., np.geomspace(.01, 100., 501), np.sqrt(.8)]
    influence = reference_continuation(ratio**2)
    actual = ko.get_secondary_compressibility_factor(kp, ratio)
    np.testing.assert_allclose(actual, (1-influence)+influence*kp, rtol=3e-15, atol=2e-16)
    assert np.all(actual >= kp) and np.all(actual <= 1.)
    ordinary = ratio**2 <= .8
    np.testing.assert_allclose(actual[ordinary], 1-ratio[ordinary]**2*(1-kp), atol=2e-16)
    sensitivity = complex_slope(lambda value: ko.get_secondary_compressibility_factor(value, ratio), kp)
    np.testing.assert_allclose(sensitivity, influence, rtol=2e-15, atol=1e-16)
    assert np.all(sensitivity[ratio > 0] > 0)
    if kp < 1:
        assert np.all(complex_slope(lambda value: ko.get_secondary_compressibility_factor(kp, value), ratio[ratio > 0]) < 0)


def test_compressibility_factor_c2_join_and_vector_gradients():
    fn = lambda ratio: ko.get_secondary_compressibility_factor(.6, ratio)
    join, h = np.sqrt(.8), 1e-6
    slopes = complex_slope(fn, join+np.array([-h, 0., h]))
    np.testing.assert_allclose(slopes, slopes[1], rtol=0, atol=1e-6)
    np.testing.assert_allclose(np.diff(slopes)/h, -.8, rtol=0, atol=5e-4)
    ratio = np.array([.1, .5, join, 1., 2., 10.])
    h = 1e-5
    finite = (fn(ratio-2*h)-8*fn(ratio-h)+8*fn(ratio+h)-fn(ratio+2*h))/(12*h)
    np.testing.assert_allclose(complex_slope(fn, ratio), finite, rtol=2e-6, atol=3e-9)


@pytest.mark.parametrize("row", [0, 1])
def test_low_aspect_ratio_retains_mach_sensitivity_and_complete_chain_rule(row):
    ar = np.array([.01, .2, .5, .8/np.sqrt(.8), 1., 1.5, 2., 2.5, 5.])
    def loss(mach):
        flow, geometry = inputs(ar, mach_out=mach, row=row)
        return ko.get_secondary_loss(flow, geometry, row)
    mach, h = .8, 1e-5
    finite = (loss(mach-2*h)-8*loss(mach-h)+8*loss(mach+h)-loss(mach+2*h))/(12*h)
    exact = complex_slope(loss, mach)
    np.testing.assert_allclose(exact, finite, rtol=2e-7, atol=2e-10)
    assert np.all(np.abs(exact) > 1e-6)
    def aspect_loss(value):
        flow, geometry = inputs(value, row=row)
        return ko.get_secondary_loss(flow, geometry, row)
    finite = (aspect_loss(ar-2*h)-8*aspect_loss(ar-h)+8*aspect_loss(ar+h)-aspect_loss(ar+2*h))/(12*h)
    np.testing.assert_allclose(complex_slope(aspect_loss, ar), finite, rtol=2e-6, atol=3e-9)


@pytest.mark.parametrize("row", [0, 1])
def test_metal_angle_projection_is_independent_of_flow_angle_loading(row):
    # Vary only the metal angle: the signed flow angles and hence loading
    # remain fixed. This exercises nonzero incidence without conflating it
    # with the separate Moustapha incidence correction.
    flow, geometry = inputs(.4, alpha1=35., alpha2=60., row=row)
    angles = np.array([-60., -20., 0., 20., 60.])
    def loss(theta):
        return ko.get_secondary_loss(flow, dict(geometry, leading_edge_angle=theta), row)
    baseline = loss(0.)
    actual = loss(angles)
    np.testing.assert_allclose(actual*np.cos(np.deg2rad(angles)), baseline, rtol=3e-15)
    expected_slope = actual*np.tan(np.deg2rad(angles))*np.pi/180
    np.testing.assert_allclose(complex_slope(loss, angles), expected_slope, rtol=4e-15, atol=1e-16)


def test_complete_secondary_loss_is_invariant_to_common_length_units():
    flow, geometry = inputs(np.array([.01, .2, 1.7, 2., 2.5, 5.]))
    baseline = ko.get_secondary_loss(flow, geometry, 0)
    for scale in (.001, 1000.):
        converted = dict(geometry)
        for key in ("height", "chord", "meridional_chord"):
            converted[key] = geometry[key]*scale
        np.testing.assert_allclose(ko.get_secondary_loss(flow, converted, 0), baseline, rtol=5e-15)
