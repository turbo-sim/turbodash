"""Production profile interpolation: bounds, gradients and corrected integration."""
from copy import deepcopy
import numpy as np
import pytest
from turbodash.loss_models import loss_model_kacker_okapuu as ko


def profile_inputs(z, thickness=.2, reynolds=5e5, mach_in=.2, mach_out=.5):
    flow = dict(Re_out=reynolds, Ma_rel_in=mach_in, Ma_rel_out=mach_out,
                p0_rel_in=1.8e5, p0_rel_is=1.8e5, p_in=1.7e5,
                p0_rel_out=1.6e5, p_out=1.2e5, beta_in=-60*z, beta_out=60.)
    geometry = dict(chord=.05, pitch=.035, maximum_thickness=.05*thickness,
                    leading_edge_angle=-60*z, hub_tip_ratio_in=.8)
    return flow, geometry


def test_reference_bounds_endpoints_and_nonzero_negative_gradients():
    z = np.linspace(-1.2, 1.2, 1201)[:, None]
    reaction = np.array([.02, .05, .12])
    impulse = np.array([.09, .1, .3])
    b = ko.get_profile_loss_blend(z, reaction, impulse)
    assert np.isfinite(b).all() and np.all(b > .8*reaction)
    assert np.all(np.diff(b, axis=0) > 0)
    for coordinate, expected in ((-1., .9*reaction), (0., reaction), (1., impulse)):
        np.testing.assert_array_equal(ko.get_profile_loss_blend(coordinate, reaction, impulse), expected)
    np.testing.assert_allclose(ko.get_profile_loss_blend(-1e6, reaction, impulse), .8*reaction, rtol=3e-13)
    negative = np.array([-1.2, -1., -.5, -.01])
    slope = np.imag(ko.get_profile_loss_blend(negative+1e-30j, .02, .09))/1e-30
    np.testing.assert_allclose(slope, -.4*.02*negative/(1+negative**2)**2, rtol=3e-14)
    assert np.all(slope > 0)


def test_convex_interpolation_also_accepts_decreasing_reference_pair():
    z = np.linspace(0, 1, 201)
    b = ko.get_profile_loss_blend(z, .1, .01)
    assert np.all((b >= .01) & (b <= .1))
    np.testing.assert_allclose(b, (1-z*z)*.1+z*z*.01, rtol=2e-15)
    outside = ko.get_profile_loss_blend(np.array([1., 1.01, 2., 1e6]), .1, .01)
    assert np.all(np.diff(outside) < 0)
    assert np.all((outside >= .008) & (outside <= .01))
    np.testing.assert_allclose(outside[-1], .008, rtol=2e-15)


@pytest.mark.parametrize("args", [(0., 0., .1), (0., .1, -.1), (np.nan, .1, .2),
                                  (0., np.inf, .2)])
def test_invalid_reference_domains_are_rejected(args):
    with pytest.raises(ValueError):
        ko.get_profile_loss_blend(*args)


def test_complex_step_and_c1_join_not_c2():
    z = np.array([-1.2, -.5, 0., .2, 1.])
    r, i, step = .027, .102, 1e-30
    dz = np.imag(ko.get_profile_loss_blend(z+1j*step, r, i))/step
    dr = np.imag(ko.get_profile_loss_blend(z, r+1j*step, i))/step
    di = np.imag(ko.get_profile_loss_blend(z, r, i+1j*step))/step
    np.testing.assert_allclose(dz, np.where(z < 0, -.4*r*z/(1+z*z)**2, 2*z*(i-r)), atol=1e-16)
    np.testing.assert_allclose(dr, np.where(z < 0, .8+.2/(1+z*z), 1-z*z), atol=1e-16)
    np.testing.assert_allclose(di, np.where(z < 0, 0, z*z), atol=1e-16)
    h = 1e-4
    f = lambda x: ko.get_profile_loss_blend(x, r, i)
    left = (3*f(0)-4*f(-h)+f(-2*h))/(2*h)
    right = (-3*f(0)+4*f(h)-f(2*h))/(2*h)
    np.testing.assert_allclose([left, right], 0, atol=1e-11)
    curvatures = [(f(0)-2*f(-h)+f(-2*h))/h**2, (f(2*h)-2*f(h)+f(0))/h**2]
    np.testing.assert_allclose(curvatures, [-.4*r, 2*(i-r)], atol=2e-9)


@pytest.mark.parametrize("reaction,impulse", [(.02, .09), (.1, .01), (.05, .05)])
def test_positive_z_continuation_is_bounded_and_slope_matches_at_impulse(reaction, impulse):
    fn = lambda z: ko.get_profile_loss_blend(z, reaction, impulse)
    z = np.r_[1., np.geomspace(1.000001, 1e6, 1001)]
    value = fn(z)
    assert np.all(np.isfinite(value))
    assert np.all((value >= .8*impulse) & (value <= 1.2*impulse))
    limit = impulse*(1+.2*np.sign(impulse-reaction))
    np.testing.assert_allclose(fn(1e8), limit, rtol=2e-15)
    h = 1e-6
    slopes = np.imag(fn(np.array([1-h, 1., 1+h])+1e-25j))/1e-25
    np.testing.assert_allclose(slopes, 2*(impulse-reaction), atol=4e-7)
    for point in (1.01, 1.2, 2., 10.):
        finite = (fn(point-2*h)-8*fn(point-h)+8*fn(point+h)-fn(point+2*h))/(12*h)
        np.testing.assert_allclose(np.imag(fn(point+1e-25j))/1e-25, finite, atol=2e-11, rtol=2e-6)


@pytest.mark.parametrize("row", [0, 1])
@pytest.mark.parametrize("condition", [(5e5, .2, .5), (1e5, .7, .9), (2e6, .9, 1.2)])
def test_full_profile_keeps_reynolds_mach_thickness_and_shock(row, condition):
    re, mi, mo = condition
    z = np.array([-1., -.5, 0., .005, .025, .045, .05, .5, 1., 1.2])
    t = np.array([.15, .2, .25, .05, .15, .4, .25, .15, .2, .25])
    flow, geometry = profile_inputs(z, t, re, mi, mo)
    before = deepcopy((flow, geometry))
    angle = ko.math.smooth_abs(60.)
    r, i = ko.nozzle_blades(.7, angle), ko.impulse_blades(.7, angle)
    tail_argument = (i-r)*(z*z-1)/(.2*i)
    blend = np.where(z < 0, r*(.8+.2/(1+z*z)),
                     np.where(z <= 1, r+(i-r)*z*z,
                              i+.2*i*tail_argument/np.sqrt(1+tail_argument**2)))
    kp = ko.get_compressible_correction_factors(mi, mo)[0]
    hub = ko.get_hub_to_mean_mach_ratio(.8, row)
    shock = .75*max(0, hub*mi-.4)**1.75*.8*(1.8e5-1.7e5)/(1.6e5-1.2e5)
    q = np.clip(z/.05, 0., 1.)
    smoothstep = 10*q**3-15*q**4+6*q**5
    exponent = np.where(z <= 1, np.maximum(0,z), 1+(z-1)/np.sqrt(1+(z-1)**2))
    factor = 1+smoothstep*((t/.2)**exponent-1)
    expected = .914*ko.get_reynolds_correction_factor(re)*(2/3*blend*factor*kp+shock)
    np.testing.assert_allclose(ko.get_profile_loss(flow, geometry, row), expected, rtol=5e-14)
    for now, old in zip((flow, geometry), before):
        for name in now:
            np.testing.assert_array_equal(now[name], old[name])


def test_full_profile_complex_step_angle_and_thickness_gradient():
    h = 1e-30
    z = -.5
    flow, geometry = profile_inputs(z+1j*h)
    slope = np.imag(ko.get_profile_loss(flow, geometry, 0))/h
    base = ko.get_profile_loss(*profile_inputs(0.), 0)
    np.testing.assert_allclose(slope/base, -.4*z/(1+z*z)**2, rtol=3e-14)
    for z in (-1., 0., .5, 1.):
        value = ko.get_profile_loss(*profile_inputs(z), 0)
        derivative = np.imag(ko.get_profile_loss(*profile_inputs(z, .2+1j*h), 0))/h
        np.testing.assert_allclose(derivative, value*max(z, 0)/.2, atol=1e-15)


@pytest.mark.parametrize("thickness", [.05, .15, .2, .25, .4])
def test_thickness_transition_is_positive_bounded_and_preserves_outer_law(thickness):
    z = np.linspace(-1.2, 1.2, 2401)
    factor = ko.get_profile_thickness_factor(z, thickness)
    original = (thickness/.2)**np.maximum(0, z)
    assert np.all(np.isfinite(factor)) and np.all(factor > 0)
    np.testing.assert_array_equal(factor[z <= 0], 1.)
    core = (z >= .05) & (z <= 1.)
    np.testing.assert_array_equal(factor[core], original[core])
    assert np.all(factor >= np.minimum(1., original)-1e-15)
    assert np.all(factor <= np.maximum(1., original)+1e-15)
    np.testing.assert_allclose(ko.get_profile_thickness_factor(.025, thickness),
                               .5*(1+(thickness/.2)**.025), rtol=2e-15)
    np.testing.assert_array_equal(ko.get_profile_thickness_factor(z, .2), 1.)


@pytest.mark.parametrize("thickness", [.05, .15, .2, .25, .4])
def test_thickness_positive_z_tail_has_finite_bounded_exponent(thickness):
    z = np.r_[1., np.geomspace(1.000001, 1e8, 501)]
    fn = lambda value: ko.get_profile_thickness_factor(value, thickness)
    factor = fn(z)
    assert np.all(np.isfinite(factor)) and np.all(factor > 0)
    ends = np.array([thickness/.2, (thickness/.2)**2])
    assert np.all(factor >= ends.min()-1e-15) and np.all(factor <= ends.max()+1e-15)
    np.testing.assert_allclose(factor[-1], ends[1], rtol=2e-15)
    h = 1e-6
    expected = thickness/.2*np.log(thickness/.2)
    slopes = np.imag(fn(np.array([1-h, 1., 1+h])+1e-25j))/1e-25
    np.testing.assert_allclose(slopes, expected, atol=3e-6)


@pytest.mark.parametrize("thickness", [.05, .15, .2, .25, .4])
def test_thickness_transition_matches_value_slope_and_curvature_at_both_joins(thickness):
    f = lambda z: ko.get_profile_thickness_factor(z, thickness)
    d = lambda z: np.imag(f(z+1e-25j))/1e-25
    h = 1e-6
    for boundary in (0., .05):
        value = 1. if boundary == 0 else (thickness/.2)**boundary
        slope = 0. if boundary == 0 else value*np.log(thickness/.2)
        curvature = 0. if boundary == 0 else value*np.log(thickness/.2)**2
        np.testing.assert_allclose(f(boundary), value, rtol=2e-15)
        np.testing.assert_allclose(d(boundary), slope, rtol=2e-14, atol=1e-15)
        for sign in (-1, 1):
            d1, d2 = d(boundary+sign*h), d(boundary+sign*2*h)
            limit = 2*d1-d2
            second = sign*(-3*d(boundary)+4*d1-d2)/(2*h)
            np.testing.assert_allclose(limit, slope, rtol=0, atol=5e-8)
            np.testing.assert_allclose(second, curvature, rtol=0, atol=2e-5)


@pytest.mark.parametrize("row", [0, 1])
def test_full_profile_has_no_slope_jump_at_zero_for_all_thicknesses(row):
    thickness = np.array([.05, .15, .2, .25, .4])
    f = lambda z: ko.get_profile_loss(*profile_inputs(z, thickness), row)
    d = lambda z: np.imag(f(z+1e-25j))/1e-25
    np.testing.assert_allclose(d(0.), 0, atol=1e-15)
    h = 1e-6
    for sign in (-1, 1):
        np.testing.assert_allclose(2*d(sign*h)-d(sign*2*h), 0, atol=1e-10)
    # Gradient in thickness inside the band is independent of the z join.
    z, tau = .025, .15
    derivative = np.imag(ko.get_profile_loss(*profile_inputs(z, tau+1e-25j), row))/1e-25
    value_without_thickness = float(ko.get_profile_loss(*profile_inputs(z, .2), row))
    expected = value_without_thickness*.5*z/tau*(tau/.2)**z
    np.testing.assert_allclose(derivative, expected, rtol=3e-14)


@pytest.mark.parametrize("row", [0, 1])
def test_low_outlet_angles_use_the_reference_surface_without_a_minimum(row):
    """Fixed-z sweeps isolate reference-angle behavior from changing turning."""
    for z in (-1., 0., .5, 1.):
        def profile(angle):
            flow, geometry = profile_inputs(z)
            # Keep the geometry direction fixed on each mirrored cascade.
            geometry['metal_angle_out'] = np.where(np.real(angle) < 0, -60., 60.)
            effective = ko.get_effective_outlet_angle(angle, geometry['metal_angle_out'])
            flow.update(beta_out=angle, beta_in=-z*effective)
            geometry["leading_edge_angle"] = -z*effective
            return ko.get_profile_loss(flow, geometry, row)
        angles = np.array([5., 10., 20., 35., 39., 40., 41., 60.])
        effective = ko.get_effective_outlet_angle(angles)
        magnitude = np.sqrt(effective**2+1e-5)
        reaction = ko.nozzle_blades(.7, magnitude)
        impulse = ko.impulse_blades(.7, magnitude)
        blend = reaction*(.8+.2/(1+z*z)) if z < 0 else (1-z*z)*reaction+z*z*impulse
        np.testing.assert_allclose(profile(angles), .914*(2/3)*.94*blend, rtol=3e-14)
        np.testing.assert_array_equal(profile(angles), profile(-angles))
        gradient = np.imag(profile(angles+1e-25j))/1e-25
        h = 1e-3
        finite = (profile(angles-2*h)-8*profile(angles-h)+8*profile(angles+h)-profile(angles+2*h))/(12*h)
        np.testing.assert_allclose(gradient, finite, rtol=2e-6, atol=1e-11)
        assert np.all(np.abs(gradient[:4]) > 1e-8)


@pytest.mark.parametrize('direction', [-1., 1.])
def test_angle_ratio_preserves_design_orientation_through_axial_exit(direction):
    theta = np.array([-60., -30., 0., 30., 60.])[:, None]
    ordinary = direction*np.array([10., 20., 80.])[None, :]
    np.testing.assert_array_equal(ko.get_cascade_angle_ratio(theta, ordinary, direction), -theta/ordinary)
    angle = np.array([-1e-6, 0., 1e-6])[None, :]
    fn = lambda beta: ko.get_cascade_angle_ratio(theta, beta, direction)
    with np.errstate(all='raise'):
        values = fn(angle)
        slopes = np.imag(fn(angle+1e-25j))/1e-25
    assert np.all(np.isfinite(values)) and np.all(np.isfinite(slopes))
    np.testing.assert_allclose(values[:, 1], -.3*theta[:, 0]/direction, rtol=2e-15)
    np.testing.assert_array_equal(np.sign(values), np.broadcast_to(np.sign(-theta/direction), values.shape))
    np.testing.assert_allclose(slopes, .03*theta+np.zeros_like(angle), rtol=3e-7, atol=1e-15)


@pytest.mark.parametrize('direction', [-1., 1.])
def test_effective_outlet_angle_is_monotone_has_no_floor_and_matches_join(direction):
    angle = direction*np.r_[-1e6, -100., -10., 0., 5., 9., 10., 20., 80.]
    fn = lambda beta: ko.get_effective_outlet_angle(beta, direction)
    with np.errstate(all='raise'):
        effective = fn(angle)
        derivative = np.imag(fn(angle+1e-25j))/1e-25
    assert np.all(direction*effective > 0) and np.all(derivative > 0)
    assert np.all(np.diff(direction*effective) > 0)
    assert direction*effective[0] < 1e-8
    np.testing.assert_array_equal(effective[-3:], angle[-3:])
    np.testing.assert_allclose(fn(0.), direction*10/3, rtol=2e-15)
    join, h = direction*10, 1e-4
    points = join+np.array([-h, 0., h])
    slopes = np.imag(fn(points+1e-25j))/1e-25
    np.testing.assert_allclose(slopes, 1., atol=4e-10)
    np.testing.assert_allclose(np.diff(slopes)/h, 0., atol=7e-6)


@pytest.mark.parametrize('thickness', [.05, .2, .4])
@pytest.mark.parametrize('row', [0, 1])
def test_profile_remains_finite_near_zero_outlet_angle_and_reversed_reference_order(thickness, row):
    flow, geometry = profile_inputs(.5, thickness)
    geometry['leading_edge_angle'] = -30.
    angles = np.r_[-1.01, -1., -0.99, -.1, -.01, -1e-6, 0., 1e-6, .01, .1, .99, 1., 1.01]
    def loss(angle):
        return ko.get_profile_loss(dict(flow, beta_out=angle), geometry, row)
    with np.errstate(all='raise'):
        values = loss(angles)
        derivative = np.imag(loss(angles+1e-25j))/1e-25
    assert np.all(np.isfinite(values)) and np.all(values > 0)
    assert np.all(np.isfinite(derivative))
    # Before continuation the same path reached order 1e5 at 0.01 degree.
    assert values.max() < 10.
    np.testing.assert_allclose(loss(np.array([-1e-8, 1e-8])), loss(0.), rtol=1e-10)
    # This surface slice formerly raised for z just above the impulse endpoint.
    geometry['pitch'] = .1*geometry['chord']
    geometry['leading_edge_angle'] = -ko.get_effective_outlet_angle(1.)*1.000001
    assert np.isfinite(loss(1.))


@pytest.mark.parametrize('theta', [-30., 30.])
@pytest.mark.parametrize('direction', [-1., 1.])
def test_profile_keeps_same_blend_branch_across_zero_exit(theta, direction):
    flow, geometry = profile_inputs(.5)
    geometry.update(leading_edge_angle=theta, metal_angle_out=60*direction)
    fn = lambda beta: ko.get_profile_loss(dict(flow, beta_out=beta), geometry, 0)
    beta = np.array([-1e-4, 0., 1e-4])
    z = ko.get_cascade_angle_ratio(theta, beta, direction)
    assert np.all(np.sign(z) == np.sign(-theta/direction))
    assert np.all(abs(z) > 8.99)
    values = fn(beta)
    derivatives = np.imag(fn(beta+1e-25j))/1e-25
    assert np.all(np.isfinite(values)) and np.all(values > 0)
    assert np.all(np.isfinite(derivatives))
    h = 1e-4
    finite = (fn(-2*h)-8*fn(-h)+8*fn(h)-fn(2*h))/(12*h)
    np.testing.assert_allclose(derivatives[1], finite, rtol=2e-6, atol=1e-11)
    np.testing.assert_allclose(derivatives[[0, 2]], derivatives[1], atol=2e-6)
    nozzle_geometry = dict(geometry, leading_edge_angle=0.)
    nozzle = ko.get_profile_loss(dict(flow, beta_out=0.), nozzle_geometry, 0)
    assert not np.isclose(values[1], nozzle, rtol=.01)


def test_reynolds_continuation_is_bounded_smooth_and_preserves_valid_correlations():
    effective = ko.get_effective_reynolds_number
    factor = ko.get_reynolds_correction_factor
    ordinary = np.array([1e4, 2e4, 1e5, 2e5, 5e5, 1e6, 1e7])
    np.testing.assert_array_equal(effective(ordinary), ordinary)
    reynolds = np.r_[-1e12, -1e8, -1e6, -1e4, 0., 1e3, 9e3, 1e4, 1.1e4]
    with np.errstate(all='raise'):
        values = factor(reynolds)
        gradients = np.imag(factor(reynolds+1e-25j))/1e-25
    assert np.all(np.isfinite(values)) and np.all(values > 0)
    assert np.all(np.isfinite(gradients)) and np.all(gradients < 0)
    assert values.max() <= (1e3/2e5)**(-.4)
    np.testing.assert_allclose(factor(-1e12), (1e3/2e5)**(-.4), rtol=1e-12)
    h = .01
    points = 1e4+np.array([-h, 0., h])
    np.testing.assert_allclose(np.imag(effective(points+1e-25j))/1e-25, 1., atol=5e-11)
    for re in (-1e6, -1e4, 0., 9000.):
        h = .1
        finite = (factor(re-2*h)-8*factor(re-h)+8*factor(re+h)-factor(re+2*h))/(12*h)
        np.testing.assert_allclose(np.imag(factor(re+1e-25j))/1e-25, finite, atol=2e-13, rtol=2e-6)


@pytest.mark.parametrize('reynolds', [-1e6, -1e4, 0., 100., 1e4])
def test_complete_profile_accepts_negative_and_zero_reynolds(reynolds):
    flow, geometry = profile_inputs(.5, reynolds=reynolds)
    with np.errstate(all='raise'):
        actual = ko.get_profile_loss(flow, geometry, 0)
    assert np.isfinite(actual) and actual > 0
    baseline = ko.get_profile_loss(*profile_inputs(.5, reynolds=5e5), 0)
    np.testing.assert_allclose(actual/baseline, ko.get_reynolds_correction_factor(reynolds), rtol=3e-15)
