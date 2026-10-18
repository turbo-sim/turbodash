"""KO equation (20), the zero-gap patch, and shared production assembly."""
import numpy as np
import pytest

from turbodash.loss_models import loss_model_kacker_okapuu as ko
from turbodash.loss_models import loss_model_moustapha as mo
from turbodash.loss_models import loss_model_benner as be


def inputs(ar=2., gap_height=.01, angle=60., z=.5):
    flow = dict(beta_in=-z*np.asarray(angle), beta_out=np.asarray(angle))
    geometry = dict(chord=.05, height=.05*np.asarray(ar),
                    tip_clearance=.05*np.asarray(ar)*np.asarray(gap_height))
    return flow, geometry


def slope(fun, x):
    return np.imag(fun(np.asarray(x, dtype=complex)+1e-25j))/1e-25


def paper_loss(flow, geometry):
    """Original unsigned-paper loading, independent of the source helpers."""
    alpha1 = np.deg2rad(-flow['beta_in'])
    alpha2 = np.deg2rad(flow['beta_out'])
    mean = np.arctan((np.tan(alpha1)-np.tan(alpha2))/2)
    cl_over_solidity = 2*(np.tan(alpha1)+np.tan(alpha2))*np.cos(mean)
    return (.37*geometry['chord']/geometry['height']
            *(geometry['tip_clearance']/geometry['chord'])**.78
            *cl_over_solidity**2*np.cos(alpha2)**2/np.cos(mean)**3)


def test_gap_factor_matches_power_law_and_is_monotone_concave():
    delta = ko.TIP_CLEARANCE_TRANSITION
    x = np.r_[0., np.geomspace(1e-12, delta, 500), np.geomspace(delta*1.01, .2, 500)]
    f = ko.get_tip_clearance_gap_factor
    with np.errstate(all='raise'):
        values, gradients = f(x), slope(f, x)
    assert values[0] == 0
    assert np.all(values >= 0) and np.all(np.diff(values) > 0)
    assert np.all(gradients > 0) and np.all(np.diff(gradients) <= 1e-13)
    assert np.all(values <= x**.78+2e-17)
    np.testing.assert_array_equal(values[x >= delta], x[x >= delta]**.78)


def test_gap_patch_c2_and_zero_endpoint_from_independent_constraints():
    p, delta = .78, ko.TIP_CLEARANCE_TRANSITION
    # Solve for the cubic from value/slope/curvature at the outer endpoint.
    a, b, d = np.linalg.solve([[1, 1, 1], [1, 2, 3], [0, 2, 6]], [1, p, p*(p-1)])
    f = ko.get_tip_clearance_gap_factor
    np.testing.assert_allclose(slope(f, 0.), a*delta**(p-1), rtol=2e-15)
    h = delta*1e-5
    # One-sided real stencil at the physical zero-gap boundary.
    right = (-3*f(0.)+4*f(h)-f(2*h))/(2*h)
    np.testing.assert_allclose(right, slope(f, 0.), rtol=3e-10)
    points = delta + np.array([-h, 0., h])
    np.testing.assert_allclose(slope(f, points), p*delta**(p-1), rtol=3e-6)
    curvature = np.diff(slope(f, points))/h
    np.testing.assert_allclose(curvature, p*(p-1)*delta**(p-2), rtol=3e-5)
    q = np.linspace(0, 1, 31)
    np.testing.assert_allclose(f(q*delta), delta**p*(a*q+b*q*q+d*q**3), rtol=3e-15)


def test_matches_paper_in_unmodified_domain_and_continues_below_it():
    rng = np.random.default_rng(20021)
    ar = rng.uniform(.2, 5., 4000)
    f, g = inputs(ar, gap_height=1.)
    f.update(beta_in=rng.uniform(-80, 80, ar.size), beta_out=rng.uniform(-80, 80, ar.size))
    g['tip_clearance'] = g['chord']*rng.uniform(.001, .1, ar.size)
    original = paper_loss(f, g)
    loading = original/(.37/ar*(g['tip_clearance']/g['chord'])**.78)
    actual = ko.get_tip_clearance_loss(f, g, 1)
    unchanged = ar >= 1
    assert unchanged.sum() > 1000
    np.testing.assert_allclose(actual[unchanged], original[unchanged], rtol=5e-14, atol=1e-15)
    assert np.all(actual <= original+1e-13) and np.all(actual >= 0)
    assert np.all(np.isfinite(actual))
    f, g = inputs(np.array([.2, 1., 5.])[:, None], np.linspace(0, .03, 21)[None, :])
    values = ko.get_tip_clearance_loss(f, g, 1)
    assert values.shape == (3, 21)
    assert np.all(np.diff(values, axis=1) > 0)


@pytest.mark.parametrize('row', [0, 1])
def test_zero_gap_zero_turning_invalid_geometry_and_angle_reflection(row):
    f, g = inputs(gap_height=0.)
    assert ko.get_tip_clearance_loss(f, g, row) == 0
    f, g = inputs(z=-1)
    assert ko.get_tip_clearance_loss(f, g, row) == 0
    f, g = inputs()
    original = ko.get_tip_clearance_loss(f, g, row)
    reflected = {key: -value for key, value in f.items()}
    np.testing.assert_allclose(ko.get_tip_clearance_loss(reflected, g, row), original, rtol=0, atol=0)
    if row == 0:
        assert original == 0
    for key, invalid in [('tip_clearance', -1e-4), ('height', -1e-4), ('chord', 0.),
                         ('chord', -1.), ('height', np.inf), ('tip_clearance', np.nan)]:
        with np.errstate(all='raise'):
            assert np.isnan(ko.get_tip_clearance_loss(f, dict(g, **{key: invalid}), row))
    assert np.isnan(ko.get_tip_clearance_gap_factor(-.001))


def test_aspect_continuation_is_bounded_monotone_and_c2():
    f = ko.get_tip_clearance_aspect_ratio_factor
    x = np.r_[0., np.geomspace(1e-10, .9999, 400), np.linspace(1, 5, 400)]
    with np.errstate(all='raise'):
        values, gradients = f(x), slope(f, x)
    assert values[0] == 2 and abs(gradients[0]) < 1e-45
    assert np.all((values > 0) & (values <= 2))
    assert np.all(gradients[1:] < 0)
    np.testing.assert_array_equal(values[x >= 1], 1/x[x >= 1])
    # Independent comparison to the unbounded-input continuation at A>0.
    a = np.linspace(.001, .999, 201)
    t = 1/a-1
    np.testing.assert_allclose(f(a), 2-1/(1+t+t*t), rtol=2e-15)
    h = 1e-6
    points = np.array([1-h, 1., 1+h])
    np.testing.assert_allclose(f(points), 1., atol=1.1*h)
    np.testing.assert_allclose(slope(f, points), -1., atol=2.1*h)
    np.testing.assert_allclose(np.diff(slope(f, points))/h, 2., rtol=2e-5)
    np.testing.assert_allclose(slope(f, h)/h, -2., rtol=4e-6)
    assert np.isnan(f(-.01)) and np.isnan(f(np.inf))


@pytest.mark.parametrize('ar', [0., .01, .2, .8, 1., 2.])
def test_zero_height_and_zero_gap_are_regular_jointly(ar):
    flow, geom = inputs(ar=ar)
    # Independent gap/chord parameter; no gap/height at H=0.
    def fn(a, x):
        return ko.get_tip_clearance_loss(flow, dict(geom, height=.05*a, tip_clearance=.05*x), 1)
    for x in (0., .0005, .001, .02):
        with np.errstate(all='raise'):
            value = fn(ar, x)
            da = slope(lambda a: fn(a, x), ar)
            dx = slope(lambda q: fn(ar, q), x)
        assert np.isfinite(value) and np.isfinite(da) and np.isfinite(dx)
        if x == 0:
            assert value == 0 and da == 0 and dx > 0
        if ar == 0:
            assert abs(da) < 1e-45
            np.testing.assert_allclose(value, .37*ko.get_tip_clearance_loading(-30, 60)*2
                                       *ko.get_tip_clearance_gap_factor(x), rtol=2e-15)
        h = 1e-7
        fd = ((-3*fn(ar, x)+4*fn(ar+h, x)-fn(ar+2*h, x))/(2*h) if ar == 0 else
              (fn(ar+h, x)-fn(ar-h, x))/(2*h))
        np.testing.assert_allclose(da, fd, atol=5e-9, rtol=2e-7)


def test_loading_c2_at_angle_continuation_joins():
    for fn, join in [(lambda b: ko.get_tip_clearance_loading(0., b), 80.),
                     (lambda b: ko.get_tip_clearance_loading(0., b), -80.),
                     (ko._tip_clearance_tangent, 80.),
                     (ko._tip_clearance_tangent, -80.)]:
        h = 1e-4
        left = (slope(fn, join)-slope(fn, join-h))/h
        right = (slope(fn, join+h)-slope(fn, join))/h
        np.testing.assert_allclose(left, right, rtol=4e-4, atol=2e-5)
        np.testing.assert_allclose(slope(fn, join+h), slope(fn, join-h), rtol=1e-4, atol=1e-5)


def test_clearance_and_secondary_share_physical_loading_with_separate_reference_angles():
    angle = np.r_[-1e6, -180., -90., np.linspace(-85., 85., 171), 90., 180., 1e6]
    tangent = ko.get_blade_tangent(angle)
    np.testing.assert_array_equal(ko._tip_clearance_tangent(angle), tangent)
    assert np.all(np.isfinite(tangent))
    assert np.all(np.diff(tangent) > 0)
    assert np.all(np.abs(tangent) <= np.tan(np.deg2rad(87.5)))
    np.testing.assert_allclose(ko.get_blade_angle(np.array([-1e8, 1e8])), [-87.5, 87.5], atol=1e-10)
    core = abs(angle) <= 80.
    np.testing.assert_allclose(tangent[core], np.tan(np.deg2rad(angle[core])), rtol=3e-15, atol=1e-15)
    np.testing.assert_allclose(ko.get_blade_cosine(angle), 1/np.sqrt(1+tangent*tangent), rtol=2e-15)
    np.testing.assert_array_equal(ko.get_blade_loading(angle[:, None], angle[None, :]),
                                  ko.get_tip_clearance_loading(angle[:, None], angle[None, :]))


def test_physical_extreme_angles_and_explicit_nonphysical_domain():
    angles = np.r_[-89.9, np.linspace(-85, 85, 101), 89.9]
    fn = ko.get_tip_clearance_loading
    bi, bo = angles[:, None], angles[None, :]
    with np.errstate(all='raise'):
        value = fn(bi, bo)
        d_in = slope(lambda v: fn(v, bo), bi)
        d_out = slope(lambda v: fn(bi, v), bo)
    assert np.all(np.isfinite(value)) and np.all(value >= 0)
    # Ordinary angles may exceed either former loading threshold unchanged.
    inlet = np.array([60., 70., 80.])
    tangent = np.tan(np.deg2rad(inlet))
    expected = 4*tangent**2*np.sqrt(1+tangent**2/4)
    np.testing.assert_allclose(fn(inlet, 0.), expected, rtol=3e-15)
    assert expected[-1] > 32
    assert np.all(np.isfinite(d_in)) and np.all(np.isfinite(d_out))
    np.testing.assert_array_equal(fn(angles, angles), 0.)
    np.testing.assert_array_equal(fn(-bi, -bo), value)
    # Physical high-angle impulse tends to 16: no tangent saturation.
    impulse = fn(-angles, angles)
    assert np.all(impulse[abs(angles) >= 85] > 15)
    assert np.all(impulse <= 16+1e-13)
    assert slope(lambda v: fn(0., v), 85.) > 0
    assert fn(0., 0.) == slope(lambda v: fn(v, 0.), 0.) == slope(lambda v: fn(0., v), 0.) == 0
    for b in (85., 88.):
        h = 1e-3
        for path in (lambda v: fn(-.5*v, v), lambda v: fn(v, 30.)):
            fd = (path(b-2*h)-8*path(b-h)+8*path(b+h)-path(b+2*h))/(12*h)
            np.testing.assert_allclose(slope(path, b), fd, rtol=1e-7, atol=1e-9)
    assert np.isnan(fn(np.inf, 0.)) and np.isnan(fn(0., np.nan))
    for b in (-1e6, -90., 90., 120., 1e6):
        assert np.isnan(fn(b, 0.)) and np.isnan(fn(0., b))


def test_gap_slope_geometry_gradients_and_length_units():
    f, g = inputs()
    y = ko.get_tip_clearance_loss(f, g, 1)
    # At fixed physical gap, exponent of chord is +0.22; height is -1.
    for key, expected in [('chord', .22*y/g['chord']), ('height', -y/g['height']),
                          ('tip_clearance', .78*y/g['tip_clearance'])]:
        fn = lambda x: ko.get_tip_clearance_loss(f, dict(g, **{key: x}), 1)
        np.testing.assert_allclose(slope(fn, g[key]), expected, rtol=3e-14)
    for scale in (1e-3, 1e3):
        np.testing.assert_allclose(ko.get_tip_clearance_loss(f, {k: v*scale for k, v in g.items()}, 1), y, rtol=3e-15)
    for ratio in (0., .0001, .0005, .001, .003, .03):
        fn = lambda x: ko.get_tip_clearance_loss(f, dict(g, tip_clearance=x*g['chord']), 1)
        h = 1e-8
        finite = ((-3*fn(0.)+4*fn(h)-fn(2*h))/(2*h) if ratio == 0 else
                  (fn(ratio-2*h)-8*fn(ratio-h)+8*fn(ratio+h)-fn(ratio+2*h))/(12*h))
        np.testing.assert_allclose(slope(fn, ratio), finite, rtol=1e-8, atol=1e-9)


@pytest.mark.parametrize('z', [-1., -.5, 0., .5, 1.])
def test_fixed_z_angle_path_gradients(z):
    def fn(angle):
        return ko.get_tip_clearance_loss(*inputs(angle=angle, z=z), 1)
    angle = np.array([0., 20., 40., 60., 80.])
    h = 1e-3
    finite = (fn(angle-2*h)-8*fn(angle-h)+8*fn(angle+h)-fn(angle+2*h))/(12*h)
    np.testing.assert_allclose(slope(fn, angle), finite, rtol=1e-8, atol=1e-10)
    assert fn(0.) == 0 and slope(fn, 0.) == 0


@pytest.mark.parametrize('model', ['ko', 'mo', 'be'])
@pytest.mark.parametrize('row', [0, 1])
def test_complete_model_assembly_inherits_clearance(model, row, blade_row_case):
    case = blade_row_case
    f, g = case['flow'], case['geometry']
    g['tip_clearance'] = .0005
    compute = {'ko': ko.compute_losses, 'mo': mo.compute_losses, 'be': be.compute_losses}[model]
    def evaluate(geometry):
        return compute(dict(case, geometry=dict(geometry, cascade_type=('stator', 'rotor')[row])))
    result = evaluate(g)
    zero = evaluate(dict(g, tip_clearance=0.))
    expected = ko.get_tip_clearance_loss(f, g, row)
    np.testing.assert_allclose(result['loss_clearance'], expected, rtol=1e-15)
    np.testing.assert_allclose(result['loss_total']-zero['loss_total'], expected, rtol=1e-13, atol=1e-15)
    for component in ('profile', 'secondary', 'trailing', 'incidence'):
        assert result['loss_'+component] == zero['loss_'+component]
