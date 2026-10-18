"""Production Figure 14 data agreement, C2 joins, domains and assembly."""
import csv
import numpy as np
import pytest
from pathlib import Path
POINTS = Path(__file__).parent / "fixtures/ko_trailing_edge_points.csv"
from turbodash.loss_models._trailing_edge import (reference_loss, energy_loss, pressure_loss, trailing_loss,
                   effective_mach, COEFFICIENTS, REFERENCE_CEILINGS)


def slope(function, x):
    return np.imag(function(np.asarray(x, dtype=complex)+1e-25j))/1e-25


def assert_c2(function, join, h=1e-6, atol=2e-4):
    x = join+np.array([-h, 0., h])
    values = function(x)
    gradients = slope(function, x)
    np.testing.assert_allclose(values[[0, 2]], values[1], rtol=0, atol=2e-5)
    np.testing.assert_allclose(gradients[[0, 2]], gradients[1], rtol=0, atol=atol)
    curvature = np.diff(gradients)/h
    np.testing.assert_allclose(curvature[0], curvature[1], rtol=3e-3, atol=atol)


@pytest.mark.parametrize('family', ['nozzle', 'impulse'])
def test_digitized_and_held_out_accuracy(family):
    with POINTS.open(encoding='utf-8') as stream:
        rows = [r for r in csv.DictReader(stream) if r['family'] == family]
    x = np.array([float(r['thickness_opening']) for r in rows])
    y = np.array([float(r['energy_loss']) for r in rows])
    sigma = np.array([float(r['sigma_energy']) for r in rows])
    errors = abs(reference_loss(x, family)-y)
    assert len(rows) == 21
    assert np.all(errors < sigma)
    held_out = np.array([r['holdout'] == '1' for r in rows])
    assert held_out.sum() == 5
    assert np.max(errors[held_out]) < .001


@pytest.mark.parametrize('family', ['nozzle', 'impulse'])
def test_reference_join_and_bounded_monotone_tail(family):
    fn = lambda x: reference_loss(x, family)
    assert_c2(fn, .4, atol=2e-4)
    x = np.r_[0., np.geomspace(1e-12, 1e6, 3001)]
    y = fn(x)
    assert y[0] == 0 and np.all(y >= 0)
    assert np.all(y <= REFERENCE_CEILINGS[family])
    assert np.all(np.diff(y) >= -1e-15)
    assert np.all(slope(fn, x) >= 0)
    assert np.isfinite(fn(1e150))
    # Endpoint slope is finite and is the fitted linear coefficient.
    np.testing.assert_allclose(slope(fn, 0.), COEFFICIENTS[family][0], rtol=2e-15)


def test_nozzle_greater_than_impulse_and_blend_bounds():
    x = np.r_[0., np.geomspace(1e-10, 1e6, 401)][:, None]
    z = np.r_[-1e6, np.linspace(-4, 4, 501), 1e6][None, :]
    r, i = reference_loss(x, 'nozzle'), reference_loss(x, 'impulse')
    assert np.all(r >= i)
    d = energy_loss(x, z)
    assert np.all(d >= .8*i-1e-15) and np.all(d <= 1.2*r+1e-15)
    assert np.all(np.diff(d, axis=0) >= -1e-15)
    assert np.all(np.diff(d, axis=1) <= 1e-15)
    np.testing.assert_array_equal(energy_loss(x, 0.), r)
    np.testing.assert_allclose(energy_loss(x, 1.), i, rtol=3e-15, atol=1e-16)
    np.testing.assert_array_equal(energy_loss(0., z), 0.)
    np.testing.assert_allclose(energy_loss(x, -1e8), 1.2*r, rtol=3e-15, atol=1e-16)
    np.testing.assert_allclose(energy_loss(x, 1e8), .8*i, rtol=3e-15, atol=1e-16)


@pytest.mark.parametrize('join', [-1., -.05, 0., .05, 1.])
def test_angle_blend_c2_and_paper_identity(join):
    fn = lambda z: energy_loss(.25, z)
    assert_c2(fn, join, h=1e-7, atol=3e-4)
    z = np.linspace(.05, 1, 100)
    r, i = reference_loss(.25, 'nozzle'), reference_loss(.25, 'impulse')
    np.testing.assert_allclose(fn(z), (1-z*z)*r+z*z*i, rtol=3e-15)


def test_mach_continuation_c2_and_increasing_without_finite_plateau():
    assert_c2(effective_mach, .9, h=1e-7, atol=4e-5)
    m = np.linspace(0, 10, 10001)
    y = effective_mach(m)
    np.testing.assert_array_equal(y[m <= .9], m[m <= .9])
    assert np.all(np.diff(y) > 0) and np.all(slope(effective_mach, m) > 0)
    assert np.all(y <= 1)
    assert np.isfinite(effective_mach(1e150))


def test_pressure_conversion_against_equation_18_and_zero_mach_limit():
    d = np.linspace(0, .3, 51)[:, None, None]
    m = np.linspace(.01, 1., 31)[None, :, None]
    g = np.array([1.1, 1.2, 1.4, 5/3])[None, None, :]
    q, n = (g-1)*m*m/2, g/(g-1)
    expected = ((1-q*d/(1-d))**(-n)-1)/(1-(1+q)**(-n))
    np.testing.assert_allclose(pressure_loss(d, m, g, continue_mach=False), expected, rtol=2e-9, atol=2e-11)
    np.testing.assert_allclose(pressure_loss(d, 0.), d/(1-d), rtol=2e-15)
    np.testing.assert_allclose(pressure_loss(d, 1e-10), d/(1-d), rtol=2e-15)
    np.testing.assert_array_equal(slope(lambda v: pressure_loss(d, v), 0.), 0.)
    # gamma->1 has a finite analytic limit as well.
    a = d/(1-d)
    limit = np.expm1(.5*.7**2*a)/(-np.expm1(-.5*.7**2))
    np.testing.assert_allclose(pressure_loss(d, .7, 1+1e-10), limit, rtol=2e-10, atol=1e-15)


@pytest.mark.parametrize('variable', ['thickness', 'z', 'mach', 'gamma'])
def test_complete_loss_gradients_against_real_differences(variable):
    kwargs = dict(thickness_opening=.23, z=.65, mach=.7, gamma=1.4)
    key = {'thickness': 'thickness_opening', 'z': 'z', 'mach': 'mach', 'gamma': 'gamma'}[variable]
    fn = lambda x: trailing_loss(**dict(kwargs, **{key: x}))
    x, h = kwargs[key], 1e-4
    finite = (fn(x-2*h)-8*fn(x-h)+8*fn(x+h)-fn(x+2*h))/(12*h)
    np.testing.assert_allclose(slope(fn, x), finite, rtol=2e-8, atol=1e-11)


def test_full_loss_extreme_grid_and_domain():
    x = np.r_[0., np.geomspace(1e-12, 1e6, 101)][:, None, None, None]
    z = np.array([-1e6, -2., -1., -.05, 0., .05, .5, 1., 2., 1e6])[None, :, None, None]
    m = np.array([0., 1e-8, .1, .9, 1., 2., 10., 1e6])[None, None, :, None]
    g = np.array([1+1e-8, 1.1, 1.4, 5/3])[None, None, None, :]
    y = trailing_loss(x, z, m, g)
    assert y.shape == (102, 10, 8, 4)
    assert np.all(np.isfinite(y)) and np.all(y >= 0) and np.all(y < 1.)
    np.testing.assert_array_equal(y[0], 0.)
    for args in ((-.1, 0., .5, 1.4), (.1, 0., -.5, 1.4), (.1, 0., .5, 1.), (.1, 0., .5, 2.)):
        assert np.isnan(trailing_loss(*args))
    assert np.isnan(pressure_loss(.3, 5., 1.4, continue_mach=False))


def production_inputs(x=.2, z=.5, m=.7, gamma=1.4, beta=60.):
    return (dict(beta_out=beta, Ma_rel_out=m, gamma_out=gamma),
            dict(opening=.01, trailing_edge_thickness=.01*np.asarray(x),
                 leading_edge_angle=-np.asarray(beta)*z,
                 metal_angle_out=beta))


def test_wrapper_geometry_scaling_and_signed_angles():
    from turbodash.loss_models import loss_model_kacker_okapuu as ko
    x = np.linspace(0, 1.2, 121)[:, None]
    z = np.linspace(-2, 2, 81)[None, :]
    flow, geometry = production_inputs(x, z)
    np.testing.assert_allclose(ko.get_trailing_edge_kinetic_energy_loss(flow, geometry), energy_loss(x, z),
                               rtol=3e-15, atol=2e-16)
    np.testing.assert_allclose(ko.get_trailing_edge_loss(flow, geometry), trailing_loss(x, z, .7, 1.4),
                               rtol=3e-15, atol=2e-16)
    negative_flow, negative_geom = production_inputs(x, z, beta=-60.)
    np.testing.assert_array_equal(ko.get_trailing_edge_loss(negative_flow, negative_geom),
                                  ko.get_trailing_edge_loss(flow, geometry))
    for scale in (1e-6, 1e6):
        geom = dict(geometry, opening=geometry['opening']*scale,
                    trailing_edge_thickness=geometry['trailing_edge_thickness']*scale)
        np.testing.assert_allclose(ko.get_trailing_edge_loss(flow, geom), trailing_loss(x, z, .7, 1.4),
                                   rtol=3e-15, atol=2e-16)
    # Only metal angle, exit angle, thickness/opening, outlet Mach and gamma enter.
    np.testing.assert_array_equal(ko.get_trailing_edge_loss(dict(flow, Ma_rel_in=1e6, Re_out=1.,
                                        beta_in=123.), dict(geometry, chord=.001, height=1e6)),
                                  ko.get_trailing_edge_loss(flow, geometry))


def test_wrapper_invalid_domain_and_axial_angle_continuation():
    from turbodash.loss_models import loss_model_kacker_okapuu as ko
    f, g = production_inputs()
    with np.errstate(all='raise'):
        for key, value in [('opening', 0.), ('opening', -.01), ('opening', np.inf),
                           ('trailing_edge_thickness', -.01), ('leading_edge_angle', np.nan)]:
            assert np.isnan(ko.get_trailing_edge_loss(f, dict(g, **{key: value})))
        for key, value in [('beta_out', np.inf), ('Ma_rel_out', -.1),
                           ('Ma_rel_out', np.inf), ('gamma_out', 1.), ('gamma_out', 2.)]:
            assert np.isnan(ko.get_trailing_edge_loss(dict(f, **{key: value}), g))
        for bad in (np.nan, np.inf, -np.inf):
            assert np.isnan(trailing_loss(bad, .5, .7, 1.4))
            assert np.isnan(trailing_loss(.2, bad, .7, 1.4))
    # An axial case with theta=0 is still exactly the nozzle reference.
    f, g = production_inputs(beta=0.)
    np.testing.assert_allclose(ko.get_trailing_edge_loss(f, g), trailing_loss(.2, 0., .7, 1.4), rtol=2e-15)
    g['leading_edge_angle'] = -30.
    points = np.array([-1e-7, 0., 1e-7])
    fn = lambda angle: ko.get_trailing_edge_loss(dict(f, beta_out=angle), g)
    assert np.all(np.isfinite(fn(points)))
    # Fixed nonzero theta stays on its original large-positive-z branch.
    np.testing.assert_allclose(fn(0.), trailing_loss(.2, 9., .7, 1.4), rtol=2e-15)
    np.testing.assert_allclose(fn(points), fn(0.)+slope(fn, 0.)*points, atol=1e-11)
    np.testing.assert_allclose(slope(fn, points), slope(fn, 0.), atol=1e-5)
    # Prescribed z remains well defined through the lower-level function.
    assert np.isfinite(trailing_loss(.2, .5, .7, 1.4))


@pytest.mark.parametrize('theta', [-30., 30.])
@pytest.mark.parametrize('direction', [-1., 1.])
def test_trailing_preserves_blend_branch_across_zero_exit(theta, direction):
    from turbodash.loss_models import loss_model_kacker_okapuu as ko
    flow, geometry = production_inputs()
    geometry.update(leading_edge_angle=theta, metal_angle_out=60*direction)
    fn = lambda beta: ko.get_trailing_edge_loss(dict(flow, beta_out=beta), geometry)
    beta = np.array([-1e-4, 0., 1e-4])
    z = ko.get_cascade_angle_ratio(theta, beta, direction)
    assert np.all(np.sign(z) == np.sign(-theta/direction))
    values = fn(beta)
    np.testing.assert_allclose(values, trailing_loss(.2, z, .7, 1.4), rtol=2e-15)
    assert np.all(np.isfinite(values)) and np.all(values > 0)
    gradients = slope(fn, beta)
    assert np.all(np.isfinite(gradients))
    h = 1e-4
    finite = (fn(-2*h)-8*fn(-h)+8*fn(h)-fn(2*h))/(12*h)
    np.testing.assert_allclose(gradients[1], finite, rtol=2e-6, atol=1e-11)
    np.testing.assert_allclose(gradients[[0, 2]], gradients[1], atol=2e-6)
    assert not np.isclose(values[1], trailing_loss(.2, 0., .7, 1.4), rtol=.01)


@pytest.mark.parametrize('variable', ['opening', 'thickness', 'metal_angle', 'exit_angle', 'Mach', 'gamma'])
def test_production_wrapper_independent_gradients(variable):
    from turbodash.loss_models import loss_model_kacker_okapuu as ko
    flow, geometry = production_inputs()
    section, key = {'opening': ('g', 'opening'), 'thickness': ('g', 'trailing_edge_thickness'),
                    'metal_angle': ('g', 'leading_edge_angle'), 'exit_angle': ('f', 'beta_out'),
                    'Mach': ('f', 'Ma_rel_out'), 'gamma': ('f', 'gamma_out')}[variable]
    value = (geometry if section == 'g' else flow)[key]
    def fn(v):
        f = dict(flow, **{key: v}) if section == 'f' else flow
        g = dict(geometry, **{key: v}) if section == 'g' else geometry
        return ko.get_trailing_edge_loss(f, g)
    h = 1e-4*max(abs(value), .001)
    finite = (fn(value-2*h)-8*fn(value-h)+8*fn(value+h)-fn(value+2*h))/(12*h)
    np.testing.assert_allclose(slope(fn, value), finite, rtol=1e-7, atol=1e-10)


@pytest.mark.parametrize('model', ['ko', 'mo', 'be'])
@pytest.mark.parametrize('row', [0, 1])
def test_complete_assembly_inherits_trailing(model, row, blade_row_case):
    from turbodash.loss_models import loss_model_kacker_okapuu as ko
    from turbodash.loss_models import loss_model_moustapha as mo
    from turbodash.loss_models import loss_model_benner as be
    case = blade_row_case
    f, g = case['flow'], case['geometry']
    compute = {'ko': ko.compute_losses, 'mo': mo.compute_losses, 'be': be.compute_losses}[model]
    def evaluate(geometry):
        return compute(dict(case, geometry=dict(geometry, cascade_type=('stator', 'rotor')[row])))
    actual = evaluate(g)
    zero = evaluate(dict(g, trailing_edge_thickness=0.))
    expected = ko.get_trailing_edge_loss(f, g)
    if model == 'be':
        np.testing.assert_allclose(actual['diagnostics']['trailing_loss_unweighted'], expected, rtol=1e-15)
        expected = expected*actual['diagnostics']['primary_loss_weight']
    np.testing.assert_allclose(actual['loss_trailing'], expected, rtol=1e-15)
    np.testing.assert_allclose(actual['loss_total']-zero['loss_total'], expected, rtol=1e-13, atol=1e-15)
    for component in ('profile', 'secondary', 'clearance', 'incidence'):
        np.testing.assert_array_equal(actual['loss_'+component], zero['loss_'+component])
