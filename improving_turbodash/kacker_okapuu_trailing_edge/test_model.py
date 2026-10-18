"""Data agreement, endpoint matching, pressure conversion and extrapolation."""
import csv
import json
import numpy as np
import pytest
from digitize import BASE
from model import (reference_loss, energy_loss, pressure_loss, trailing_loss,
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
    with (BASE/'data/digitized_points.csv').open(encoding='utf-8') as stream:
        rows = [r for r in csv.DictReader(stream) if r['family'] == family]
    x = np.array([float(r['thickness_opening']) for r in rows])
    y = np.array([float(r['energy_loss']) for r in rows])
    sigma = np.array([float(r['sigma_energy']) for r in rows])
    assert np.all(abs(reference_loss(x, family)-y) < sigma)
    results = json.loads((BASE/'data/fit_parameters.json').read_text(encoding='utf-8'))
    candidate = next(r for r in results['candidate_fits'] if r['family'] == family and r['degree'] == 3)
    assert candidate['holdout_points'] == 5
    assert candidate['holdout_maximum_error'] < .001


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
