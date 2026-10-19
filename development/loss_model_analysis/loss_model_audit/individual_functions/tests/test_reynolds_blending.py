"""Behavioral contracts for the deliberately broader KO Reynolds transitions."""

from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from cases import manufactured
from common import modules


def correction(reynolds):
    return modules("candidate")["ko"].get_reynolds_correction_factor(reynolds)


def test_exact_outer_power_laws_and_plateau():
    low = np.geomspace(1e4, 2e5 / 1.5, 300)
    plateau = np.linspace(3e5, 1e6 / 1.5, 300)
    high = np.geomspace(1.5e6, 1e7, 300)
    np.testing.assert_allclose(correction(low), (low / 2e5) ** -0.4, rtol=1e-14)
    np.testing.assert_array_equal(correction(plateau), np.ones_like(plateau))
    np.testing.assert_allclose(correction(high), (high / 1e6) ** -0.2, rtol=1e-14)


def test_broad_transitions_are_positive_monotone_and_bounded():
    reynolds = np.geomspace(1e4, 1e7, 20001)
    values = correction(reynolds)
    assert np.all(np.isfinite(values)) and np.all(values > 0)
    assert np.all(np.diff(values) <= 1e-14)
    hard = np.where(
        reynolds < 2e5,
        (reynolds / 2e5) ** -0.4,
        np.where(reynolds <= 1e6, 1.0, (reynolds / 1e6) ** -0.2),
    )
    relative = values / hard
    assert np.min(relative) >= 1.5 ** -0.0375 - 1e-14
    assert np.max(relative) <= 1.5 ** 0.075 + 1e-14
    np.testing.assert_allclose(correction(2e5), 1.5 ** 0.075, rtol=1e-14)
    np.testing.assert_allclose(correction(1e6), 1.5 ** -0.0375, rtol=1e-14)

    # These points were outside the old exp(+/-0.05) bands. The revised
    # transitions must have meaningful width on both sides of each threshold.
    assert correction(2e5 / 1.25) > 1.25 ** 0.4
    assert correction(2e5 * 1.25) > 1.0
    assert correction(1e6 / 1.25) < 1.0
    assert correction(1e6 * 1.25) < 1.25 ** -0.2


@pytest.mark.parametrize(
    "boundary,expected_log_slope",
    [(2e5 / 1.5, -0.4), (3e5, 0.0), (1e6 / 1.5, 0.0), (1.5e6, -0.2)],
)
def test_boundary_value_slope_and_curvature_match_outer_branches(boundary, expected_log_slope):
    # Differentiating ln(f) against ln(Re) avoids tiny dimensional gradients.
    # Smoothness of the positive exponential preserves these C2 joins in Re.
    def log_value(offset):
        return np.log(correction(boundary * np.exp(offset)))

    for step in (1e-3, 3e-4):
        value = log_value(0.0)
        for direction in (-1, 1):
            f1, f2, f3 = [log_value(direction * k * step) for k in (1, 2, 3)]
            slope = direction * (-3 * value + 4 * f1 - f2) / (2 * step)
            curvature = (2 * value - 5 * f1 + 4 * f2 - f3) / step**2
            np.testing.assert_allclose(slope, expected_log_slope, atol=3e-6, rtol=0)
            np.testing.assert_allclose(curvature, 0.0, atol=3e-5, rtol=0)


@pytest.mark.parametrize("threshold,min_slope", [(2e5, -0.4), (1e6, -0.2)])
def test_logarithmic_slopes_stay_between_empirical_limits(threshold, min_slope):
    offsets = np.linspace(-np.log(1.5), np.log(1.5), 10001)
    values = np.log(correction(threshold * np.exp(offsets)))
    slopes = np.diff(values) / np.diff(offsets)
    assert np.min(slopes) >= min_slope - 1e-10
    assert np.max(slopes) <= 1e-10
    center_slope = (values[5001] - values[4999]) / (offsets[5001] - offsets[4999])
    np.testing.assert_allclose(center_slope, min_slope / 2, atol=1e-10)


def test_array_like_inputs_and_invalid_domain():
    values = [[1e5, 2e5], [1e6, 2e6]]
    actual = correction(values)
    assert actual.shape == (2, 2)
    for index in np.ndindex(actual.shape):
        assert actual[index] == correction(np.asarray(values)[index])
    with np.errstate(divide="ignore", invalid="ignore"):
        assert np.all(np.isfinite(correction([-1, 0])))
        assert np.all(~np.isfinite(correction([np.nan, np.inf, -np.inf])))


def test_profile_loss_applies_the_broader_correction():
    case = manufactured()
    flow = deepcopy(case["flow"])
    ko = modules("candidate")["ko"]
    flow["Re_out"] = 5e5
    unit_reynolds_loss = ko.get_profile_loss(flow, case["geometry"], 0)
    flow["Re_out"] = np.array([1e5, 1.6e5, 2e5, 2.5e5, 5e5, 8e5, 1e6, 1.25e6, 2e6, 1e8, 1e12])
    actual = ko.get_profile_loss(flow, case["geometry"], 0)
    np.testing.assert_allclose(actual, unit_reynolds_loss * correction(flow["Re_out"]), rtol=1e-14)


def test_high_reynolds_factor_has_requested_nonzero_limit():
    ko = modules("candidate")["ko"]
    reynolds = np.geomspace(1e7, 1e100, 1001)
    values = correction(reynolds)
    expected_limit = (5e7/1e6)**(-.2)
    assert np.isfinite(values).all()
    assert np.all(np.diff(values) <= 1e-14)
    assert np.all(values >= expected_limit-1e-15)
    np.testing.assert_allclose(values[-1], expected_limit, rtol=2e-15)
    np.testing.assert_allclose(ko.get_effective_reynolds_number(reynolds[-1]), 5e7, rtol=2e-15)
    # The limit is approached gradually; there is no finite hard threshold.
    assert correction(5e7) > expected_limit
    assert correction(1e8) > correction(1e9) > expected_limit


def test_high_reynolds_join_and_derivatives():
    ko = modules("candidate")["ko"]
    # Dimensionless coordinate keeps first/second derivative checks well scaled.
    effective = lambda q: ko.get_effective_reynolds_number(1e7*q)/1e7
    derivative = lambda q: np.imag(effective(q+1e-25j))/1e-25
    h = 1e-5
    np.testing.assert_allclose(derivative(np.array([1-h,1.,1+h])), 1., atol=1e-10)
    np.testing.assert_allclose([(derivative(1)-derivative(1-h))/h,
                               (derivative(1+h)-derivative(1))/h], 0., atol=1e-5)
    function = lambda q: correction(1e7*q)
    for q in (1.,1.01,2.,5.,20.,100.):
        h = 1e-4*q
        finite = (function(q-2*h)-8*function(q-h)+8*function(q+h)-function(q+2*h))/(12*h)
        exact = np.imag(function(q+1e-25j))/1e-25
        np.testing.assert_allclose(exact, finite, rtol=2e-7, atol=2e-11)
        assert exact < 0
    slope = lambda q: np.imag(np.log(function(np.exp(q+1e-25j))))/1e-25
    np.testing.assert_allclose(slope(np.array([-1e-5,0.,1e-5])), -.2, atol=2e-9)


@pytest.mark.parametrize("row", [0,1])
def test_complete_profile_keeps_shock_at_large_reynolds(row):
    ko = modules("candidate")["ko"]
    case = manufactured()
    flow = dict(case["flow"], Ma_rel_in=.8, Re_out=5e5,
                p0_rel_in=2e5, p_in=1e5, p0_rel_out=2e5, p_out=1e5)
    geometry = dict(case["geometry"], hub_tip_ratio_in=-.5)
    unit_profile = ko.get_profile_loss(flow,geometry,row)
    unit_shock = .914*ko.get_shock_loss(flow,geometry,row)["Y_shock"]
    extreme = dict(flow,Re_out=1e100)
    profile = ko.get_profile_loss(extreme,geometry,row)
    no_shock = ko.get_profile_loss(dict(extreme,p_in=extreme["p0_rel_in"]),geometry,row)
    np.testing.assert_allclose(profile,unit_profile*50**(-.2),rtol=2e-14)
    np.testing.assert_allclose(profile-no_shock,unit_shock*50**(-.2),rtol=2e-14)
    assert profile > no_shock > 0
