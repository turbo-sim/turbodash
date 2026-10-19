"""Fundamental candidate contracts; no complete-row or stage evaluations."""

from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from cases import manufactured
from common import modules


def candidate():
    return modules("candidate")


def one_sided_derivatives(fun, x, h):
    y = fun(x)
    left = [fun(x - k * h) for k in (1, 2, 3)]
    right = [fun(x + k * h) for k in (1, 2, 3)]
    d1 = (
        (3 * y - 4 * left[0] + left[1]) / (2 * h),
        (-3 * y + 4 * right[0] - right[1]) / (2 * h),
    )
    d2 = (
        (2 * y - 5 * left[0] + 4 * left[1] - left[2]) / h**2,
        (2 * y - 5 * right[0] + 4 * right[1] - right[2]) / h**2,
    )
    return np.asarray(d1), np.asarray(d2)


def assert_c2(fun, x, h=1e-4, atol=2e-4):
    d1, d2 = one_sided_derivatives(fun, x, h)
    assert np.all(np.isfinite([*d1, *d2]))
    np.testing.assert_allclose(d1[0], d1[1], atol=atol, rtol=2e-4)
    np.testing.assert_allclose(d2[0], d2[1], atol=atol, rtol=2e-3)


def test_compact_extrema_bounds_and_outer_identity():
    from turbodash.loss_models import _numerics as n

    x = np.linspace(-1, 1, 2001)
    p = n.positive_part(x, 0.1)
    assert np.all(p >= np.maximum(x, 0) - 1e-15)
    assert np.all(np.diff(p) >= -1e-15)
    np.testing.assert_array_equal(p[x <= -0.1], 0)
    np.testing.assert_array_equal(p[x >= 0.1], x[x >= 0.1])
    assert_c2(lambda value: n.positive_part(value, 0.1), -0.1)
    assert_c2(lambda value: n.positive_part(value, 0.1), 0.1)
    assert n.smooth_maximum(0, 0, half_width=0) == 0


def test_reynolds_factor_monotone_and_outer_exponents():
    ko = candidate()["ko"]
    reynolds = np.geomspace(1e4, 1e7, 1001)
    correction = ko.get_reynolds_correction_factor(reynolds)
    assert np.all(correction > 0)
    assert np.all(np.diff(correction) <= 1e-15)
    np.testing.assert_allclose(ko.get_reynolds_correction_factor(1e5), 2**0.4)
    np.testing.assert_allclose(ko.get_reynolds_correction_factor(5e5), 1)
    np.testing.assert_allclose(ko.get_reynolds_correction_factor(2e6), 2**-0.2)
    for threshold in (2e5, 1e6):
        for offset in (-np.log(ko.REYNOLDS_TRANSITION_FACTOR), np.log(ko.REYNOLDS_TRANSITION_FACTOR)):
            assert_c2(
                lambda z: ko.get_reynolds_correction_factor(threshold * np.exp(z)),
                offset,
            )


def test_compressibility_bounds_and_c2_joins():
    ko = candidate()["ko"]
    mach = np.linspace(0.01, 2, 1001)
    kp, _, k1 = ko.get_compressible_correction_factors(0.9, mach)
    assert np.all(kp >= 0.4+0.2/13 - 1e-14)
    assert np.all(k1 >= -1e-14) and np.all(k1 <= 1 + 1e-14)
    assert np.all(np.diff(k1) <= 1e-14)
    for join in (0.15, 0.25, 0.9, 1.0):
        assert_c2(
            lambda x: ko.get_compressible_correction_factors(0.3, x)[2],
            join,
            h=1e-5,
            atol=5e-4,
        )


def test_digitized_hub_fit_monotonicity_and_smooth_continuation():
    ko = candidate()["ko"]
    for cid in (0, 1):
        amplitude, end = ko.HUB_FIT[cid]
        limit = 1+amplitude*(end-.3)**2
        values = ko.get_hub_to_mean_mach_ratio(np.linspace(-1., 1, 601), cid)
        assert np.all(np.diff(values) <= 1e-14)
        assert np.min(values) >= 1 and np.max(values) <= limit+1e-12
        np.testing.assert_allclose(ko.get_hub_to_mean_mach_ratio(-1e100, cid), limit, rtol=2e-14)
        effective_at_zero = .3+.2/(1+2.5+2.5**2)
        np.testing.assert_allclose(ko.get_hub_to_mean_mach_ratio(0., cid),
                                   1+amplitude*(end-effective_at_zero)**2, rtol=2e-14)
        for join in (.5, end-.01, end+.01):
            d1, _ = one_sided_derivatives(lambda x: ko.get_hub_to_mean_mach_ratio(x, cid), join, 1e-6)
            np.testing.assert_allclose(d1[0], d1[1], atol=1e-4)


def test_zero_thickness_clearance_and_blockage_monotonicity():
    ko = candidate()["ko"]
    case = manufactured()
    f, g = case["flow"], deepcopy(case["geometry"])
    g["trailing_edge_thickness"] = 0
    g["tip_clearance"] = 0
    assert ko.get_trailing_edge_loss(f, g) == 0
    assert ko.get_tip_clearance_loss(f, g, 0) == 0
    assert ko.get_tip_clearance_loss(f, g, 1) == 0
    g["trailing_edge_thickness"] = g["opening"] * np.linspace(0, 0.6, 1001)
    trailing = ko.get_trailing_edge_loss(f, g)
    assert np.all(trailing >= 0) and np.all(np.diff(trailing) >= -1e-14)


@pytest.mark.parametrize("model", ["ko", "benner"])
def test_secondary_patch_positive_monotone_and_c2(model):
    mods = candidate()
    case = manufactured()
    f, g = case["flow"], deepcopy(case["geometry"])

    def loss(ar):
        g["height"] = g["chord"] * ar
        return (
            mods["ko"].get_secondary_loss(f, g, 0)
            if model == "ko"
            else mods["benner"].get_secondary_loss(f, g, 0.01)
        )

    lower, upper = (1.5, 2.5) if model == "ko" else (1.8, 2.2)
    values = np.asarray([loss(x) for x in np.linspace(lower, upper, 2001)])
    assert np.all(values > 0)
    assert np.all(np.diff(values) < 0)
    for join in (lower, 2, upper):
        assert_c2(loss, join)


def test_nozzle_and_impulse_fit_joins_and_spacing_optima():
    ko = candidate()["ko"]
    for fn in (ko.nozzle_blades, ko.impulse_blades):
        for angle in (59, 60, 61, 62, 63, 64):
            assert_c2(lambda value: fn(0.8, value), angle, h=2e-4, atol=1e-4)
        for angle in (40, 50, 60, 70, 80):
            # Current calibrated optimum, independent of the Aungier baseline.
            p = ko._AM_REFERENCE_COEFFICIENTS["nozzle" if fn is ko.nozzle_blades else "impulse"]
            t = (angle - (60 if fn is ko.nozzle_blades else 55)) / (20 if fn is ko.nozzle_blades else 15)
            center = np.exp(p[4] + p[5]*t + p[6]*t*t)
            values = fn(np.linspace(center - 0.1, center + 0.1, 1001), angle)
            assert np.all(values > 0)
            assert 0 < np.argmin(values) < len(values) - 1


def test_nozzle_spacing_gradients_remain_finite_through_angle_patch():
    ko = candidate()["ko"]
    for phi in (29.0, 29.5, 30.0, 30.5, 31.0):
        for optimum in (0.46 + phi / 77, 0.614 + phi / 130):
            fn = lambda spacing: ko.nozzle_blades(spacing, 90 - phi)
            estimates = []
            for step in (1e-3, 1e-4, 1e-5, 1e-6):
                slopes, _ = one_sided_derivatives(fn, optimum, step)
                estimates.append(slopes)
            assert np.all(np.isfinite(estimates))
            np.testing.assert_allclose(estimates[-1][0], estimates[-1][1], atol=1e-7)
            np.testing.assert_allclose(estimates[-1], estimates[-2], atol=1e-6)
