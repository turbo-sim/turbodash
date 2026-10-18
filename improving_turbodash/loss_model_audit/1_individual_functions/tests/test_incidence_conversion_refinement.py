"""Reference limits, domain gaps and local smoothness of revised candidates."""

from pathlib import Path
import sys

import numpy as np
import pytest

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE / "scripts"))
from common import modules


@pytest.mark.parametrize("gamma", [1 + 1e-10, 1.1, 1.33, 1.4])
def test_unrestricted_conversion_roundtrip_including_small_mach(gamma):
    c = modules("candidate")["conversion"]
    mach = np.array([0, 1e-12, 1e-8, 1e-4, 0.2, 0.7, 2, 4])[:, None]
    pressure_loss = np.array([-1e-7, 0, 1e-12, 0.05, 1, 20])[None, :]
    energy = c.convert_stagnation_pressure_to_kinetic_energy_loss(
        mach, gamma, pressure_loss
    )
    actual = c.convert_kinetic_energy_to_stagnation_pressure_loss(
        mach, gamma, energy, limit_output=False
    )
    np.testing.assert_allclose(
        actual, np.broadcast_to(pressure_loss, actual.shape), rtol=2e-13, atol=1e-18
    )


def test_conversion_reference_limits_and_no_small_loss_offset():
    c = modules("candidate")["conversion"]
    d = np.array([-1e-7, 0, 1e-12, 0.01, 0.2])
    np.testing.assert_array_equal(
        c.convert_kinetic_energy_to_stagnation_pressure_loss(0, 1.4, d),
        d / (1 - d),
    )
    np.testing.assert_array_equal(
        c.convert_stagnation_pressure_to_kinetic_energy_loss(0, 1.4, d),
        d / (1 + d),
    )
    mach = np.linspace(0, 4, 100)
    np.testing.assert_array_equal(
        c.convert_kinetic_energy_to_stagnation_pressure_loss(mach, 1.4, 0), 0
    )
    np.testing.assert_array_equal(
        c.convert_stagnation_pressure_to_kinetic_energy_loss(mach, 1.4, 0), 0
    )


def test_enthalpy_wrappers_retain_zero_mach_reference_and_domain_gaps():
    c = modules("candidate")["conversion"]
    zeta = np.array([-1e-7, 0, 1e-12, 0.05, 0.2])
    np.testing.assert_allclose(
        c.convert_enthalpy_to_stagnation_pressure_loss(0, 1.4, zeta),
        zeta,
        rtol=2e-15,
        atol=0,
    )
    np.testing.assert_allclose(
        c.convert_stagnation_pressure_to_enthalpy_loss(0, 1.4, zeta),
        zeta,
        rtol=2e-15,
        atol=0,
    )
    for invalid in [-2, -1, np.inf, np.nan]:
        assert np.isnan(
            c.convert_enthalpy_to_stagnation_pressure_loss(0.7, 1.4, invalid)
        )


def test_conversion_matches_original_identity_away_from_cancellation_and_caps():
    c = modules("candidate")["conversion"]
    mach = np.array([0.2, 0.7, 1.2, 2.0])[:, None]
    d = np.array([0.001, 0.05, 0.2])[None, :]
    gamma = 1.4
    exponent = (gamma - 1) / gamma
    pressure_ratio = (1 + (gamma - 1) / 2 * mach**2) ** (-1 / exponent)
    expected = (
        ((1 - d * pressure_ratio ** (-exponent)) / (1 - d)) ** (-1 / exponent) - 1
    ) / (1 - pressure_ratio)
    actual = c.convert_kinetic_energy_to_stagnation_pressure_loss(
        mach, gamma, d, limit_output=False
    )
    np.testing.assert_allclose(actual, expected, rtol=3e-11, atol=1e-14)


def test_conversion_cap_and_mach_guard_are_optional_and_exposed():
    c = modules("candidate")["conversion"]
    assert (
        c.convert_kinetic_energy_to_stagnation_pressure_loss(0.7, 1.4, 0.6, False) > 1
    )
    assert c.convert_kinetic_energy_to_stagnation_pressure_loss(0.7, 1.4, 0.6) == 1
    diagnostic = c.get_kinetic_energy_conversion_diagnostics(2, 1.4, 0.9)
    assert diagnostic["domain_margin"] < 0
    assert np.isnan(diagnostic["Y_raw"])
    assert np.isfinite(diagnostic["Y_guarded"])
    assert diagnostic["Y_limited"] == 1
    assert 0 < diagnostic["Mach_effective"] < 2
    d = np.linspace(0, 0.999, 1001)
    for mach in [0, 0.2, 0.7, 2, 10]:
        y = c.convert_kinetic_energy_to_stagnation_pressure_loss(mach, 1.4, d)
        assert np.all(np.isfinite(y))
        assert np.all((y >= 0) & (y <= 1))
        assert np.min(np.diff(y)) >= -2e-14


@pytest.mark.parametrize(
    "mach,gamma,d",
    [
        (-1, 1.4, 0.2),
        (np.inf, 1.4, 0.2),
        (np.nan, 1.4, 0.2),
        (1, 1, 0.2),
        (1, 0.98, 0.2),
        (1, 1.4, 1),
        (1, 1.4, 1.1),
        (1, 1.4, np.nan),
    ],
)
def test_conversion_does_not_turn_invalid_inputs_into_finite_caps(mach, gamma, d):
    c = modules("candidate")["conversion"]
    assert np.isnan(
        c.convert_kinetic_energy_to_stagnation_pressure_loss(mach, gamma, d)
    )
    assert np.isnan(
        c.convert_kinetic_energy_to_stagnation_pressure_loss(mach, gamma, d, False)
    )


def test_benner_patch_preserves_outer_polynomials_and_reference():
    new = modules("candidate")["benner"]
    chi = np.r_[np.linspace(-10, -0.1, 201), np.linspace(0, 20, 401)]
    coefficients=[0.,-6.149e-5,1.327e-3,-2.506e-4,-1.542e-4,9.017e-5,1.106e-5,-5.318e-6,3.711e-7]
    polynomial=np.polynomial.polynomial.polyval(chi,coefficients)
    value_at_five=np.polynomial.polynomial.polyval(5.,coefficients)
    slope_at_five=np.polynomial.polynomial.polyval(5.,np.polynomial.polynomial.polyder(coefficients))
    expected=np.where(chi<0,-8.720e-4*chi+1.358e-4*chi**2,
                      np.where(chi<=5,polynomial,value_at_five+slope_at_five*(chi-5)))
    np.testing.assert_allclose(new.get_incidence_profile_loss_increment(chi,loss_limit=None),expected,rtol=2e-13,atol=1e-14)
    assert new.get_incidence_profile_loss_increment(0) == 0
    local = new.get_incidence_profile_loss_increment(np.linspace(-0.1, 0, 1001))
    assert np.all(local >= 0)
    assert np.all(np.diff(local) < 0)
    # A signed empirical increment is not silently clipped or recentered.
    assert new.get_incidence_profile_loss_increment(0.02) < 0
    capped = new.get_incidence_profile_loss_increment(np.linspace(0, 100, 1001))
    assert np.max(capped) == 0.5


def _one_sided_jets(function, x, h):
    offsets = np.arange(5) * h
    right = function(x + offsets)
    left = function(x - offsets)
    first_weights = np.array([-25, 48, -36, 16, -3]) / 12
    second_weights = np.array([35, -104, 114, -56, 11]) / 12
    return (
        -left @ first_weights / h,
        right @ first_weights / h,
        left @ second_weights / h**2,
        right @ second_weights / h**2,
    )


@pytest.mark.parametrize("join", [-0.1, 0])
def test_benner_patch_matches_first_and_second_derivatives(join):
    b = modules("candidate")["benner"]
    function = lambda x: b.get_incidence_profile_loss_increment(x, loss_limit=None)
    first_left, first_right, second_left, second_right = _one_sided_jets(
        function, join, 1e-4
    )
    assert abs(first_left - first_right) < 3e-10
    assert abs(second_left - second_right) < 2e-8


@pytest.mark.parametrize("raw_loss", [0.98, 1.02])
def test_conversion_cap_is_c2_at_both_transition_ends(raw_loss):
    c = modules("candidate")["conversion"]
    d = c.convert_stagnation_pressure_to_kinetic_energy_loss(0.7, 1.4, raw_loss)
    function = lambda x: c.convert_kinetic_energy_to_stagnation_pressure_loss(
        0.7, 1.4, x
    )
    first_left, first_right, second_left, second_right = _one_sided_jets(
        function, d, 1e-5
    )
    assert abs(first_left - first_right) < 1e-7
    assert abs(second_left - second_right) < 2e-3
