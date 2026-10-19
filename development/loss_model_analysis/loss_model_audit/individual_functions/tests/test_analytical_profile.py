"""Independent study parity, bounded continuation, and shared-model regression."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from common import modules
from cases import manufactured

spec = importlib.util.spec_from_file_location("standalone_am", ROOT / "development/loss_model_analysis/ainley_mathieson_fitting/analytic_fits.py")
standalone = importlib.util.module_from_spec(spec)
spec.loader.exec_module(standalone)


@pytest.mark.parametrize("family,upper", [("nozzle", 80.), ("impulse", 70.)])
def test_reference_values_broadcasting_and_derivatives_match_standalone(family, upper):
    fn = getattr(modules("candidate")["ko"], family + "_blades")
    xx = np.unique(np.r_[0., 1e-300, 1e-100, np.geomspace(1e-12, .3, 81), np.linspace(.3, 3., 401)])[:, None]
    aa = np.linspace(0., 85., 171)[None, :]
    expected = lambda x, a: standalone.profile_loss(x, a, family)
    np.testing.assert_allclose(fn(xx, aa), expected(xx, aa), rtol=4e-14, atol=5e-15)
    for x, a in ((.3, 40.), (.75, 60.), (0., upper), (3., upper)):
        assert np.ndim(fn(x, a)) == 0
        np.testing.assert_allclose(fn(x, a), expected(x, a), rtol=4e-14)
    for x, a in ((xx + 1e-25j, aa), (xx, aa + 1e-25j)):
        np.testing.assert_allclose(np.imag(fn(x, a))/1e-25, np.imag(expected(x, a))/1e-25, rtol=2e-10, atol=3e-12)
    assert np.all(np.isfinite(fn(xx, aa)))
    assert np.all((fn(xx, aa) > 0) & (fn(xx, aa) <= .5))
    np.testing.assert_array_equal(fn(0., aa), .5)
    np.testing.assert_allclose(np.imag(fn(1e-25j, aa))/1e-25, 0., atol=3e-12)
    with pytest.raises(ValueError):
        fn([0., -.1], 60.)


@pytest.mark.parametrize("family,upper", [("nozzle", 80.), ("impulse", 70.)])
def test_reference_low_spacing_order_and_c2_continuation(family, upper):
    fn = getattr(modules("candidate")["ko"], family + "_blades")
    alpha = np.linspace(40., upper, 1001)
    x = np.r_[0., np.geomspace(1e-12, .3, 151), np.linspace(.3, .5, 151)][:, None]
    slope = np.imag(fn(x, alpha + 1e-25j))/1e-25
    assert np.min(slope) >= -2e-10
    raw = lambda x, a: standalone.raw_profile_loss(x, a, family)
    for a in np.linspace(40., upper, 7):
        q = standalone._terms(a, family)[1]
        for root in (brentq(lambda x: raw(x, a)-.25, 1e-10, q), brentq(lambda x: raw(x, a)-.25, q, 10.)):
            assert fn(root, a) == pytest.approx(.25, abs=2e-12)
            d1 = lambda x: np.imag(fn(x + 1e-25j, a))/1e-25
            h = min(root, .1)*1e-6
            reference_d1 = np.imag(raw(root + 1e-25j, a))/1e-25
            reference_d2 = ((np.imag(raw(root+h+1e-25j, a))-np.imag(raw(root-h+1e-25j, a)))/1e-25)/(2*h)
            for sign in (-1., 1.):
                np.testing.assert_allclose(2*d1(root+sign*h)-d1(root+sign*2*h), reference_d1, rtol=2e-7, atol=1e-8)
                np.testing.assert_allclose((d1(root+sign*2*h)-d1(root+sign*h))/(sign*h), reference_d2, rtol=2e-4, atol=2e-5)


@pytest.mark.parametrize("model", ["ko", "mo", "benner"])
@pytest.mark.parametrize("row", ["stator", "rotor"])
def test_full_model_uses_the_standalone_reference_without_other_changes(monkeypatch, model, row):
    mods = modules("candidate")
    for pitch in (.3, .8, 1.7, 3.):
        case = manufactured()
        case["geometry"]["cascade_type"] = row
        case["geometry"]["pitch"] = pitch*case["geometry"]["chord"]
        snapshot = deepcopy(case)
        actual = mods[model].compute_losses(case)
        with monkeypatch.context() as patch:
            patch.setattr(mods["ko"], "nozzle_blades", standalone.nozzle_blades)
            patch.setattr(mods["ko"], "impulse_blades", standalone.impulse_blades)
            expected = mods[model].compute_losses(case)
        assert actual.keys() == expected.keys()
        for key in actual:
            if isinstance(actual[key], dict):
                assert actual[key].keys() == expected[key].keys()
                for diagnostic in actual[key]:
                    np.testing.assert_allclose(actual[key][diagnostic], expected[key][diagnostic], rtol=4e-13, atol=5e-14)
            else:
                np.testing.assert_allclose(actual[key], expected[key], rtol=4e-13, atol=5e-14)
        assert case == snapshot


def test_corrected_ko_profile_is_not_capped():
    ko = modules("candidate")["ko"]
    case = manufactured()
    case["flow"]["Re_out"] = 1e3
    case["geometry"]["pitch"] = 0.
    assert ko.get_profile_loss(case["flow"], case["geometry"], 0) > .5
