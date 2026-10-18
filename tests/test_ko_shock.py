"""Approved shock correlation, production integration and complex-step gradients."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
from turbodash.loss_models_new import loss_model_kacker_okapuu as ko


def inputs(mach=.6, ratio=.5, reynolds=5e5):
    flow = dict(Ma_rel_in=mach, Ma_rel_out=.6, Re_out=reynolds, beta_out=60.,
                p0_rel_in=2e5, p_in=1.6e5, p0_rel_out=1.8e5, p_out=1.3e5)
    geometry = dict(hub_tip_ratio_in=ratio, chord=.05, pitch=.035,
                    maximum_thickness=.01, leading_edge_angle=0.)
    return flow, geometry


@pytest.mark.parametrize("row", [0, 1])
def test_matches_approved_standalone_components(row):
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    path = Path(__file__).resolve().parents[1]/"improving_turbodash/kacker_okapuu_shock/shock_loss.py"
    spec = importlib.util.spec_from_file_location("approved_shock", path)
    approved = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(approved)
    f, g = inputs(np.linspace(0, 1.2, 91)[None, :], np.array([.01, .15, .3, .5, .8, 1.])[:, None])
    actual, expected = ko.get_shock_loss(f, g, row), approved.get_shock_loss(f, g, row)
    assert actual.keys() == expected.keys()
    for key in actual:
        np.testing.assert_allclose(actual[key], expected[key], rtol=2e-12, atol=2e-14)
    assert np.all((actual["Y_shock"] >= 0) & (actual["Y_shock"] <= .5))
    np.testing.assert_allclose(actual["Y_shock"], actual["Y_shock_mean_inlet"]*actual["head_ratio_effective"], rtol=2e-15)


@pytest.mark.parametrize("row", [0, 1])
@pytest.mark.parametrize("reynolds", [1e5, 5e5, 2e6])
def test_profile_assembles_shock_with_reynolds_and_outer_factor(row, reynolds):
    f, g = inputs(reynolds=reynolds)
    shock = ko.get_shock_loss(f, g, row)["Y_shock"]
    without = dict(f, p_in=f["p0_rel_in"])
    difference = ko.get_profile_loss(f, g, row)-ko.get_profile_loss(without, g, row)
    np.testing.assert_allclose(difference, .914*ko.get_reynolds_correction_factor(reynolds)*shock, rtol=2e-14)
    # The ideal/outlet pressure alias is no longer an inlet shock reference.
    np.testing.assert_array_equal(ko.get_profile_loss(dict(f, p0_rel_is=1.), g, row),
                                  ko.get_profile_loss(f, g, row))


def test_profile_calls_the_shared_helper(monkeypatch):
    f, g = inputs()
    without = ko.get_profile_loss(dict(f, p_in=f["p0_rel_in"]), g, 0)
    calls = []
    def fixed_shock(flow, geometry, row):
        calls.append((flow, geometry, row))
        return {"Y_shock": .123}
    monkeypatch.setattr(ko, "get_shock_loss", fixed_shock)
    np.testing.assert_allclose(ko.get_profile_loss(f, g, 0)-without, .914*.123, rtol=2e-14)
    assert len(calls) == 1 and calls[0] == (f, g, 0)


@pytest.mark.parametrize("row", [0, 1])
def test_onset_c2_and_exact_zero(row):
    function = lambda mh: ko.shock_onset(mh)[1]
    derivative = lambda mh: np.imag(function(mh+1e-25j))/1e-25
    np.testing.assert_array_equal(function(np.array([0., .3, .4])), 0.)
    for boundary in (.4, .5):
        h = 1e-6
        d0 = derivative(boundary)
        np.testing.assert_allclose(derivative(boundary+np.array([-h, h])), d0, atol=3e-6)
        curvatures = [(d0-derivative(boundary-h))/h, (derivative(boundary+h)-d0)/h]
        np.testing.assert_allclose(curvatures[0], curvatures[1], atol=3e-3)
    upper = np.array([.5, .6, 1.2])
    np.testing.assert_allclose(function(upper), (upper-.4)**1.75, rtol=2e-15)
    hub = lambda r: ko.get_hub_to_mean_mach_ratio(r, row)
    dhub = lambda r: np.imag(hub(r+1e-25j))/1e-25
    assert abs(dhub(0.)) < 1e-20
    np.testing.assert_allclose(dhub(.3-1e-7), dhub(.3+1e-7), atol=3e-6)
    np.testing.assert_allclose((dhub(.3)-dhub(.3-1e-6))/1e-6,
                               (dhub(.3+1e-6)-dhub(.3))/1e-6, atol=2e-3)


@pytest.mark.parametrize("row", [0, 1])
def test_complete_shock_complex_step_gradients(row):
    def function(x):
        m, ratio, q1, q2 = x
        f, g = inputs(m, ratio)
        f.update(p_in=f["p0_rel_in"]-q1, p_out=f["p0_rel_out"]-q2)
        return ko.get_shock_loss(f, g, row)["Y_shock"]
    for ratio in (.15, .3, .6):
        x = np.array([.45/ko.get_hub_to_mean_mach_ratio(ratio, row), ratio, 4e4, 5e4])
        for i, h in enumerate((1e-6, 1e-6, 1., 1.)):
            imaginary = np.zeros(4, dtype=complex); imaginary[i] = 1e-25j
            offset = np.zeros(4); offset[i] = h
            exact = np.imag(function(x+imaginary))/1e-25
            finite = (function(x-2*offset)-8*function(x-offset)+8*function(x+offset)-function(x+2*offset))/(12*h)
            np.testing.assert_allclose(exact, finite, rtol=2e-8, atol=2e-12)


def test_invalid_domains_return_nan():
    for changes, ratio in (({"p_in":2.1e5}, .5), ({"p_out":1.8e5}, .5),
                           ({"p_out":1.9e5}, .5), ({"Ma_rel_in":-.1}, .5), ({}, 0.), ({}, 1.1)):
        f, g = inputs(ratio=ratio)
        f.update(changes)
        assert np.isnan(ko.get_shock_loss(f, g, 0)["Y_shock"])
        with pytest.raises(ValueError): ko.validate_shock_inputs(f, g, 0)


def test_upper_continuation_preserves_low_losses_and_has_c2_join():
    low = np.linspace(0., .25, 51)
    np.testing.assert_array_equal(ko.continue_shock_loss(low), low)
    values = ko.continue_shock_loss(np.r_[low, np.geomspace(.250001, 1e100, 101)])
    assert np.all(np.isfinite(values)) and np.all(np.diff(values) >= 0)
    assert np.all((values >= 0) & (values <= .5))
    np.testing.assert_allclose(ko.continue_shock_loss(np.array([.5, 1., 5.])),
                               [.5-.25/3, .5-.25/13, .5-.25/381], rtol=2e-15)
    derivative = lambda y: np.imag(ko.continue_shock_loss(y+1e-25j))/1e-25
    h = 1e-7
    np.testing.assert_allclose([derivative(.25-h), derivative(.25), derivative(.25+h)], 1., atol=1e-10)
    np.testing.assert_allclose([(derivative(.25)-derivative(.25-h))/h,
                               (derivative(.25+h)-derivative(.25))/h], 0., atol=2e-5)
    assert 0 < derivative(100.) < 1e-6


@pytest.mark.parametrize("row", [0, 1])
def test_complete_shock_is_bounded_for_small_exit_heads(row):
    f, g = inputs(.8, .5)
    f["p_out"] = f["p0_rel_out"]-np.geomspace(1e-8, 1e5, 151)
    result = ko.get_shock_loss(f, g, row)
    assert np.all(np.isfinite(result["Y_shock"]))
    assert np.all((result["Y_shock"] >= 0) & (result["Y_shock"] <= .5))
    assert np.all(np.diff(result["Y_shock"]) <= 1e-15)
    np.testing.assert_allclose(result["Y_shock"][0], .5, atol=1e-12)
    np.testing.assert_allclose(result["Y_shock_raw"], result["Y_shock_mean_inlet"]*result["head_ratio"])
    np.testing.assert_allclose(result["Y_shock"], result["Y_shock_mean_inlet"]*result["head_ratio_effective"])
    assert np.all(result["head_ratio_effective"] <= result["head_ratio"]*(1+1e-14))


@pytest.mark.parametrize("row", [0, 1])
def test_complete_upper_branch_pressure_derivatives(row):
    f, g = inputs(.8, .5)
    numerator = float(ko.get_shock_loss(f, g, row)["Y_shock_mean_inlet"])*(f["p0_rel_in"]-f["p_in"])
    def function(q2):
        return ko.get_shock_loss(dict(f, p_out=1e5, p0_rel_out=1e5+q2), g, row)["Y_shock"]
    for raw in (.1, .25, .5, 2., 20.):
        q2 = numerator/raw
        exact = np.imag(function(q2+1e-25j))/1e-25
        h = q2*1e-4
        finite = (function(q2-2*h)-8*function(q2-h)+8*function(q2+h)-function(q2+2*h))/(12*h)
        np.testing.assert_allclose(exact, finite, rtol=3e-7, atol=1e-13)
        assert exact < 0
