"""Approved shock correlation, production integration and complex-step gradients."""
import numpy as np
import pytest
from turbodash.loss_models import loss_model_kacker_okapuu as ko


def inputs(mach=.6, ratio=.5, reynolds=5e5):
    flow = dict(Ma_rel_in=mach, Ma_rel_out=.6, Re_out=reynolds, beta_out=60.,
                p0_rel_in=2e5, p_in=1.6e5, p0_rel_out=1.8e5, p_out=1.3e5)
    geometry = dict(hub_tip_ratio_in=ratio, chord=.05, pitch=.035,
                    maximum_thickness=.01, leading_edge_angle=0.)
    return flow, geometry


@pytest.mark.parametrize("row", [0, 1])
def test_chart_domain_matches_independent_ko_expression(row):
    f, g = inputs(np.linspace(.5, 1.2, 91)[None, :], np.array([.5, .6, 1.])[:, None])
    actual = ko.get_shock_loss(f, g, row)
    amplitude,end = ko.HUB_FIT[row]
    factor = 1+amplitude*np.maximum(end-g["hub_tip_ratio_in"],0)**2
    expected_raw = .75*(factor*f["Ma_rel_in"]-.4)**1.75*g["hub_tip_ratio_in"]*.8
    np.testing.assert_allclose(actual["f_hub"],np.broadcast_to(factor,actual["f_hub"].shape),rtol=2e-14)
    np.testing.assert_allclose(actual["Y_shock_raw"],expected_raw,rtol=2e-14)
    low = expected_raw <= .25
    np.testing.assert_allclose(actual["Y_shock"][low], expected_raw[low], rtol=2e-14, atol=1e-16)
    assert np.all((actual["Y_shock"] >= 0) & (actual["Y_shock"] <= .7))
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
    np.testing.assert_allclose(dhub(.5-1e-7), dhub(.5+1e-7), atol=3e-6)
    np.testing.assert_allclose((dhub(.5)-dhub(.5-1e-6))/1e-6,
                               (dhub(.5+1e-6)-dhub(.5))/1e-6, atol=2e-3)


@pytest.mark.parametrize("row", [0, 1])
def test_complete_shock_complex_step_gradients(row):
    def function(x):
        m, ratio, q1, q2 = x
        f, g = inputs(m, ratio)
        f.update(p_in=f["p0_rel_in"]-q1, p_out=f["p0_rel_out"]-q2)
        return ko.get_shock_loss(f, g, row)["Y_shock"]
    for ratio in (-1., -.1, 0., .15, .3, .5, .6):
        x = np.array([.45/ko.get_hub_to_mean_mach_ratio(ratio, row), ratio, 4e4, 5e4])
        for i, h in enumerate((1e-6, 1e-6, 1., 1.)):
            imaginary = np.zeros(4, dtype=complex); imaginary[i] = 1e-25j
            offset = np.zeros(4); offset[i] = h
            exact = np.imag(function(x+imaginary))/1e-25
            finite = (function(x-2*offset)-8*function(x-offset)+8*function(x+offset)-function(x+2*offset))/(12*h)
            np.testing.assert_allclose(exact, finite, rtol=2e-8, atol=2e-12)


def test_invalid_domains_return_nan():
    for changes, ratio in (({"p_in":2.1e5}, .5), ({"p_out":1.8e5}, .5),
                           ({"p_out":1.9e5}, .5), ({"Ma_rel_in":-.1}, .5), ({}, np.nan), ({}, 1.1)):
        f, g = inputs(ratio=ratio)
        f.update(changes)
        assert np.isnan(ko.get_shock_loss(f, g, 0)["Y_shock"])
        with pytest.raises(ValueError): ko.validate_shock_inputs(f, g, 0)


def test_upper_continuation_preserves_low_losses_and_has_c2_join():
    low = np.linspace(0., .25, 51)
    np.testing.assert_array_equal(ko.continue_shock_loss(low), low)
    values = ko.continue_shock_loss(np.r_[low, np.geomspace(.250001, 1e100, 101)])
    assert np.all(np.isfinite(values)) and np.all(np.diff(values) >= 0)
    assert np.all((values >= 0) & (values <= .7))
    np.testing.assert_allclose(ko.continue_shock_loss(np.array([.7, 1.6, 8.8])),
                               [.7-.45/3, .7-.45/13, .7-.45/381], rtol=2e-15)
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
    assert np.all((result["Y_shock"] >= 0) & (result["Y_shock"] <= .7))
    assert np.all(np.diff(result["Y_shock"]) <= 1e-15)
    np.testing.assert_allclose(result["Y_shock"][0], .7, atol=1e-12)
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


@pytest.mark.parametrize("row", [0, 1])
def test_signed_geometry_uses_one_coordinate_for_mach_and_span(row):
    ratios = np.array([-1e100, -1., -.1, 0., .1, .3, .5, .8, 1.])
    f, g = inputs(.8, ratios)
    ko.validate_shock_inputs(f, g, row)
    result = ko.get_shock_loss(f, g, row)
    effective = result["effective_hub_tip_ratio"]
    assert np.isfinite(result["Y_shock"]).all()
    assert np.all(result["Y_shock"] > 0)
    np.testing.assert_array_equal(result["hub_tip_ratio"], ratios)
    np.testing.assert_allclose(effective[:1], .3, atol=1e-15)
    np.testing.assert_allclose(effective[ratios == 0], .3+.2/(1+2.5+2.5**2), rtol=1e-15)
    np.testing.assert_array_equal(effective[ratios >= .5], ratios[ratios >= .5])
    np.testing.assert_allclose(result["Y_shock_mean_inlet"], result["Y_shock_hub"]*effective)
    # Independent unrounded chart formula at low ratios catches double mapping.
    amplitude, end = ko.HUB_FIT[row]
    np.testing.assert_allclose(result["f_hub"][:6], 1+amplitude*(end-effective[:6])**2)
    map_slope = lambda q: np.imag(ko.effective_hub_tip_ratio(q+1e-25j))/1e-25
    np.testing.assert_allclose([map_slope(.5-1e-6),map_slope(.5),map_slope(.5+1e-6)],1.,atol=1e-9)
