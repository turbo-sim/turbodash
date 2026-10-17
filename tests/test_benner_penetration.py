"""Physical penetration bound, smooth joins and primary-loss bookkeeping."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from turbodash import core_turbine as core
from turbodash.loss_models import loss_model_benner as benner


BASELINE = json.loads((Path(__file__).parent / "fixtures/continuous_geometry_baseline.json").read_text())
COMPONENTS = ("profile", "trailing", "incidence", "secondary", "clearance")


@pytest.mark.parametrize("width", [.01, .025, .05])
def test_limiter_bounds_monotonicity_and_unchanged_low_penetration(width):
    raw = np.r_[np.linspace(0, 2, 10001), np.finfo(float).max]
    effective = benner.limit_penetration_depth(raw, transition_half_width=width)
    assert np.isfinite(effective).all()
    assert (effective >= 0).all() and (effective <= .5).all()
    assert (effective <= raw).all()
    assert (np.diff(effective) >= -1e-15).all()
    np.testing.assert_array_equal(effective[raw <= .5 - width], raw[raw <= .5 - width])
    np.testing.assert_array_equal(effective[raw >= .5 + width], .5)
    assert benner.limit_penetration_depth(0, transition_half_width=width) == 0


def test_default_transition_and_c2_joins():
    assert benner.limit_penetration_depth(.5) == pytest.approx(.490625)
    assert benner.limit_penetration_depth(.45) == .45
    assert benner.limit_penetration_depth(.55) == .5
    for join, slope in ((.45, 1), (.55, 0)):
        h = 2e-5
        f = benner.limit_penetration_depth
        assert (f(join) - f(join - h)) / h == pytest.approx(slope, abs=1e-6)
        assert (f(join + h) - f(join)) / h == pytest.approx(slope, abs=1e-6)
        for direction in (-1, 1):
            curvature = (f(join + direction * 2 * h) - 2 * f(join + direction * h) + f(join)) / h**2
            assert abs(curvature) < .03


@pytest.mark.parametrize("raw", [-1, np.nan, np.inf, -np.inf, [0, np.nan]])
def test_invalid_predictions_are_not_hidden_by_saturation(raw):
    with pytest.raises(ValueError, match="Raw penetration ratio"):
        benner.limit_penetration_depth(raw)


@pytest.mark.parametrize("width", [0, -.1, .5, np.nan, np.inf, [0.05, 0.01], 1e-300])
def test_invalid_transition_width_is_rejected(width):
    with pytest.raises(ValueError, match="half-width"):
        benner.limit_penetration_depth(.6, transition_half_width=width)


@pytest.fixture
def high_penetration_inputs():
    inputs = deepcopy(BASELINE["eps80"]["inputs"])
    inputs["stages"][0]["degree_reaction"] = .75
    inputs["blade_velocity_ratio"] = .85
    return inputs


def test_common_primary_weight_raw_diagnostics_and_untouched_secondary(high_penetration_inputs):
    result = core.compute_turbine_performance({"inputs": high_penetration_inputs})
    stage = result["stages_performance"][0]
    losses = stage["losses"]["rotor"]
    diag = losses["diagnostics"]
    assert diag["penetration_raw_ratio"] == pytest.approx(.977945, abs=1e-6)
    assert diag["penetration_effective_ratio"] == .5
    assert diag["penetration_outside_model_domain"]
    assert not diag["penetration_smoothing_active"]
    assert diag["primary_loss_weight"] == .5
    for name in ("profile", "trailing", "incidence"):
        assert losses[f"loss_{name}"] == .5 * diag[f"{name}_loss_unweighted"]
    assert diag["primary_loss_weighted"] == pytest.approx(sum(losses[f"loss_{name}"] for name in ("profile", "trailing", "incidence")))
    assert losses["loss_total"] == pytest.approx(sum(losses[f"loss_{name}"] for name in COMPONENTS))
    inlet, outlet = stage["flow_stations"][2:]
    flow = {"beta_in": inlet["beta"], "beta_out": outlet["beta"]}
    assert losses["loss_secondary"] == benner.get_secondary_loss(flow, stage["geometry_meanline"]["rotor"], diag["inlet_displacement_thickness_height_ratio"])


@pytest.mark.parametrize("case", ["eps0", "eps80"])
def test_selected_design_loss_coefficients_remain_exactly_unchanged(case):
    result = core.compute_turbine_performance({"inputs": deepcopy(BASELINE[case]["inputs"])})
    for row in ("stator", "rotor"):
        losses = result["stages_performance"][0]["losses"][row]
        assert losses["diagnostics"]["penetration_raw_ratio"] < .45
        for name in COMPONENTS + ("total",):
            assert losses[f"loss_{name}"] == BASELINE[case]["continuous"]["stages_performance"][0]["losses"][row][f"loss_{name}"]


@pytest.mark.parametrize("key,value", [("height", 0), ("chord", -1), ("pitch", np.inf), ("meridional_chord", np.nan)])
def test_invalid_geometry_is_rejected(key, value):
    geometry = dict(BASELINE["eps0"]["continuous"]["stages_performance"][0]["geometry"]["rotor"])
    geometry[key] = value
    with pytest.raises(ValueError):
        benner.get_penetration_depth({"beta_in": 40., "beta_out": -60.}, geometry, .01)


@pytest.mark.parametrize("angle", [-90, 90, 100, np.nan, np.inf])
def test_invalid_angles_are_rejected(angle):
    geometry = BASELINE["eps0"]["continuous"]["stages_performance"][0]["geometry"]["rotor"]
    with pytest.raises(ValueError, match="flow angles"):
        benner.get_penetration_depth({"beta_in": angle, "beta_out": -60.}, geometry, .01)


def test_zero_loading_and_zero_boundary_layer_have_zero_raw_penetration():
    geometry = BASELINE["eps0"]["continuous"]["stages_performance"][0]["geometry"]["rotor"]
    assert benner.get_penetration_depth({"beta_in": 0., "beta_out": 0.}, geometry, 0.) == 0.


@pytest.mark.parametrize("reynolds", [0, -1, np.nan, np.inf])
def test_invalid_inlet_reynolds_is_rejected_before_correlations(reynolds):
    with pytest.raises(ValueError, match="Inlet Reynolds"):
        benner._compute_losses_benner_jit({"Re_in": reynolds}, {}, 0, .011, 0.)


@pytest.mark.parametrize("delta", [-1, np.nan, np.inf])
def test_invalid_reference_boundary_layer_is_rejected_before_correlations(delta):
    with pytest.raises(ValueError, match="Reference displacement"):
        benner._compute_losses_benner_jit({"Re_in": 3e5}, {}, 0, delta, 0.)


@pytest.mark.parametrize("delta", [-1, np.nan, np.inf])
def test_invalid_boundary_layer_is_not_hidden_by_the_limiter(delta):
    geometry = BASELINE["eps0"]["continuous"]["stages_performance"][0]["geometry"]["rotor"]
    with pytest.raises(ValueError, match="displacement-thickness"):
        benner.get_penetration_depth({"beta_in": 40., "beta_out": -60.}, geometry, delta)


def test_nonfinite_derived_aspect_ratio_is_rejected_even_if_raw_formula_would_be_zero():
    geometry = dict(BASELINE["eps0"]["continuous"]["stages_performance"][0]["geometry"]["rotor"])
    geometry.update(height=1e308, chord=1e-308)
    with pytest.raises(ValueError, match="aspect ratio"):
        benner.get_penetration_depth({"beta_in": 40., "beta_out": -60.}, geometry, 0.)


def test_extreme_finite_boundary_layer_is_rejected_when_its_raw_prediction_overflows():
    geometry = BASELINE["eps0"]["continuous"]["stages_performance"][0]["geometry"]["rotor"]
    with pytest.raises(ValueError, match="Raw penetration"):
        benner.get_penetration_depth({"beta_in": 40., "beta_out": -60.}, geometry, 1e308)
