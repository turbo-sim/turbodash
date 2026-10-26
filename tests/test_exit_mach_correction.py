"""The optional exit penalty reaches every row without changing the default."""

from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import yaml

from turbodash.core import turbine as core
from turbodash.loss_models import loss_model
from turbodash.loss_models.loss_model_kacker_okapuu import (
    get_exit_mach_correction_factor,
)
from turbodash.loss_models.options import EXIT_MACH_CORRECTION_MODELS

ROOT = Path(__file__).parents[1]
COMPONENTS = ("profile", "secondary", "clearance", "trailing", "incidence", "total")


def turbine_case(kind):
    path = ROOT / "demos/basic_demo" / f"demo_{kind}_turbine.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def row_inputs():
    """Capture physically consistent production row inputs for standalone tests."""
    captured = {}
    evaluate = loss_model.evaluate_loss_model

    def capture(options, parameters):
        captured[parameters["geometry"]["cascade_type"]] = deepcopy(parameters)
        return evaluate(options, parameters)

    with patch.object(loss_model, "evaluate_loss_model", capture):
        core.compute_turbine_performance(turbine_case("axial"))
    for parameters in captured.values():
        geometry = parameters["geometry"]
        geometry["gauging_angle"] = np.rad2deg(
            np.arccos(np.clip(geometry["A_throat"] / geometry["A_out"], 0, 1))
        )
    return captured


def test_factor_is_unity_through_sonic_exit_and_grows_quadratically():
    mach = np.array([0, 0.5, 1, 1.1, 1.2, 1.5, 2])
    np.testing.assert_allclose(
        get_exit_mach_correction_factor(mach), [1, 1, 1, 1.6, 3.4, 16, 61]
    )
    derivatives = get_exit_mach_correction_factor(mach + 1e-30j).imag / 1e-30
    np.testing.assert_allclose(derivatives, [0, 0, 0, 12, 24, 60, 120])


@pytest.mark.parametrize("model", EXIT_MACH_CORRECTION_MODELS)
@pytest.mark.parametrize("row", ["stator", "rotor"])
@pytest.mark.parametrize("mach,factor", [(0.8, 1), (1, 1), (1.2, 3.4), (1.5, 16)])
def test_only_profile_is_scaled_in_each_model(row_inputs, model, row, mach, factor):
    parameters = deepcopy(row_inputs[row])
    parameters["flow"]["Ma_rel_out"] = mach
    before = deepcopy(parameters)
    omitted = loss_model.evaluate_loss_model(model, parameters)
    disabled = loss_model.evaluate_loss_model(
        dict(model=model, exit_mach_correction=False), parameters
    )
    enabled = loss_model.evaluate_loss_model(
        dict(model=model, exit_mach_correction=True), parameters
    )
    assert parameters == before
    assert enabled["loss_profile"] == pytest.approx(factor * disabled["loss_profile"])
    for component in COMPONENTS:
        key = f"loss_{component}"
        assert omitted[key] == disabled[key]
        if component not in ("profile", "total"):
            assert enabled[key] == disabled[key]
    assert enabled["loss_total"] == pytest.approx(
        disabled["loss_total"] + (factor - 1) * disabled["loss_profile"]
    )
    if model == "benner":
        assert (
            enabled["diagnostics"]["primary_loss_weight"]
            == disabled["diagnostics"]["primary_loss_weight"]
        )
        assert enabled["diagnostics"]["profile_loss_unweighted"] == pytest.approx(
            factor * disabled["diagnostics"]["profile_loss_unweighted"]
        )


@pytest.mark.parametrize("kind", ["axial", "radial_inflow"])
@pytest.mark.parametrize("model", EXIT_MACH_CORRECTION_MODELS)
def test_yaml_option_reaches_stator_and_rotor_in_meanline(kind, model):
    cfg = turbine_case(kind)
    cfg["inputs"]["loss_model"] = model
    original = deepcopy(cfg)
    omitted = core.compute_turbine_performance(cfg)
    assert cfg == original
    cfg["inputs"]["exit_mach_correction"] = False
    disabled = core.compute_turbine_performance(cfg)
    cfg["inputs"]["exit_mach_correction"] = True
    enabled = core.compute_turbine_performance(yaml.safe_load(yaml.safe_dump(cfg)))
    assert enabled["inputs"]["exit_mach_correction"] is True
    assert disabled["overall_performance"] == omitted["overall_performance"]
    assert (
        enabled["overall_performance"]["efficiency_tt"]
        < disabled["overall_performance"]["efficiency_tt"]
    )
    for old_stage, off_stage, on_stage in zip(
        omitted["stages_performance"],
        disabled["stages_performance"],
        enabled["stages_performance"],
        strict=True,
    ):
        for row, index, mach_key in (("stator", 1, "Ma_abs"), ("rotor", 3, "Ma_rel")):
            mach = on_stage["flow_stations"][index][mach_key]
            assert mach > 1  # Exercise the active correction for both rows.
            factor = 1 + 60 * (mach - 1) ** 2
            assert old_stage["losses"][row] == off_stage["losses"][row]
            assert on_stage["losses"][row]["loss_profile"] == pytest.approx(
                factor * off_stage["losses"][row]["loss_profile"]
            )


def test_multistage_meanline_passes_option_to_every_row(monkeypatch):
    cfg = turbine_case("axial")
    cfg["inputs"]["exit_mach_correction"] = True
    cfg["inputs"]["stages"].append(
        dict(deepcopy(cfg["inputs"]["stages"][0]), name="stage_2")
    )
    observed = []
    evaluate = core.compute_loss_coefficient

    def capture(geometry, **kwargs):
        observed.append((kwargs["cascade_type"], kwargs["exit_mach_correction"]))
        return evaluate(geometry, **kwargs)

    monkeypatch.setattr(core, "compute_loss_coefficient", capture)
    core.compute_turbine_performance(cfg)
    assert observed == [("stator", True), ("rotor", True)] * 2


@pytest.mark.parametrize("value", ["false", "true", 0, 1, None, [], {}])
def test_non_boolean_yaml_option_is_rejected_before_calculation(value):
    with pytest.raises(ValueError, match="exit_mach_correction must be a boolean"):
        core.compute_turbine_performance({"inputs": {"exit_mach_correction": value}})
    with pytest.raises(ValueError, match="exit_mach_correction must be a boolean"):
        loss_model.evaluate_loss_model(
            dict(model="benner", exit_mach_correction=value), {}
        )


@pytest.mark.parametrize("model", ["isentropic", "custom"])
def test_non_empirical_models_are_unchanged(row_inputs, model):
    options = dict(model=model, custom_value=0.1)
    assert loss_model.evaluate_loss_model(
        options, row_inputs["stator"]
    ) == loss_model.evaluate_loss_model(
        dict(options, exit_mach_correction=True), row_inputs["stator"]
    )
