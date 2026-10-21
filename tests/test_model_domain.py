"""Physical trial rejection without modifying valid meanline calculations."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import jaxprop as jxp
import numpy as np
import pytest
import yaml

from turbodash.core import ModelConsistencyError, ModelDomainError
from turbodash.core import turbine
from turbodash.core.stage import compute_performance_stage

ROOT = Path(__file__).resolve().parents[1]


def case(kind="axial", stages=1):
    config = yaml.safe_load((ROOT / f"demos/optimization/case_{kind}.yaml").read_text())
    config["inputs"]["stages"] = [deepcopy(config["inputs"]["stages"][0]) for _ in range(stages)]
    for i, stage in enumerate(config["inputs"]["stages"]):
        stage["name"] = f"stage_{i + 1}"
    return config


# Captured from the unmodified forward model before adding domain checks.
BASELINE = {
    ("axial", 1): (.8375998510552765, .9008639739279823, 1873990.6683183801, [.41540594894103317]),
    ("axial", 2): (.8717605092042644, .9111129341860491, 1950419.4720176216, [.2937363634422198, .30892033612251363]),
    ("axial", 3): (.8504459330597642, .8971732573707621, 1902731.6450157054, [.23983473644535502, .2651106898917023, .2673768017539429]),
    ("radial", 1): (.8532871916143269, .9157944000855863, 1909088.4895290155, [.47612659292879367]),
    ("radial", 2): (.8985446296625565, .934763882374485, 2010344.4967591106, [.33667234256386075, .4950552513900362]),
    ("radial", 3): (.8993381137819023, .933476412148578, 2012119.7857986675, [.2748918165968623, .44178043803799033, .5505805581183797]),
}


@pytest.mark.parametrize("kind,stages", BASELINE)
def test_valid_axial_radial_and_multistage_results_preserved(kind, stages):
    config = case(kind, stages)
    before = deepcopy(config)
    with np.errstate(all="raise"):
        result = turbine.compute_turbine_performance(config)
    expected = BASELINE[kind, stages]
    actual = result["overall_performance"]
    assert [actual[key] for key in ("efficiency_ts", "efficiency_tt", "power_actual")] == pytest.approx(expected[:3], rel=2e-13)
    assert [stage["stage_performance"]["flow_coefficient"] for stage in result["stages_performance"]] == pytest.approx(expected[3], rel=2e-13)
    assert config == before


def failing_trial(config, stage_index=0):
    config["inputs"]["blade_velocity_ratio"] = .665076231447185
    stage = config["inputs"]["stages"][stage_index]
    stage.update(degree_reaction=.27399846359538044, stator_exit_angle=50.,
                 meridional_velocity_ratio_12=.5, meridional_velocity_ratio_23=.5,
                 meridional_velocity_ratio_34=.5)
    return config


def test_original_rotor_failure_has_unmodified_radicand_and_stage_metadata():
    with np.errstate(all="raise"), pytest.raises(ModelDomainError) as captured:
        turbine.compute_turbine_performance(failing_trial(case()))
    error = captured.value
    assert error.recoverable is True
    assert error.code == "invalid_radicand"
    assert error.quantity == "rotor_exit_radicand"
    assert float(error.value) == pytest.approx(-.604483587, abs=2e-9)
    assert error.margin == error.value
    assert (error.stage_index, error.stage_name, error.station) == (0, "stage_1", 4)
    assert json.loads(json.dumps(error.to_dict(), allow_nan=False))["value"] < 0


@pytest.mark.parametrize("kind", ["axial", "radial"])
def test_later_stage_uses_its_own_inlet_and_radius_conditions(kind):
    config = failing_trial(case(kind, 2), stage_index=1)
    with np.errstate(all="raise"), pytest.raises(ModelDomainError) as captured:
        turbine.compute_turbine_performance(config)
    assert captured.value.stage_index == 1
    assert captured.value.stage_name == "stage_2"
    assert captured.value.quantity == "rotor_exit_radicand"
    assert captured.value.station == 4


@pytest.mark.parametrize("updates,quantity", [
    ({"blade_velocity_ratio": 0.}, "blade_velocity_ratio"),
    ({"degree_reaction": .5, "stator_exit_angle": 0., "meridional_velocity_ratio_12": 2.}, "flow_coefficient_squared"),
    ({"degree_reaction": 1., "stator_exit_angle": 0.}, "flow_coefficient_denominator"),
])
def test_triangle_domain_checked_before_invalid_numerical_operations(updates, quantity):
    kwargs = dict(blade_velocity_ratio=.6, stage_spouting_velocity=100.,
                  stator_inlet_angle=0., stator_exit_angle=70., degree_reaction=.5)
    kwargs.update(updates)
    with np.errstate(all="raise"), pytest.raises(ModelDomainError) as captured:
        turbine.compute_stage_velocity_triangles(**kwargs)
    assert captured.value.quantity == quantity


def test_plot_sweep_masks_invalid_roots_without_eager_evaluation():
    nu = np.linspace(.1, 2., 100)
    kwargs = dict(stage_spouting_velocity=100., stator_inlet_angle=0., stator_exit_angle=50.,
                  degree_reaction=.27399846359538044, meridional_velocity_ratio_12=.5,
                  meridional_velocity_ratio_23=.5, meridional_velocity_ratio_34=.5)
    with np.errstate(all="raise"):
        result = turbine.compute_stage_velocity_triangles(nu, **kwargs)
    first_invalid = np.flatnonzero(~result["domain_valid"])[0]
    assert np.all(np.isnan(result["beta4"][first_invalid:]))
    for i in range(first_invalid):
        scalar = turbine.compute_stage_velocity_triangles(nu[i], **kwargs)
        for key in ("phi", "psi", "beta4", "eta_ts", "eta_tt", "v_4"):
            assert result[key][i] == scalar[key]


def test_plot_sweep_handles_zero_divisor_and_negative_flow_radicand():
    with np.errstate(all="raise"):
        result = turbine.compute_stage_velocity_triangles(np.array([0., .6]), 100., 0., 0., .5,
                                                          meridional_velocity_ratio_12=2.)
    assert not np.any(result["domain_valid"])
    assert np.all(np.isnan(result["phi"]))


@pytest.mark.parametrize("updates", [
    {"meridional_velocity_ratio_34": 0.},
    {"stator_exit_angle": 0., "degree_reaction": 1.},
    {"stage_spouting_velocity": np.array([100., 100.]), "blade_velocity_ratio": 0.},
])
def test_plot_arrays_with_invalid_scalar_inputs_are_masked(updates):
    kwargs = dict(blade_velocity_ratio=np.array([.5, .6]), stage_spouting_velocity=100.,
                  stator_inlet_angle=0., stator_exit_angle=70., degree_reaction=.5)
    kwargs.update(updates)
    with np.errstate(all="raise"):
        result = turbine.compute_stage_velocity_triangles(**kwargs)
    assert not np.any(result["domain_valid"])
    assert np.all(np.isnan(result["beta4"]))


def test_legacy_triangle_api_uses_the_same_domain_checks():
    with pytest.raises(ModelDomainError) as captured:
        compute_performance_stage(0., 50., .27399846359538044, .665076231447185,
                                  meridional_velocity_ratio_12=.5,
                                  meridional_velocity_ratio_23=.5,
                                  meridional_velocity_ratio_34=.5)
    assert captured.value.quantity == "rotor_exit_radicand"


@pytest.mark.parametrize("updates,quantity", [
    ({"exit_pressure": 600000.}, "overall_isentropic_enthalpy_drop"),
    ({"mass_flow_rate": 0.}, "mass_flow_rate"),
    ({"rotational_speed": 0.}, "angular_velocity"),
])
def test_invalid_global_scales_are_explicit_domains(updates, quantity):
    config = case()
    config["inputs"].update(updates)
    with np.errstate(all="raise"), pytest.raises(ModelDomainError) as captured:
        turbine.compute_turbine_performance(config)
    assert captured.value.quantity == quantity


def test_stage_spouting_velocity_rejects_nonpositive_local_drop_before_root():
    with np.errstate(all="raise"), pytest.raises(ModelDomainError) as captured:
        turbine.compute_stage_performance(
            fluid=None, state_in_0={"s": 1., "h": 10.}, state_out_s={"h": 11.},
            mass_flow_rate=1., omega=1., radius_1=1., radius_2=1., radius_3=1., radius_4=1.,
            stator_inlet_angle=0., stator_exit_angle=70., degree_reaction=.5,
            meridional_velocity_ratio_12=1., meridional_velocity_ratio_23=1., meridional_velocity_ratio_34=1.)
    assert captured.value.quantity == "stage_isentropic_enthalpy_drop"
    assert captured.value.value == -1.


def test_consistency_errors_are_not_recoverable_domain_errors():
    original = turbine.compute_stage_velocity_triangles

    def broken_equation(*args, **kwargs):
        result = original(*args, **kwargs)
        result["psi"] += .1
        return result

    with patch.object(turbine, "compute_stage_velocity_triangles", broken_equation):
        with pytest.raises(ModelConsistencyError) as captured:
            turbine.compute_turbine_performance(case())
    assert not captured.value.recoverable
    assert captured.value.stage_index == 0
    assert "Work coefficient" in str(captured.value)


def test_runtime_consistency_checks_survive_python_optimized_mode():
    code = """
from turbodash.core.turbine import assert_velocity_triangle
from turbodash.core.errors import ModelConsistencyError
try:
    assert_velocity_triangle(10., 1., 20., 0., 0., 'station test')
except ModelConsistencyError as error:
    print(error.recoverable)
else:
    raise RuntimeError('Consistency check disappeared under python -O')
"""
    process = subprocess.run([sys.executable, "-O", "-c", code], cwd=ROOT, capture_output=True, text=True, check=True)
    assert process.stdout.strip() == "False"


def test_known_thermodynamic_flash_failure_is_recoverable():
    fluid = jxp.Fluid("Air", backend="HEOS")
    with pytest.raises(ModelDomainError) as captured:
        turbine._thermodynamic_state(fluid, jxp.HmassSmass_INPUTS, -1.e8, 4000., station=2)
    assert captured.value.code == "thermodynamic_state"
    assert captured.value.station == 2


def test_unknown_property_library_error_is_not_reclassified():
    def broken_state(*args):
        try:
            raise KeyError("programming error")
        except KeyError as error:
            raise RuntimeError("unexpected property error") from error

    with pytest.raises(RuntimeError, match="unexpected property error") as captured:
        turbine._thermodynamic_state(SimpleNamespace(get_state=broken_state), jxp.HmassSmass_INPUTS, 1., 2.)
    assert not isinstance(captured.value, ModelDomainError)


def test_domain_metadata_supports_strict_json_with_nonfinite_values():
    error = ModelDomainError("bad value", code="nonfinite_quantity", quantity="test", value=np.float64(np.nan))
    assert json.loads(json.dumps(error.to_dict(), allow_nan=False))["value"] == "nan"


def test_loss_model_domain_sentinel_rejected_with_row_and_stage_metadata():
    from turbodash.loss_models import loss_model
    original = loss_model.evaluate_loss_model

    def invalid_loss(*args, **kwargs):
        losses = original(*args, **kwargs)
        losses["loss_profile"] = np.nan
        return losses

    with patch.object(loss_model, "evaluate_loss_model", invalid_loss):
        with pytest.raises(ModelDomainError) as captured:
            turbine.compute_turbine_performance(case())
    assert captured.value.code == "loss_model_domain"
    assert captured.value.row == "stator"
    assert captured.value.station == 2
    assert captured.value.stage_index == 0
