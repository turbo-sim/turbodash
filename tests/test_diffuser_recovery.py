"""Prescribed exhaust recovery changes only the overall efficiency reference."""

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from turbodash.core.diffuser import diffuser_settings
from turbodash.core.turbine import compute_turbine_performance, compute_turbine_efficiency_trends
from turbodash.optimization import load_case, evaluate_objective, evaluate_constraints
from turbodash.apps.app_turbine_optimization.opt_model import (
    build_settings, output_catalog, variable_catalog,
)
from turbodash.io.yaml import dump_turbine_yaml


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("kind", ["axial", "radial"])
@pytest.mark.parametrize("stage_count", [1, 2])
def test_recovery_endpoints_formula_and_unchanged_turbine_solution(kind, stage_count):
    config = load_case(ROOT / f"demos/optimization/case_{kind}.yaml")
    if stage_count == 2:
        config["inputs"]["stages"].append(deepcopy(config["inputs"]["stages"][0]))
    original = deepcopy(config)
    results = []
    for fraction in (0, .6, 1):
        candidate = deepcopy(config)
        candidate["inputs"]["diffuser"]["kinetic_energy_recovery_fraction"] = fraction
        solved = compute_turbine_performance(candidate)
        performance = solved["overall_performance"]
        velocity = solved["stages_performance"][-1]["flow_stations"][3]["v"]
        expected = performance["power_actual"] / (
            performance["mass_flow_rate"] * (
                performance["isentropic_enthalpy_drop"] - fraction * velocity**2 / 2
            )
        )
        assert performance["efficiency_ts_with_recovery"] == pytest.approx(expected)
        trends = compute_turbine_efficiency_trends(solved, [candidate["inputs"]["blade_velocity_ratio"], .3, .5, .8])
        if stage_count == 1:
            assert trends["eta_ts_with_recovery"][0] == pytest.approx(expected)
        trend_work = performance["isentropic_enthalpy_drop"] - trends["loss_total"] - trends["leaving_kinetic_energy"]
        assert trends["eta_ts_with_recovery"] == pytest.approx(
            trend_work / (performance["isentropic_enthalpy_drop"] - fraction * trends["leaving_kinetic_energy"])
        )
        if fraction == 0:
            assert trends["eta_ts_with_recovery"] == pytest.approx(trends["eta_ts"])
        if fraction == 1:
            assert trends["eta_ts_with_recovery"] == pytest.approx(trends["eta_tt"])
        results.append(solved)
    assert config == original
    zero, middle, full = [result["overall_performance"] for result in results]
    assert zero["efficiency_ts_with_recovery"] == zero["efficiency_ts"]
    assert full["efficiency_ts_with_recovery"] == full["efficiency_tt"]
    assert zero["efficiency_ts_with_recovery"] < middle["efficiency_ts_with_recovery"] < full["efficiency_ts_with_recovery"]
    for solved in results[1:]:
        assert solved["stages_performance"] == results[0]["stages_performance"]
        assert {k: v for k, v in solved["overall_performance"].items() if k != "efficiency_ts_with_recovery"} == {
            k: v for k, v in zero.items() if k != "efficiency_ts_with_recovery"
        }


def test_legacy_cases_default_to_sixty_percent_and_export_after_stages():
    config = load_case(ROOT / "demos/optimization/case_axial.yaml")
    config["inputs"].pop("diffuser")
    original = deepcopy(config)
    solved = compute_turbine_performance(config)
    assert solved["inputs"]["diffuser"]["kinetic_energy_recovery_fraction"] == .6
    exported = yaml.safe_load(dump_turbine_yaml(config))
    keys = list(exported["inputs"])
    assert keys.index("diffuser") == keys.index("stages") + 1
    assert exported["inputs"]["diffuser"]["kinetic_energy_recovery_fraction"] == .6
    assert config == original


@pytest.mark.parametrize("settings", [None, [], 0.6, "0.6",
    *[{"kinetic_energy_recovery_fraction": value} for value in
      (-.01, 1.01, float("nan"), float("inf"), True, None, "0.6")],
])
def test_invalid_recovery_assumptions_are_rejected(settings):
    with pytest.raises(ValueError, match="diffuser"):
        diffuser_settings({"diffuser": settings})


def test_recovered_efficiency_is_an_overall_objective_and_constraint_only():
    config = load_case(ROOT / "demos/optimization/case_axial.yaml")
    solved = compute_turbine_performance(config)
    path = "$overall_performance.efficiency_ts_with_recovery"
    catalog = output_catalog(config, solved)
    assert catalog[path]["objective"]
    assert not any("stage_performance.efficiency_ts_with_recovery" in key for key in output_catalog(config))
    assert not any("diffuser" in key for key in variable_catalog(config))
    settings = build_settings(
        config, [dict(path="blade_velocity_ratio", lower=.5, upper=.7)],
        [dict(path=path, comparison=">=", target=.8, scale=1)],
        path, "maximize", 1e-8, 50, 1e-6, solved,
    )
    value = solved["overall_performance"]["efficiency_ts_with_recovery"]
    assert evaluate_objective(solved, settings["objective_function"]) == pytest.approx(value)
    _, inequalities, _ = evaluate_constraints(solved, settings["constraints"])
    assert inequalities[0] == pytest.approx(.8 - value)
