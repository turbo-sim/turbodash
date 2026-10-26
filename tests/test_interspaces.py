"""Passage geometry, dimensionless scaling, and enforced interspace limits."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

from turbodash.core.errors import ModelDomainError
from turbodash.core.interspaces import (
    add_interspace_geometry, get_interspace_geometry, normalize_axial_spacing,
)
from turbodash.core.turbine import compute_turbine_performance
from turbodash.optimization import TurbineOptimizationProblem, evaluate_constraints


def passage_results(kind="axial"):
    stages = []
    for heights in ((.04, .04, .08, .08), (.10, .10, .12, .12)):
        radii = (.4, .35, .3, .25) if kind == "radial" else (.3,) * 4
        stages.append(dict(
            geometry_meanline={
                "stator": dict(chord_meridional=.1, opening=.01, flaring_angle=0.),
                "rotor": dict(chord_meridional=.2, opening=.02, flaring_angle=0.),
            },
            flow_stations=[dict(r=r, H=h) for r, h in zip(radii, heights, strict=True)],
        ))
    return dict(inputs=dict(turbine_type=kind, stages=[
        dict(axial_spacing_before_rotor=.25),
        dict(axial_spacing_before_stator=.4, axial_spacing_before_rotor=.5),
    ]), stages_performance=stages)


def test_every_actual_gap_is_owned_by_the_downstream_row():
    result = add_interspace_geometry(passage_results())
    first, second = result["stages_performance"]
    assert set(first["geometry_interspaces"]) == {"before_rotor"}
    assert set(second["geometry_interspaces"]) == {"before_stator", "before_rotor"}
    expected = [(0, "rotor", .05, .04), (1, "stator", .04, .02), (1, "rotor", .1, .02)]
    for index, row, length, height_change in expected:
        gap = get_interspace_geometry(result, index, row)
        assert gap["meridional_length"] == pytest.approx(length)
        assert gap["flaring_angle"] == pytest.approx(np.degrees(np.arctan(height_change / (2 * length))))
        assert gap["hub_wall_angle"] == pytest.approx(-gap["flaring_angle"])
        assert gap["shroud_wall_angle"] == pytest.approx(gap["flaring_angle"])
    assert all(set(stage["geometry_meanline"]) == {"stator", "rotor"} for stage in (first, second))
    with pytest.raises(ValueError, match="first"):
        get_interspace_geometry(result, 0, "stator")


def test_ragged_wildcard_constraints_detect_a_gap_missed_by_row_limits():
    result = add_interspace_geometry(passage_results())
    row_limit = dict(variable="$stages_performance.*.geometry_meanline.*.flaring_angle",
                     type="<=", value=10, normalize=10)
    gap_limit = dict(row_limit, variable="$stages_performance.*.geometry_interspaces.*.flaring_angle")
    _, rows, _ = evaluate_constraints(result, [row_limit])
    _, gaps, report = evaluate_constraints(result, [gap_limit])
    assert rows.shape == (4,) and np.all(rows < 0)
    assert gaps.shape == (3,) and np.any(gaps > 0)
    assert len(report) == 3


@pytest.mark.parametrize("kind", ["axial", "radial"])
def test_flaring_is_invariant_when_all_turbine_dimensions_scale(kind):
    initial = passage_results(kind)
    scaled = deepcopy(initial)
    for stage in scaled["stages_performance"]:
        for station in stage["flow_stations"]:
            for key in ("r", "H"):
                station[key] *= 7
        for geometry in stage["geometry_meanline"].values():
            for key in ("chord_meridional", "opening"):
                geometry[key] *= 7
    for result in (initial, scaled):
        add_interspace_geometry(result)
    for index, row in ((0, "rotor"), (1, "stator"), (1, "rotor")):
        original = get_interspace_geometry(initial, index, row)
        enlarged = get_interspace_geometry(scaled, index, row)
        assert enlarged["meridional_length"] == pytest.approx(7 * original["meridional_length"])
        for quantity in ("flaring_angle", "hub_wall_angle", "shroud_wall_angle"):
            assert enlarged[quantity] == pytest.approx(original[quantity])


def test_radial_length_comes_from_station_radii_without_axial_inputs():
    result = passage_results("radial")
    result["inputs"]["stages"] = [{}, {}]
    result["stages_performance"][1]["flow_stations"][0]["r"] = .2
    add_interspace_geometry(result)
    gap = get_interspace_geometry(result, 0, "rotor")
    assert gap["meridional_length"] == pytest.approx(.05)
    assert "spacing_ratio" not in gap and "axial_length" not in gap
    interstage = get_interspace_geometry(result, 1, "stator")
    assert interstage["meridional_length"] == pytest.approx(.05)


def test_radial_zero_extent_is_flat_or_a_reported_right_angle_jump():
    result = passage_results("radial")
    stations = result["stages_performance"][0]["flow_stations"]
    stations[2] = dict(stations[1])
    gap = get_interspace_geometry(result, 0, "rotor")
    assert gap["flaring_angle"] == 0 and not gap["degenerate_interspace"]
    stations[2]["H"] *= 2
    gap = get_interspace_geometry(result, 0, "rotor")
    assert gap["flaring_angle"] == 90 and gap["degenerate_interspace"]


def test_axial_wall_angles_include_mean_radius_movement():
    result = passage_results()
    stations = result["stages_performance"][0]["flow_stations"]
    stations[2]["H"] = stations[1]["H"]
    stations[2]["r"] += .02
    gap = get_interspace_geometry(result, 0, "rotor")
    assert gap["flaring_angle"] == 0
    assert gap["hub_wall_angle"] == pytest.approx(np.degrees(np.arctan(.02 / .05)))
    assert gap["shroud_wall_angle"] == gap["hub_wall_angle"]


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True, "0.3"])
def test_invalid_axial_ratios_are_rejected(value):
    result = passage_results()
    result["inputs"]["stages"][0]["axial_spacing_before_rotor"] = value
    with pytest.raises((ValueError, ModelDomainError), match="positive"):
        add_interspace_geometry(result)


def real_case(kind="axial"):
    path = Path(__file__).resolve().parents[1] / "demos/optimization" / f"case_{kind}.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_legacy_migration_preserves_baseline_gap_then_fixes_its_ratio():
    config = real_case()
    config["inputs"]["stages"][0].pop("axial_spacing_before_rotor", None)
    before = deepcopy(config)
    baseline = compute_turbine_performance(config)
    original = baseline["stages_performance"][0]
    gap = original["geometry_interspaces"]["before_rotor"]
    assert gap["meridional_length"] == pytest.approx(1.5 * original["geometry_meanline"]["stator"]["opening"])
    normalized = normalize_axial_spacing(config, results=baseline)
    ratio = normalized["inputs"]["stages"][0]["axial_spacing_before_rotor"]
    assert config == before
    normalized["inputs"]["stages"][0]["blade_geometry"]["rotor"]["aspect_ratio"] *= 2
    changed = compute_turbine_performance(normalized)
    updated = changed["stages_performance"][0]
    assert updated["geometry_interspaces"]["before_rotor"]["spacing_ratio"] == ratio
    assert updated["geometry_interspaces"]["before_rotor"]["meridional_length"] == pytest.approx(.5 * gap["meridional_length"])


def test_spacing_does_not_change_meanline_performance_and_is_not_optimizable():
    config = real_case()
    first = compute_turbine_performance(config)
    config["inputs"]["stages"][0]["axial_spacing_before_rotor"] = 1.5
    second = compute_turbine_performance(config)
    assert first["overall_performance"] == second["overall_performance"]
    assert first["stages_performance"][0]["losses"] == second["stages_performance"][0]["losses"]
    config["optimization"]["design_variables"] = {
        "stages.0.axial_spacing_before_rotor": dict(lower_value=.1, upper_value=2),
    }
    with pytest.raises(ValueError, match="fixed geometry"):
        TurbineOptimizationProblem(config)


def test_optimizer_freezes_legacy_spacing_at_the_effective_initial_design():
    config = real_case()
    config["inputs"]["stages"][0].pop("axial_spacing_before_rotor", None)
    before = deepcopy(config)
    problem = TurbineOptimizationProblem(config)
    initial = problem.evaluate(problem.x0)
    ratio = initial["config"]["inputs"]["stages"][0]["axial_spacing_before_rotor"]
    assert ratio > 0 and problem.n_evaluations == 1
    trial = problem.x0.copy()
    trial[0] += .001
    updated = problem.evaluate(trial)
    assert updated["config"]["inputs"]["stages"][0]["axial_spacing_before_rotor"] == ratio
    assert config == before
