"""Physical applicability and numeric contracts of optimization picker presets."""

from copy import deepcopy
import math
from pathlib import Path

import numpy as np
import pytest

from turbodash import optimization as opt
from turbodash.apps.app_turbine_optimization.opt_model import (
    build_settings,
    constraint_groups,
    nested_config,
    output_catalog,
    variable_catalog,
    variable_groups,
)
from turbodash.core.turbine import compute_turbine_performance
from turbodash.apps.app_turbine_optimization.opt_selection import apply_selection_operations


ROOT = Path(__file__).resolve().parents[1]


def case(kind="axial", count=1):
    config = opt.load_case(ROOT / f"demos/optimization/case_{kind}.yaml")
    stage = config["inputs"]["stages"][0]
    config["inputs"]["stages"] = [deepcopy(stage) for _ in range(count)]
    for i, stage in enumerate(config["inputs"]["stages"]):
        stage["name"] = ["High pressure", "Low pressure", "Exhaust"][i]
    return config


def test_catalog_hierarchy_keeps_stage_names_rows_and_legacy_labels():
    config = case(count=2)
    variables = variable_catalog(config)
    assert variables["blade_velocity_ratio"]["hierarchy"] == ["Overall"]
    reaction = variables["stages.1.degree_reaction"]
    assert reaction["hierarchy"] == ["Stage 2 · Low pressure", "Stage inputs"]
    assert reaction["short_label"] == "Reaction [-]"
    assert reaction["label"] == "Stage 2 · Reaction [-]"
    assert variables["stages.0.blade_geometry.rotor.solidity_chord"]["hierarchy"] == [
        "Stage 1 · High pressure", "Rotor geometry"
    ]
    outputs = output_catalog(config)
    assert outputs["$stages_performance.*.geometry_meanline.*.height"]["hierarchy"] == [
        "All stages", "Both rows geometry"
    ]
    assert outputs["$stages_performance.1.flow_stations.3.Ma_rel"]["hierarchy"] == [
        "Stage 2 · Low pressure", "Station 4"
    ]
    assert outputs["$overall_performance.efficiency_ts"]["objective"]
    assert not outputs["$stages_performance.*.stage_performance.efficiency_ts"]["objective"]
    assert all(entry["short_label"] and entry["hierarchy"] for entry in variables.values())
    assert all(entry["short_label"] and entry["hierarchy"] for entry in outputs.values())


@pytest.mark.parametrize("speed_type,speed", [("specific", .7), ("actual", 30000)])
def test_speed_preset_uses_current_speed_definition_in_flat_and_app_configs(speed_type, speed):
    config = case()
    config["inputs"].update(rotational_speed_type=speed_type, rotational_speed=speed)
    flat = variable_catalog(config)
    nested = variable_catalog(nested_config(config))
    assert nested == flat
    choice = flat["rotational_speed"]
    assert choice["lower"] < speed < choice["upper"]
    assert choice["short_label"] == ("Specific speed [-]" if speed_type == "specific" else "Rotational speed [rpm]")
    assert variable_groups(config)["overall"]["paths"] == ["blade_velocity_ratio", "rotational_speed"]


@pytest.mark.parametrize("kind", ["axial", "radial"])
@pytest.mark.parametrize("count", [1, 3])
def test_presets_are_applicable_bounded_and_hold_one_work_weight_fixed(kind, count):
    config = case(kind, count)
    before = deepcopy(config)
    catalog = variable_catalog(config)
    groups = variable_groups(config, catalog)
    for group in groups.values():
        assert group["hierarchy"] and group["description"]
        assert group["paths"]
        assert len(group["paths"]) == len(set(group["paths"]))
        for path in group["paths"]:
            entry = catalog[path]
            assert entry["preset_active"]
            assert math.isfinite(entry["lower"]) and math.isfinite(entry["upper"])
            assert entry["lower"] < entry["upper"]
            assert entry["value"] == opt.read_path(config["inputs"], path)
    all_paths = groups["all"]["paths"]
    assert not any(path.startswith(("inlet_", "exit_", "mass_", "fluid_", "loss_")) for path in all_paths)
    assert not any("leading_edge" in path or "wedge" in path or "thickness_location" in path for path in catalog)
    assert ("radial_radii" in groups) == (kind == "radial")
    assert any("aspect_ratio" in path for path in all_paths) == (kind == "axial")
    if count > 1:
        weights = [f"stages.{i}.work_fraction_split" for i in range(count)]
        assert all(path in catalog for path in weights)
        assert groups["work_split"]["paths"] == weights[:-1]
        assert set(weights[:-1]) <= set(all_paths)
        assert weights[-1] not in all_paths
    else:
        assert "work_split" not in groups
        assert not any("work_fraction_split" in path for path in catalog)
    assert config == before


@pytest.mark.parametrize("ratio", [.5, .999, 1.001, 1.8])
@pytest.mark.parametrize("turbine_type", ["radial", "radial_inflow", "radial_outflow"])
def test_radial_row_bounds_keep_direction_and_avoid_zero_chord(ratio, turbine_type):
    config = case("radial")
    config["inputs"]["turbine_type"] = turbine_type
    for pair in ("12", "34"):
        config["inputs"]["stages"][0][f"radius_ratio_{pair}"] = ratio
    catalog = variable_catalog(config)
    for pair in ("12", "34"):
        entry = catalog[f"stages.0.radius_ratio_{pair}"]
        assert entry["lower"] < ratio < entry["upper"]
        if ratio < 1:
            assert 0 < entry["lower"] < entry["upper"] < 1
        else:
            assert 1 < entry["lower"] < entry["upper"]


def test_radial_catalog_skips_only_unused_initial_radius_link():
    config = case("radial", count=2)
    catalog = variable_catalog(config)
    assert "stages.0.radius_ratio_01" not in catalog
    assert "stages.1.radius_ratio_01" in catalog
    result = compute_turbine_performance(config)
    changed = deepcopy(config)
    changed["inputs"]["stages"][0]["radius_ratio_01"] = 1.3
    same = compute_turbine_performance(changed)
    assert same["overall_performance"] == result["overall_performance"]
    for actual, expected in zip(same["stages_performance"], result["stages_performance"]):
        assert actual["geometry_meanline"] == expected["geometry_meanline"]
    changed["inputs"]["stages"][1]["radius_ratio_01"] = 1.1
    different = compute_turbine_performance(changed)
    assert different["stages_performance"][1]["geometry_meanline"]["stator"]["radius_in"] != (
        result["stages_performance"][1]["geometry_meanline"]["stator"]["radius_in"]
    )


def test_fixed_work_weight_removes_normalization_null_direction():
    config = case(count=3)
    original = compute_turbine_performance(config)
    scaled = deepcopy(config)
    for stage in scaled["inputs"]["stages"]:
        stage["work_fraction_split"] *= 5
    equivalent = compute_turbine_performance(scaled)
    assert equivalent["overall_performance"] == pytest.approx(original["overall_performance"])
    bounds = variable_catalog(scaled)
    for i in range(3):
        choice = bounds[f"stages.{i}.work_fraction_split"]
        assert choice["lower"] < 5 < choice["upper"]
    scaled["inputs"]["stages"][0]["work_fraction_split"] *= 1.2
    changed = compute_turbine_performance(scaled)
    assert changed["stages_performance"][0]["stage_performance"]["isentropic_enthalpy_drop"] != (
        original["stages_performance"][0]["stage_performance"]["isentropic_enthalpy_drop"]
    )


@pytest.mark.parametrize("loss_model", ["benner", "kacker_okapuu", "moustapha", "isentropic"])
def test_thickness_presets_follow_actual_loss_sensitivity(loss_model):
    config = case()
    config["inputs"]["loss_model"] = loss_model
    groups = variable_groups(config)
    stator_path = "stages.0.blade_geometry.stator.maximum_thickness_to_chord"
    rotor_path = "stages.0.blade_geometry.rotor.maximum_thickness_to_chord"
    assert stator_path not in groups["all"]["paths"]
    assert (rotor_path in groups["all"]["paths"]) == (loss_model != "isentropic")
    result = compute_turbine_performance(config)
    changed = deepcopy(config)
    opt.set_path(changed["inputs"], stator_path, .2)
    stator_changed = compute_turbine_performance(changed)
    assert stator_changed["overall_performance"] == pytest.approx(result["overall_performance"])
    opt.set_path(changed["inputs"], rotor_path, .2)
    rotor_changed = compute_turbine_performance(changed)
    difference = abs(rotor_changed["overall_performance"]["efficiency_ts"] -
                     result["overall_performance"]["efficiency_ts"])
    if loss_model != "isentropic":
        assert difference > 1e-7
        config["inputs"]["inlet_flow_angle"] = -10
        assert stator_path in variable_groups(config)["all"]["paths"]
    else:
        assert difference < 1e-12


def test_display_only_geometry_and_zero_incidence_inputs_are_not_variables():
    config = case()
    result = compute_turbine_performance(config)
    for row in config["inputs"]["stages"][0]["blade_geometry"].values():
        row.update(maximum_thickness_location=.4, leading_edge_radius_to_max_thickness=.4,
                   leading_edge_wedge_angle=20, trailing_edge_wedge_angle=10)
    changed = compute_turbine_performance(config)
    assert changed["overall_performance"] == pytest.approx(result["overall_performance"], abs=1e-12)
    catalog = variable_catalog(config)
    assert not any(key in path for path in catalog for key in (
        "thickness_location", "leading_edge", "wedge_angle"
    ))


@pytest.mark.parametrize("kind", ["axial", "radial"])
def test_constraint_presets_resolve_into_physical_two_sided_limits(kind):
    config = case(kind, count=2)
    result = compute_turbine_performance(config)
    catalog = output_catalog(config, result)
    groups = constraint_groups(config, catalog)
    flaring = groups["flaring_limits"]["rows"]
    assert [(row["comparison"], row["target"]) for row in flaring] == [(">=", -10), ("<=", 10)]
    assert len(groups["flaring_limits"]["paths"]) == 1
    assert all(row["scale"] == 10 for row in flaring)
    for group in groups.values():
        for row in group["rows"]:
            values = opt.resolve_value(row["path"], result)
            assert np.all(np.isfinite(values))
            assert row["comparison"] in ("<=", ">=", "=")
            assert row["scale"] > 0
    mach = groups["subsonic"]["rows"][0]
    assert (mach["comparison"], mach["target"], mach["scale"]) == ("<=", 1, 1)
    height = groups["minimum_height"]["rows"][0]
    assert (height["comparison"], height["target"], height["scale"]) == (">=", .001, .01)
    rows = [*groups["subsonic"]["rows"], *flaring, height]
    settings = build_settings(
        config, [dict(path="blade_velocity_ratio", lower=.5, upper=.7)], rows,
        "$overall_performance.efficiency_ts", "maximize", 1e-8, 50, 1e-6, result,
    )
    equalities, inequalities, report = opt.evaluate_constraints(result, settings["constraints"])
    # Every wildcard flaring/height row expands over two rows in both stages.
    assert len(settings["constraints"]) == 4
    assert len(equalities) == 0
    assert len(inequalities) == len(report) == 13
    assert np.all(np.isfinite(inequalities))
    np.testing.assert_allclose(inequalities[1:5] + inequalities[5:9], -2)


def test_group_generation_honors_filtered_catalogs_without_mutation():
    config = case(count=2)
    variables = variable_catalog(config)
    variable_subset = {"stages.1.degree_reaction": variables["stages.1.degree_reaction"]}
    before = deepcopy(variable_subset)
    groups = variable_groups(config, variable_subset)
    assert set(groups) == {"stage_design", "stage_1_design", "all"}
    assert all(group["paths"] == ["stages.1.degree_reaction"] for group in groups.values())
    assert variable_subset == before
    outputs = output_catalog(config)
    key = "$stages_performance.*.geometry_meanline.*.flaring_angle"
    output_subset = {key: outputs[key]}
    before = deepcopy(output_subset)
    limits = constraint_groups(config, output_subset)
    assert set(limits) == {"flaring_limits"}
    assert len(limits["flaring_limits"]["rows"]) == 2
    assert output_subset == before
    assert variable_groups(config, {}) == {}
    assert constraint_groups(config, {}) == {}


def test_hub_tip_constraints_resolve_all_four_boundaries_in_each_stage():
    config = case(count=2)
    result = compute_turbine_performance(config)
    catalog = output_catalog(config, result)
    groups = constraint_groups(config, catalog)
    rows = groups["hub_tip_ratios"]["rows"]
    assert len(rows) == 4
    for blade_row in ("stator", "rotor"):
        for boundary in ("in", "out"):
            path = f"$stages_performance.*.geometry_meanline.{blade_row}.hub_tip_ratio_{boundary}"
            assert path in groups["hub_tip_ratios"]["paths"]
            for stage in range(2):
                individual = path.replace(".*.", f".{stage}.")
                assert individual in groups[f"stage_{stage}_hub_tip_ratios"]["paths"]
                assert catalog[individual]["scale"] == 1
                assert catalog[individual]["comparison"] == ">="
                assert catalog[individual]["target"] == .8
    settings = build_settings(
        config, [dict(path="blade_velocity_ratio", lower=.5, upper=.7)], rows,
        "$overall_performance.efficiency_ts", "maximize", 1e-8, 50, 1e-6, result,
    )
    equalities, inequalities, report = opt.evaluate_constraints(result, settings["constraints"])
    assert len(equalities) == 0
    assert len(inequalities) == len(report) == 8
    expected = [
        result["stages_performance"][i]["geometry_meanline"][row][f"hub_tip_ratio_{end}"]
        for row in ("stator", "rotor") for end in ("in", "out") for i in range(2)
    ]
    np.testing.assert_allclose(inequalities, .8 - np.array(expected))
    # A maximum and an equality are supported independently of the preset minimum.
    edited = deepcopy(groups["stage_1_hub_tip_ratios"]["rows"])
    edited[0].update(comparison="<=", target=.9)
    edited[1].update(comparison="=", target=.8)
    settings = build_settings(
        config, [], edited, "$overall_performance.efficiency_ts", "maximize",
        1e-8, 50, 1e-6, result, require_variables=False,
    )
    equalities, inequalities, report = opt.evaluate_constraints(result, settings["constraints"])
    assert len(equalities) == 1 and len(inequalities) == 3
    geometry = result["stages_performance"][1]["geometry_meanline"]["stator"]
    assert inequalities[0] == pytest.approx(geometry["hub_tip_ratio_in"] - .9)
    assert equalities[0] == pytest.approx(geometry["hub_tip_ratio_out"] - .8)


def test_hub_tip_group_toggle_preserves_targets_and_other_constraints():
    config = case()
    catalog = output_catalog(config)
    groups = constraint_groups(config, catalog)
    rows = apply_selection_operations([], catalog, groups, "constraint", [
        dict(kind="constraint", group="subsonic", selected=True),
        dict(kind="constraint", group="hub_tip_ratios", selected=True),
    ])
    assert len(rows) == 5
    rows[1]["target"] = .7123456789
    before = deepcopy(rows)
    assert apply_selection_operations(rows, catalog, groups, "constraint", [
        dict(kind="constraint", group="hub_tip_ratios", selected=True),
    ]) == before
    remaining = apply_selection_operations(rows, catalog, groups, "constraint", [
        dict(kind="constraint", group="hub_tip_ratios", selected=False),
    ])
    assert remaining == before[:1]


@pytest.mark.parametrize("kind", ["radial", "radial_inflow", "radial_outflow"])
def test_radial_model_does_not_offer_fixed_hub_tip_ratios(kind):
    config = case("radial")
    config["inputs"]["turbine_type"] = kind
    assert not any("hub_tip_ratio" in path for path in output_catalog(config))
    assert not any("hub_tip_ratio" in key for key in constraint_groups(config))
