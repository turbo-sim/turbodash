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


@pytest.mark.parametrize("kind", ["axial", "radial_inflow", "radial_outflow"])
@pytest.mark.parametrize("speed_type,speed", [("specific", .7), ("actual", 30000)])
def test_specific_speed_upper_bound_is_one_only_for_radial_outflow(kind, speed_type, speed):
    config = case("axial" if kind == "axial" else "radial")
    config["inputs"].update(rotational_speed_type=speed_type, rotational_speed=speed)
    if kind == "radial_outflow":
        config["inputs"]["stages"][0].update(
            radius_ratio_12=.8, radius_ratio_23=.95, radius_ratio_34=.8,
        )
    catalog = variable_catalog(config)
    rows = apply_selection_operations([], catalog, variable_groups(config, catalog), "variable", [
        dict(kind="variable", group="overall", selected=True),
    ])
    selected = next(row for row in rows if row["path"] == "rotational_speed")
    expected = speed * 1.5 if speed_type == "actual" else (1 if kind == "radial_outflow" else 1.5)
    assert selected["upper"] == expected
    assert selected["lower"] == (speed * .5 if speed_type == "actual" else .2)


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
        if ratio < 1:
            assert (entry["lower"], entry["upper"]) == (.75, .85)
        else:
            assert entry["lower"] < ratio < entry["upper"]
            assert 1 < entry["lower"] < entry["upper"]


@pytest.mark.parametrize("group", ["radial_radii", "all"])
def test_radial_outflow_groups_use_requested_radius_variable_bounds(group):
    config = case("radial", count=3)
    for stage in config["inputs"]["stages"]:
        stage.update(radius_ratio_12=.8, radius_ratio_23=.95, radius_ratio_34=.8)
    original = deepcopy(config)
    catalog = variable_catalog(config)
    groups = variable_groups(config, catalog)
    rows = apply_selection_operations([], catalog, groups, "variable", [
        dict(kind="variable", group=group, selected=True),
    ])
    selected = {row["path"]: row for row in rows}
    for index in range(3):
        for pair, bounds in (("12", (.75, .85)), ("23", (.925, .975)), ("34", (.75, .85))):
            row = selected[f"stages.{index}.radius_ratio_{pair}"]
            assert (row["lower"], row["upper"]) == bounds
    # Reapplying a group preserves bounds the user has already edited.
    selected["stages.0.radius_ratio_12"]["lower"] = .78
    assert apply_selection_operations(rows, catalog, groups, "variable", [
        dict(kind="variable", group=group, selected=True),
    ]) == rows
    assert config == original


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
    assert [(row["comparison"], row["target"]) for row in flaring] == [(">=", -10), ("<=", 10)] * 2
    assert len(groups["flaring_limits"]["paths"]) == 2
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
    # Four blade rows and three interspaces have distinct two-sided limits.
    assert len(settings["constraints"]) == 6
    assert len(equalities) == 0
    assert len(inequalities) == len(report) == 19
    assert np.all(np.isfinite(inequalities))
    np.testing.assert_allclose(inequalities[1:5] + inequalities[5:9], -2)
    np.testing.assert_allclose(inequalities[9:12] + inequalities[12:15], -2)


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


@pytest.mark.parametrize("kind", ["axial", "radial"])
def test_flow_angle_constraints_and_turning_use_each_rows_physical_frame(kind):
    config = case(kind, count=2)
    result = compute_turbine_performance(config)
    catalog = output_catalog(config, result)
    groups = constraint_groups(config, catalog)
    for index, stage in enumerate(result["stages_performance"]):
        stations = stage["flow_stations"]
        for station in range(4):
            for key in ("alpha", "beta"):
                path = f"$stages_performance.{index}.flow_stations.{station}.{key}"
                assert catalog[path]["scale"] == 90
                assert opt.read_path(result, path) == stations[station][key]
                wildcard = path.replace(f".{index}.", ".*.", 1)
                assert len(opt.resolve_value(wildcard, result)) == 2
        for blade_row, start, angle in (("stator", 0, "alpha"), ("rotor", 2, "beta")):
            expected = abs(stations[start + 1][angle] - stations[start][angle])
            assert stage["geometry_meanline"][blade_row]["flow_turning"] == pytest.approx(expected)
            inlet = groups[f"stage_{index}_{blade_row}_inlet_relative_angle"]["rows"]
            assert inlet == [dict(path=f"$stages_performance.{index}.flow_stations.{start}.beta",
                                  comparison=">=", target=0, scale=90)]
        assert stations[0]["alpha"] == stations[0]["beta"]
    rows = groups["minimum_turning"]["rows"]
    assert len(rows) == 2
    assert all((row["comparison"], row["target"], row["scale"]) == (">=", 50, 90) for row in rows)
    settings = build_settings(
        config, [dict(path="blade_velocity_ratio", lower=.5, upper=.7)], rows,
        "$overall_performance.efficiency_ts_with_recovery", "maximize", 1e-8, 50, 1e-6, result,
    )
    equalities, inequalities, report = opt.evaluate_constraints(result, settings["constraints"])
    expected = [stage["geometry_meanline"][row]["flow_turning"]
                for row in ("stator", "rotor") for stage in result["stages_performance"]]
    assert len(equalities) == 0 and len(inequalities) == len(report) == 4
    np.testing.assert_allclose(inequalities, (50 - np.array(expected)) / 90)
    assert "stage_1_minimum_turning" in groups


@pytest.mark.parametrize("incoming,outgoing,allowed", [(20, 22, False), (-20, -18, False),
    (-25, 25, True), (25, -25, True), (-20, 20, False), (0, 0, False)])
def test_minimum_turning_rejects_nearly_straight_rows_for_either_angle_sign(incoming, outgoing, allowed):
    from turbodash.core.turbine import compute_blade_row_geometry
    blade = case()["inputs"]["stages"][0]["blade_geometry"]["stator"]
    geometry = compute_blade_row_geometry("axial", .1, .1, .02, .02,
                                          incoming, outgoing, blade_geometry=blade)
    result = {"stages_performance": [{"geometry_meanline": {"stator": geometry}}]}
    _, residuals, _ = opt.evaluate_constraints(result, [dict(
        variable="$stages_performance.0.geometry_meanline.stator.flow_turning",
        type=">=", value=50, normalize=90,
    )])
    assert bool(residuals[0] <= 0) is allowed


def test_inlet_and_turning_groups_toggle_independently_and_preserve_edits():
    config = case()
    catalog, groups = output_catalog(config), constraint_groups(config)
    rows = apply_selection_operations([], catalog, groups, "constraint", [
        dict(kind="constraint", group="minimum_turning", selected=True),
        dict(kind="constraint", group="stator_inlet_relative_angle", selected=True),
        dict(kind="constraint", group="rotor_inlet_relative_angle", selected=True),
    ])
    assert len(rows) == 4
    rows[0]["target"] = 35.123456789
    before = deepcopy(rows)
    assert apply_selection_operations(rows, catalog, groups, "constraint", [
        dict(kind="constraint", group="minimum_turning", selected=True),
    ]) == before
    assert apply_selection_operations(rows, catalog, groups, "constraint", [
        dict(kind="constraint", group="rotor_inlet_relative_angle", selected=False),
    ]) == before[:3]


@pytest.mark.parametrize("kind", ["axial", "radial"])
def test_interspace_groups_cover_each_actual_gap_and_spacing_is_never_a_variable(kind):
    config = case(kind, count=2)
    if kind == "axial":
        config["inputs"]["stages"][1]["axial_spacing_before_stator"] = .4123456789
    result = compute_turbine_performance(config)
    outputs = output_catalog(config, result)
    variables = variable_catalog(config)
    assert not any("axial_spacing" in path for path in variables)
    assert not any("axial_spacing" in path for path in variable_groups(config)["all"]["paths"])
    assert not any(".0.geometry_interspaces.before_stator." in path for path in outputs)
    assert "$stages_performance.1.geometry_interspaces.before_stator.flaring_angle" in outputs
    assert "$stages_performance.0.geometry_interspaces.before_rotor.flaring_angle" in outputs
    groups = constraint_groups(config, outputs)
    assert "interspace_flaring_limits" not in groups
    group = groups["flaring_limits"]
    assert len(group["rows"]) == 4
    gap_rows = [row for row in group["rows"] if ".geometry_interspaces." in row["path"]]
    assert len(gap_rows) == 2
    for row in gap_rows:
        assert row["scale"] == 10
        assert np.asarray(opt.resolve_value(row["path"], result)).size == 3
    for index, count in ((0, 1), (1, 2)):
        rows = groups[f"stage_{index}_limits"]["rows"]
        gap_rows = [row for row in rows if ".geometry_interspaces." in row["path"]]
        assert len(gap_rows) == 2
        assert all(np.asarray(opt.resolve_value(row["path"], result)).size == count for row in gap_rows)


def _components(component):
    if hasattr(component, "to_plotly_json"):
        yield component
        yield from _components(getattr(component, "children", None))
    elif isinstance(component, (list, tuple)):
        for child in component:
            yield from _components(child)


@pytest.mark.parametrize("kind", ["axial", "radial"])
def test_spacing_inputs_roundtrip_and_only_mount_applicable_rows(kind):
    from turbodash.apps.app_turbine_optimization import app as module
    config = case(kind, count=2)
    if kind == "axial":
        config["inputs"]["stages"][0]["axial_spacing_before_rotor"] = .3123456789
        config["inputs"]["stages"][1]["axial_spacing_before_stator"] = .4123456789
        config["inputs"]["stages"][1]["axial_spacing_before_rotor"] = .5123456789
    before = deepcopy(config)
    stages = module.loaded_stages(config)
    editor, _, _ = module.render_stage_accordion(stages, kind)
    inputs = [component for component in _components(editor)
              if isinstance(getattr(component, "id", None), dict)
              and component.id.get("elem") == "input"]
    spacings = {(component.id["stage"], component.id["key"]): component.value
                for component in inputs if component.id["key"].startswith("axial_spacing_")}
    expected = {(0, "axial_spacing_before_rotor"): .3123456789,
                (1, "axial_spacing_before_stator"): .4123456789,
                (1, "axial_spacing_before_rotor"): .5123456789} if kind == "axial" else {}
    assert spacings == expected
    assert not any(getattr(component, "children", None) == "Axial spacing"
                   for component in _components(editor))
    for index, stage_editor in enumerate(editor):
        for row_editor in _components(stage_editor):
            row = row_editor.to_plotly_json()["props"].get("data-blade-row")
            if row not in ("stator", "rotor"):
                continue
            keys = {component.id["key"] for component in _components(row_editor)
                    if isinstance(getattr(component, "id", None), dict)
                    and component.id.get("elem") == "input"}
            spacing_key = f"axial_spacing_before_{row}"
            assert (spacing_key in keys) == (kind == "axial" and (row == "rotor" or index > 0))
    overall_ids = [dict(scope="overall", key=key) for key in module.OVERALL_DEFAULTS]
    rebuilt = module.live_config(module.loaded_overall_values(config, overall_ids),
                                 [component.value for component in inputs], overall_ids,
                                 [component.id for component in inputs], stages)
    for index, stage in enumerate(rebuilt["inputs"]["stages"]):
        for field in ("axial_spacing_before_stator", "axial_spacing_before_rotor"):
            if (index, field) in expected:
                assert stage[field] == expected[index, field]
            else:
                assert field not in stage
    assert config == before
    if kind == "axial":
        bad = deepcopy(stages)
        bad[0]["axial_spacing_before_rotor"] = 0
        with pytest.raises(ValueError, match="greater than zero"):
            module.live_config(module.loaded_overall_values(config, overall_ids), [],
                               overall_ids, [], bad)


def test_legacy_spacing_load_resolves_once_and_preserves_baseline_physical_gaps():
    from turbodash.apps.app_turbine_optimization import app as module
    config = case(count=2)
    for stage in config["inputs"]["stages"]:
        stage.pop("axial_spacing_before_rotor", None)
        stage.pop("axial_spacing_before_stator", None)
    baseline = compute_turbine_performance(config)
    stages = module.loaded_stages(config)
    assert "axial_spacing_before_rotor" not in config["inputs"]["stages"][0]
    for index, row in ((0, "rotor"), (1, "stator"), (1, "rotor")):
        gap = baseline["stages_performance"][index]["geometry_interspaces"][f"before_{row}"]
        chord = baseline["stages_performance"][index]["geometry_meanline"][row]["chord_meridional"]
        assert stages[index][f"axial_spacing_before_{row}"] * chord == pytest.approx(gap["meridional_length"])
