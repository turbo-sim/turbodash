"""The production editor has one state writer and a distinct commit boundary."""
import base64
from copy import deepcopy
import json
from pathlib import Path

from dash import Dash, no_update
from dash._callback_context import context_value
from dash._utils import AttributeDict
from dash.exceptions import PreventUpdate
import numpy as np
import pytest
import yaml

from turbodash.apps.shared.components import blade_modeller as ui
from turbodash.apps.shared.startup import components
from turbodash.geometry.meanline import initialize_geometry, row_parameter_state, update_geometry_override
from turbodash.geometry.blade_parametrization.parameters import resolve_parameters
from turbodash.visualization import vtk


@pytest.fixture
def results():
    saved = json.loads((Path(__file__).parent / "fixtures" / "meanline_geometry_reference.json").read_text())["eps0"]
    result = saved["continuous"]
    result["stages_performance"][0]["geometry_meanline"] = deepcopy(result["stages_performance"][0]["geometry"])
    result["stages_performance"][0]["geometry_discrete"] = saved["integer_geometry"]
    return result


@pytest.fixture
def state(results):
    value = initialize_geometry(results)
    value["_epoch"] = "accepted-calculation-1"
    return value


@pytest.fixture
def editor():
    app = Dash(__name__, suppress_callback_exceptions=True)
    app.layout = ui.geometry_panel()
    ui.register_callbacks(app)
    callback = next(value["callback"].__wrapped__ for key, value in app.callback_map.items()
                    if "geometry_state.data" in key)
    return callback


def invoke(editor, trigger, state, *, committed=None, results=None, value=None,
           stage=0, row="stator", drag=None, numeric=None, upload=None, journal=None, result_timestamp=None):
    row_id = ui.selected_row(state, stage, row)["id"]
    event = journal
    action = None
    epoch = state["_epoch"]
    dragging = trigger == "blade_modeller_drag"
    if isinstance(trigger, dict):
        action = {"kind": "reset" if trigger["type"] == "blade-reset" else "set",
                  "path": trigger["path"], "value": value}
    elif trigger in ("blade_modeller_numeric", "blade_modeller_drag"):
        payload = drag if dragging else numeric
        action = {"kind": "set", "path": payload["path"], "value": payload["value"]}
        epoch = payload["epoch"]
    elif trigger == "blade_modeller_reset":
        action = {"kind": "reset"}
    elif trigger == "blade_modeller_upload":
        action = {"kind": "import", "contents": upload}
    if action:
        action.update(row=row_id, sequence=state["revision"] + 1)
        event = {"epoch": epoch, "sequence": action["sequence"], "dragging": dragging,
                 "base": state, "actions": [action], "latest": action}
        trigger = "blade_modeller_events"
    identity = json.dumps(trigger, sort_keys=True, separators=(",", ":")) if isinstance(trigger, dict) else trigger
    token = context_value.set(AttributeDict(triggered_inputs=[{"prop_id": identity + ".value", "value": value}]))
    try:
        return editor(results, stage, row, event, [{"type": "blade-parameter", "path": path} for path in ui.PATHS],
                      [{"type": "blade-slider", "path": path} for path in ui.SLIDER_PATHS],
                      state, state if committed is None else committed, result_timestamp)
    finally:
        context_value.reset(token)


def value(state, path, row_id="stage_01/stator"):
    return ui._lookup(resolve_parameters(row_parameter_state(state, row_id)).values, path)


def test_inline_numeric_fields_allow_arbitrary_floats():
    fields = {item.id["path"]: item for item in components(ui.controls())
              if isinstance(getattr(item, "id", None), dict) and item.id.get("type") == "blade-parameter"}
    assert fields["thickness.leading_edge_fullness"].step == "any"
    assert fields["geometry.mean_radius"].step == "any"
    assert fields["cascade.blade_count"].step == 1
    assert ui.slider_settings("thickness.leading_edge_fullness", .123456789)["step"] == .01
    assert ui.from_control("geometry.mean_radius", 51.234567891234) == 0.051234567891234


def test_preview_uses_stable_shared_vtk_host_and_committed_geometry(monkeypatch):
    monkeypatch.setattr(vtk, "vtk_available", lambda: True)
    app = Dash(__name__, suppress_callback_exceptions=True)
    app.layout = ui.geometry_panel()
    ui.register_callbacks(app)
    preview = next(item for item in components(app.layout)
                   if getattr(item, "id", None) == "blade_modeller_preview")
    children = list(components(preview))
    identifiers = [getattr(item, "id", None) for item in children]
    for required in ("geometry-scene", "geometry-vtk-data", "geometry-vtk-display-state",
                     "geometry-vtk-rows", "geometry-vtk-copy-controls", "geometry-vtk-fit"):
        assert identifiers.count(required) == 1
    assert not any(item.to_plotly_json()["type"] in ("Graph", "View") for item in children)
    assert all("geometry-scene.children" not in key and "blade_modeller_preview.children" not in key
               and "blade_modeller_preview.figure" not in key for key in app.callback_map)
    prepare = app.callback_map["blade_modeller_3d_proposal.data"]
    assert {item["id"] for item in prepare["inputs"]} == {
        "geometry_committed", "geometry_stage", "geometry_row"}
    assert any("geometry-vtk-data.data" in key for key in app.callback_map)


def test_missing_vtk_dependency_keeps_editor_and_explains_install(monkeypatch):
    monkeypatch.setattr(vtk, "vtk_available", lambda: False)
    panel = ui.geometry_panel()
    children = list(components(panel))
    note = next(item for item in children if getattr(item, "id", None) == "geometry-vtk-note")
    assert "Install the Turbodash mesh extra" in note.children
    assert any(getattr(item, "id", None) == "geometry-scene" for item in children)
    assert any(getattr(item, "id", None) == "geometry_profile" for item in children)


def test_precise_edit_updates_only_one_row_without_reformatting_other_inputs(editor, state):
    path = "camberline.turning_location"
    precise = .512345678912345
    output = invoke(editor, {"type": "blade-parameter", "path": path}, state, value=precise)
    live, committed = output[:2]
    assert value(live, path) == precise
    assert committed == live
    assert live["rows"][1] == state["rows"][1]
    assert all(item is no_update for index, item in enumerate(output[2]) if ui.PATHS[index] != path)
    assert output[3][ui.SLIDER_PATHS.index(path)] == precise
    assert value(state, path) != precise


def test_live_drag_defers_commit_until_release(editor, state):
    path = "camberline.turning_location"
    event = {"row": "stage_01/stator", "epoch": state["_epoch"], "path": path, "value": .55}
    output = invoke(editor, "blade_modeller_drag", state, drag=event)
    live = output[0]
    assert value(live, path) == .55
    assert output[1] is no_update
    assert all(item is no_update for item in output[3])
    committed = invoke(editor, {"type": "blade-slider", "path": path}, live,
                       committed=state, value=.55)
    assert committed[1]["rows"] == live["rows"]
    assert committed[1]["revision"] > live["revision"]


def test_independent_fullness_and_invalid_combination_keeps_effective_state(editor, state):
    path = "thickness.leading_edge_fullness"
    output = invoke(editor, {"type": "blade-parameter", "path": path}, state, value=.1)
    changed = output[0]
    assert value(changed, path) == .1
    assert value(changed, "thickness.trailing_edge_fullness") == value(state, "thickness.trailing_edge_fullness")
    rejected = invoke(editor, {"type": "blade-parameter", "path": path}, changed, value=.9)
    assert rejected[0]["rows"] == changed["rows"] == rejected[1]["rows"]
    assert "not applied" in rejected[10]


def test_section_failure_with_admissible_parameter_domains_preserves_both_views(editor, state):
    # Each value satisfies FIELD_SPECS, but automatic pressure-side selection
    # is undefined for a zero-turning blade when relief becomes nonzero.
    row_id = "stage_01/stator"
    state = update_geometry_override(state, row_id, "camberline.inlet_metal_angle_deg", 20.)
    state = update_geometry_override(state, row_id, "camberline.outlet_metal_angle_deg", 20.)
    candidate = update_geometry_override(state, row_id, "leading_edge.relief_strength", .2)
    resolve_parameters(row_parameter_state(candidate, row_id))
    output = invoke(editor, {"type": "blade-parameter", "path": "leading_edge.relief_strength"}, state, value=.2)
    assert output[0]["rows"] == state["rows"] == output[1]["rows"]
    assert "zero-turning" in output[10]
    assert value(state, "leading_edge.relief_strength") == 0.


def test_selection_preserves_overrides_and_new_success_clears_them(editor, state, results):
    edited = update_geometry_override(state, "stage_01/stator", "camberline.turning_location", .6)
    selection = invoke(editor, "geometry_row", edited, row="rotor")
    assert selection[0] is no_update and selection[1] is no_update
    assert selection[11] == "stage_01/rotor"
    reset = invoke(editor, "result_store", edited, results=results)[0]
    assert reset["baseline_id"] == edited["baseline_id"]
    assert reset["_epoch"] != edited["_epoch"]
    assert all(not item["parameters"]["overrides"] for item in reset["rows"])


def test_failed_result_and_previous_epoch_event_cannot_replace_geometry(editor, state):
    with pytest.raises(PreventUpdate):
        invoke(editor, "result_store", state, results=None)
    with pytest.raises(PreventUpdate):
        invoke(editor, "blade_modeller_drag", state, drag={
            "row": "stage_01/stator", "epoch": "old-calculation", "path": "camberline.turning_location", "value": .6})


def test_field_and_row_reset_restore_baseline(editor, state):
    updated = update_geometry_override(state, "stage_01/stator", "camberline.turning_location", .6)
    updated = update_geometry_override(updated, "stage_01/stator", "thickness.leading_edge_fullness", .1)
    output = invoke(editor, {"type": "blade-reset", "path": "camberline.turning_location"}, updated)
    assert value(output[0], "camberline.turning_location") == value(state, "camberline.turning_location")
    assert value(output[0], "thickness.leading_edge_fullness") == .1
    reset = invoke(editor, "blade_modeller_reset", output[0])[0]
    assert not reset["rows"][0]["parameters"]["overrides"]


def test_numeric_commit_fallback_preserves_precise_reentered_number(editor, state):
    path = "thickness.leading_edge_fullness"
    event = {"row": "stage_01/stator", "epoch": state["_epoch"], "path": path, "value": .123456789123}
    updated = invoke(editor, "blade_modeller_numeric", state, numeric=event)[0]
    assert value(updated, path) == .123456789123
    duplicate = {"epoch": updated["_epoch"], "sequence": updated["revision"], "base": state,
                 "actions": [], "latest": {}}
    with pytest.raises(PreventUpdate):
        invoke(editor, "blade_modeller_events", updated, journal=duplicate)


def test_high_resolution_views_and_preview_use_current_effective_shape(state):
    row_id = "stage_01/stator"
    before = ui.evaluate_2d(state, row_id)
    edited = update_geometry_override(state, row_id, "camberline.turning_location", .57)
    after = ui.evaluate_2d(edited, row_id)
    assert before["token"] != after["token"]
    original = np.asarray(before["figures"]["camberline"].data[0].y)
    changed = np.asarray(after["figures"]["camberline"].data[0].y)
    assert len(changed) >= 601 and not np.array_equal(original, changed)
    assert all(trace.line.simplify is False for figure in after["figures"].values() for trace in figure.data)
    original_preview = ui.evaluate_3d(state, row_id)
    preview = ui.evaluate_3d(edited, row_id)
    assert preview["token"] == after["token"]
    assert preview["data"]["key"] != original_preview["data"]["key"]
    assert len(preview["data"]["rows"]) == 1
    row = preview["data"]["rows"][0]
    assert row["id"] == "stage_1_stator" and row["geometry_type"] == "axial"
    assert row["num_blades"] == value(edited, "cascade.blade_count")
    assert {"blade", "hub", "shroud"} <= row["surfaces"].keys()
    points = np.asarray(row["surfaces"]["blade"]["points"]).reshape(-1, 3)
    original_points = original_preview["data"]["rows"][0]["surfaces"]["blade"]["points"]
    assert len(points) >= 1200 and np.isfinite(points).all()
    assert not np.array_equal(points.ravel(), original_points)
    assert len(row["surfaces"]["blade"]["polys"]) > 0


def test_yaml_import_retains_meanline_baseline(state):
    # Use a mild row, so imported validation tests the file transaction rather
    # than the highly turned reference fixture's aerodynamic acceptability.
    imported = row_parameter_state(state, "stage_01/stator").to_dict()
    imported["overrides"].update({"thickness.maximum_thickness_ratio": .05,
                                  "camberline.inlet_metal_angle_deg": 0.,
                                  "camberline.outlet_metal_angle_deg": 20.})
    content = "data:application/yaml;base64," + base64.b64encode(yaml.safe_dump(imported).encode()).decode()
    changed = ui.import_row_yaml(content, state, "stage_01/stator")
    assert changed["rows"][0]["parameters"]["source"] == state["rows"][0]["parameters"]["source"]
    assert value(changed, "thickness.maximum_thickness_ratio") == .05
    assert changed["rows"][1] == state["rows"][1]
    with pytest.raises(ValueError, match="Could not import"):
        ui.import_row_yaml("data:application/yaml;base64,%%invalid%%", state, "stage_01/stator")


def test_multiple_stage_rows_retain_independent_edits(editor, results):
    results["stages_performance"].append(deepcopy(results["stages_performance"][0]))
    state = initialize_geometry(results)
    state["_epoch"] = "two-stage-calculation"
    first = update_geometry_override(state, "stage_01/stator", "camberline.turning_location", .55)
    second = invoke(editor, {"type": "blade-parameter", "path": "camberline.turning_location"},
                    first, stage=1, row="rotor", value=.62)[0]
    assert value(second, "camberline.turning_location", "stage_01/stator") == .55
    assert value(second, "camberline.turning_location", "stage_02/rotor") == .62
    assert second["rows"][1] == state["rows"][1]
    assert second["rows"][2] == state["rows"][2]


def test_radial_preview_uses_same_effective_parameters(state):
    row_id = "stage_01/stator"
    for path, edited in (("geometry.type", "radial"), ("geometry.inlet_radius", .06),
                         ("geometry.outlet_radius", .04), ("geometry.blade_height", .005),
                         ("thickness.maximum_thickness_ratio", .05)):
        # Set radii before switching stream surface because equal radii are
        # admissible only while the independent geometry remains axial.
        if path == "geometry.type":
            continue
        state = update_geometry_override(state, row_id, path, edited)
    state = update_geometry_override(state, row_id, "geometry.type", "radial")
    before = ui.evaluate_2d(state, row_id)
    preview = ui.evaluate_3d(state, row_id)
    assert preview["token"] == before["token"]
    row = preview["data"]["rows"][0]
    assert row["geometry_type"] == "radial" and not row["is_linear"]
    z = np.asarray(row["surfaces"]["blade"]["points"]).reshape(-1, 3)[:, 2]
    assert np.isclose(z.min(), 0.) and np.isclose(z.max(), 5.)


def journal(state, actions, *, sequence=None):
    pending = [{**action, "row": action.get("row", "stage_01/stator"),
                "sequence": state["revision"] + index + 1} for index, action in enumerate(actions)]
    return {"epoch": state["_epoch"], "base": deepcopy(state), "actions": pending,
            "sequence": sequence or pending[-1]["sequence"], "latest": pending[-1], "dragging": False}


def test_two_rapid_fields_fold_cumulative_pending_actions_without_waiting(editor, state):
    pending = journal(state, [
        {"kind": "set", "path": "camberline.turning_location", "value": .56789123},
        {"kind": "set", "path": "thickness.trailing_edge_fullness", "value": .12345678},
    ])
    # Both requests were sent with State=current at revision zero.
    first_event = {**pending, "sequence": 1, "actions": pending["actions"][:1], "latest": pending["actions"][0]}
    first = invoke(editor, "blade_modeller_events", state, journal=first_event)[0]
    second = invoke(editor, "blade_modeller_events", state, journal=pending)[0]
    assert value(first, "camberline.turning_location") == .56789123
    assert value(second, "camberline.turning_location") == .56789123
    assert value(second, "thickness.trailing_edge_fullness") == .12345678
    assert second["revision"] == 2
    with pytest.raises(PreventUpdate):
        invoke(editor, "blade_modeller_events", second, journal=first_event)


def test_invalid_pending_value_and_import_retain_previous_valid_override(editor, state):
    pending = journal(state, [
        {"kind": "set", "path": "thickness.leading_edge_fullness", "value": .12},
        {"kind": "set", "path": "thickness.leading_edge_fullness", "value": .9},
        {"kind": "import", "contents": "data:application/yaml;base64,%%invalid%%"},
    ])
    output = invoke(editor, "blade_modeller_events", state, journal=pending)
    assert value(output[0], "thickness.leading_edge_fullness") == .12
    assert output[0]["revision"] == 3
    # The acknowledged valid base drops rejected history before the next edit.
    next_event = journal(output[0], [{"kind": "set", "path": "camberline.turning_location", "value": .6}])
    following = invoke(editor, "blade_modeller_events", output[0], journal=next_event)
    assert value(following[0], "thickness.leading_edge_fullness") == .12
    assert following[10] == ""


def test_pending_reset_boundary_does_not_replay_cleared_edits(editor, state):
    state = update_geometry_override(state, "stage_01/stator", "thickness.leading_edge_fullness", .2)
    pending = journal(state, [
        {"kind": "set", "path": "thickness.leading_edge_fullness", "value": .12},
        {"kind": "reset"},
        {"kind": "set", "path": "camberline.turning_location", "value": .6},
    ])
    output = invoke(editor, "blade_modeller_events", state, journal=pending)
    assert value(output[0], "thickness.leading_edge_fullness") == 0.
    assert value(output[0], "camberline.turning_location") == .6
    assert output[2][ui.PATHS.index("thickness.leading_edge_fullness")] == 0.


def test_new_accepted_result_supersedes_edit_before_baseline_callback_finishes(editor, state, results):
    state["_result_timestamp"] = 100
    pending = journal(state, [{"kind": "set", "path": "camberline.turning_location", "value": .6}])
    output = invoke(editor, "blade_modeller_events", state, journal=pending,
                    results=results, result_timestamp=200)[0]
    assert output["baseline_id"] == state["baseline_id"]
    assert output["_epoch"] != state["_epoch"] and output["_result_timestamp"] == 200
    assert all(not row["parameters"]["overrides"] for row in output["rows"])


def test_initial_selector_update_initializes_available_result_even_without_state(editor, results):
    token = context_value.set(AttributeDict(triggered_inputs=[{"prop_id": "geometry_stage.value", "value": 0}]))
    try:
        output = editor(results, 0, "stator", None, [], [], None, None, 100)
    finally:
        context_value.reset(token)
    assert len(output[0]["rows"]) == 2 and output[1] == output[0]
    assert output[0]["_result_timestamp"] == 100


def test_save_includes_numeric_blur_edit_before_editor_response(state):
    app = Dash(__name__, suppress_callback_exceptions=True)
    app.layout = ui.geometry_panel()
    ui.register_callbacks(app)
    save = app.callback_map["blade_modeller_download.data"]["callback"].__wrapped__
    pending = journal(state, [{"kind": "set", "path": "camberline.turning_location", "value": .567891234}])
    downloaded = yaml.safe_load(save(1, state, 0, "stator", pending, None)["content"])
    assert downloaded["overrides"]["camberline.turning_location"] == .567891234
    with pytest.raises(PreventUpdate):
        save(1, state, 0, "stator", pending, 200)


def test_pending_import_then_edit_projects_all_imported_controls(editor, state):
    imported = row_parameter_state(state, "stage_01/stator").to_dict()
    imported["overrides"].update({"thickness.maximum_thickness_ratio": .05,
                                  "camberline.inlet_metal_angle_deg": 0.,
                                  "camberline.outlet_metal_angle_deg": 20.})
    contents = "data:application/yaml;base64," + base64.b64encode(yaml.safe_dump(imported).encode()).decode()
    pending = journal(state, [{"kind": "import", "contents": contents},
                              {"kind": "set", "path": "camberline.turning_location", "value": .6}])
    output = invoke(editor, "blade_modeller_events", state, journal=pending)
    assert value(output[0], "thickness.maximum_thickness_ratio") == .05
    assert output[2][ui.PATHS.index("thickness.maximum_thickness_ratio")] == .05
    assert output[2][ui.PATHS.index("camberline.outlet_metal_angle_deg")] == 20.
