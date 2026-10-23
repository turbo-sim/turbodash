"""Whole-design YAML preserves the geometry editor's inputs and load lifecycle."""

import base64
from copy import deepcopy
import json
from pathlib import Path

from dash import no_update
from dash._callback_context import context_value
from dash._utils import AttributeDict
from dash.exceptions import PreventUpdate
import pytest
from plotly.utils import PlotlyJSONEncoder
import yaml

from turbodash.apps.app_turbine_optimization import app as application
from turbodash.apps.shared.components import blade_modeller as ui
from turbodash.geometry.blade_parametrization.parameters import resolve_parameters
from turbodash.geometry.meanline import (
    baseline_id, initialize_geometry, row_parameter_state, update_geometry_override,
)
from turbodash.geometry.serialization import (
    export_geometry, restore_geometry, validate_geometry,
)


ROOT = Path(__file__).resolve().parents[1]


def browser_inputs(config):
    identifiers = [{"scope": "overall", "key": key} for key in application.OVERALL_DEFAULTS]
    return (application.loaded_overall_values(config, identifiers), [], identifiers,
            [], application.loaded_stages(config))


def solve_for_browser(config):
    """Match Dash's serialization of solver scalars into result_store."""
    return json.loads(json.dumps(application.core.compute_turbine_performance(config),
                                 cls=PlotlyJSONEncoder))


@pytest.fixture(scope="module", params=["axial", "radial"])
def design(request):
    config = yaml.safe_load((ROOT / f"demos/optimization/case_{request.param}.yaml").read_text())
    second = deepcopy(config["inputs"]["stages"][0])
    second["name"] = "second_stage"
    config["inputs"]["stages"].append(second)
    live = browser_inputs(config)
    current = application.live_config(*live)
    results = solve_for_browser(current)
    state = initialize_geometry(results)
    state.update(_epoch="accepted-design", _result_timestamp=100)
    for index, record in enumerate(state["rows"]):
        changes = {
            "cascade.blade_count": 31 + index,
            "camberline.interpolation": "slope",
            "camberline.turning_location": .51 + .01 * index,
            "camberline.turning_concentration": 1.3 + .1 * index,
            "thickness.maximum_thickness_ratio": .12 + .01 * index,
            "thickness.maximum_thickness_location": .35,
            "thickness.leading_edge_fullness": .02,
            "thickness.trailing_edge_fullness": -.02,
            "thickness.leading_edge_rounding_exponent": 3.2,
            "thickness.trailing_edge_rounding_exponent": 3.4,
            "correction.preserve_physical_thickness": True,
            "correction.warning_tolerance": .017,
            "numerics.number_of_points": 241 + 20 * index,
            "flow_domain.inlet_chords": .7 + .1 * index,
            "flow_domain.outlet_chords": 1.4 + .1 * index,
        }
        for path, value in changes.items():
            state = update_geometry_override(state, record["id"], path, value)
    return dict(config=current, live=live, results=results, state=state)


@pytest.fixture
def notices(monkeypatch):
    messages = []
    monkeypatch.setattr(application, "set_props", lambda identity, props: messages.append((identity, props)))
    return messages


def editor_arguments():
    options = application.solver_editor_options()
    return [*[[] for _ in range(10)], "$overall_performance.efficiency_ts", "maximize",
            1e-8, 150, 1e-6,
            *[options[name] for name, _ in application.SLSQP_EXTRA_CONTROLS], options]


def download(design, state=None, pending=None, timestamp=100, *, legacy=False):
    live = design["live"]
    editor = editor_arguments()
    if not legacy:
        editor.extend([design["state"] if state is None else state, pending, timestamp])
    return application.download_yaml(
        1, design["results"], application.input_signature(*live), *live, *editor)


def upload(document):
    content = document if isinstance(document, str) else yaml.safe_dump(document)
    return "data:application/yaml;base64," + base64.b64encode(content.encode()).decode()


def transition(trigger, results, state, loaded, *, result_timestamp=201, loaded_timestamp=200,
               stage=0, row="stator", event=None, proposal=False):
    prepare = application.app.callback_map["blade_modeller_edit_proposal.data"]["callback"].__wrapped__
    edit = prepare if proposal else prepare.transition
    property_name = "modified_timestamp" if trigger == "loaded_cfg_store" else "data"
    token = context_value.set(AttributeDict(triggered_inputs=[{
        "prop_id": trigger + "." + property_name,
        "value": loaded_timestamp if trigger == "loaded_cfg_store" else results,
    }]))
    try:
        return edit(results, stage, row, event,
                    [{"type": "blade-parameter", "path": path} for path in ui.PATHS],
                    [{"type": "blade-slider", "path": path} for path in ui.SLIDER_PATHS],
                    state, state, result_timestamp, loaded, loaded_timestamp)
    finally:
        context_value.reset(token)


def journal(state, actions, *, epoch=None):
    actions = [dict(action, row=action.get("row", "stage_01/stator"),
                    sequence=state["revision"] + index + 1)
               for index, action in enumerate(actions)]
    return {"epoch": state["_epoch"] if epoch is None else epoch,
            "sequence": actions[-1]["sequence"], "base": state,
            "actions": actions, "latest": actions[-1], "dragging": False}


def test_download_load_and_resave_preserve_every_stage_and_geometry_control(design, notices):
    original_results, original_state = deepcopy(design["results"]), deepcopy(design["state"])
    downloaded = download(design)
    saved = yaml.safe_load(downloaded["content"])
    expected = export_geometry(design["results"], design["state"])
    assert saved["geometry"] == expected
    assert len(saved["geometry"]["stages"]) == 2
    for stage in saved["geometry"]["stages"]:
        for row in ("stator", "rotor"):
            parameters = stage[row]
            assert type(parameters["schema_version"]) is int
            assert type(parameters["overrides"]["cascade.blade_count"]) is int
            assert type(parameters["overrides"]["numerics.number_of_points"]) is int
            assert parameters["overrides"]["correction.preserve_physical_thickness"] is True
            assert parameters["overrides"]["camberline.interpolation"] == "slope"
    loaded = application.load_design(upload(downloaded["content"]))
    assert loaded is not no_update
    assert loaded["geometry"] == expected
    fresh = solve_for_browser({"inputs": application.geometry_load_inputs(loaded)})
    output = transition("result_store", fresh, design["state"], loaded)
    restored = output[0]
    assert output[1] == restored
    assert restored["baseline_id"] == baseline_id(fresh)
    assert restored["_epoch"] != design["state"]["_epoch"]
    assert restored["_result_timestamp"] == 201
    assert restored["_loaded_geometry_timestamp"] == 200
    assert output[-1] == restored["_epoch"]
    assert restored["revision"] == 0
    proposed = transition("result_store", fresh, design["state"], loaded, proposal=True)
    assert proposed["loaded_timestamp"] == 200
    assert proposed["result_timestamp"] == 201
    assert proposed["epoch"] == proposed["values"][0]["_epoch"]
    for index, row in enumerate(restored["rows"]):
        original = design["state"]["rows"][index]["parameters"]
        assert row["parameters"] == original
        resolved = resolve_parameters(row_parameter_state(restored, row["id"]))
        assert resolved.provenance["cascade.blade_count"] == "override"
        assert resolved.provenance["geometry.blade_height"] == "inherited"
        shown = transition("geometry_stage", fresh, restored, loaded,
                           stage=row["stage_index"], row=row["row"])
        assert shown[2] == [ui.to_control(path, ui._lookup(resolved.values, path)) for path in ui.PATHS]
    resaved = download(dict(design, results=fresh, live=browser_inputs(loaded)), restored, timestamp=201)
    assert yaml.safe_load(resaved["content"])["geometry"] == expected
    assert "_load_id" not in yaml.safe_load(resaved["content"])
    assert design["results"] == original_results
    assert design["state"] == original_state


def test_restore_keeps_explicit_override_even_when_equal_to_inherited_value(design):
    state = design["state"]
    row_id = state["rows"][0]["id"]
    inherited = row_parameter_state(state, row_id).source["geometry"]["blade_height"]
    edited = update_geometry_override(state, row_id, "geometry.blade_height", inherited)
    document = yaml.safe_load(yaml.safe_dump(export_geometry(design["results"], edited)))
    restored = restore_geometry(design["results"], document)
    resolved = resolve_parameters(row_parameter_state(restored, row_id))
    assert resolved.values["geometry"]["blade_height"] == inherited
    assert resolved.provenance["geometry.blade_height"] == "override"
    assert restored["rows"][0]["parameters"] == edited["rows"][0]["parameters"]


def test_matching_load_restores_immediately_and_applies_only_to_its_results(design, notices):
    loaded = application.load_design(upload(download(design)["content"]))
    # dcc.Store can retain an old result timestamp when equal inputs yield
    # identical solve output. The existing accepted result is usable at once.
    immediate = transition("loaded_cfg_store", design["results"], design["state"], loaded,
                           result_timestamp=100)[0]
    assert export_geometry(design["results"], immediate) == loaded["geometry"]
    assert immediate["_result_timestamp"] == 100
    assert immediate["_loaded_geometry_timestamp"] == 200
    assert immediate["_loaded_geometry_pending_result"] is True
    unrelated = deepcopy(loaded)
    unrelated["inputs"]["mass_flow_rate"] *= 1.1
    with pytest.raises(PreventUpdate):
        transition("loaded_cfg_store", design["results"], design["state"], unrelated,
                   result_timestamp=100)
    # A matching solve already queued by the upload keeps its geometry once.
    restored = transition("result_store", design["results"], immediate, loaded)[0]
    assert export_geometry(design["results"], restored) == loaded["geometry"]
    assert restored["_loaded_geometry_pending_result"] is False
    assert restored["_epoch"] != immediate["_epoch"]
    refreshed = transition("result_store", design["results"], restored, loaded,
                           result_timestamp=202)[0]
    assert refreshed["_epoch"] != restored["_epoch"]
    assert refreshed["_loaded_geometry_timestamp"] == 200
    assert all(not row["parameters"]["overrides"] for row in refreshed["rows"])
    # A different accepted design also cancels the pending upload. Returning
    # to the uploaded inputs later must keep the new meanline defaults.
    differing_results = solve_for_browser({"inputs": application.geometry_load_inputs(unrelated)})
    differing = transition("result_store", differing_results, immediate, loaded)[0]
    assert differing["baseline_id"] != immediate["baseline_id"]
    assert differing["_loaded_geometry_timestamp"] == 200
    assert not differing.get("_loaded_geometry_pending_result")
    assert all(not row["parameters"]["overrides"] for row in differing["rows"])
    returned = transition("result_store", design["results"], differing, loaded,
                          result_timestamp=202)[0]
    assert returned["baseline_id"] == immediate["baseline_id"]
    assert all(not row["parameters"]["overrides"] for row in returned["rows"])
    # A second upload of the same YAML has its own load event and restores it.
    reloaded = transition("result_store", design["results"], returned, loaded,
                          result_timestamp=301, loaded_timestamp=300)[0]
    assert export_geometry(design["results"], reloaded) == loaded["geometry"]


def test_repeated_identical_uploads_create_distinct_load_events_without_saving_metadata(design, notices):
    content = upload(download(design)["content"])
    first = application.load_design(content)
    second = application.load_design(content)
    assert notices.count(("load_button", {"contents": None})) == 2
    notice_count = len(notices)
    with pytest.raises(PreventUpdate):
        application.load_design(None)
    assert len(notices) == notice_count
    assert first is not no_update and second is not no_update
    assert first["_load_id"] and second["_load_id"]
    assert first["_load_id"] != second["_load_id"]
    assert {key: value for key, value in first.items() if key != "_load_id"} == {
        key: value for key, value in second.items() if key != "_load_id"}
    current = {"inputs": application.geometry_load_inputs(second)}
    fresh = solve_for_browser(current)
    restored = transition("result_store", fresh, design["state"], second)[0]
    resaved = yaml.safe_load(download(
        dict(design, results=fresh, live=browser_inputs(second)), restored, timestamp=201)["content"])
    assert "_load_id" not in resaved
    assert resaved["inputs"] == second["inputs"]
    assert resaved["geometry"] == second["geometry"]


def test_legacy_yaml_loads_meanline_defaults_and_legacy_download_call_still_works(design, notices):
    legacy = deepcopy(design["config"])
    assert "geometry" not in legacy
    loaded = application.load_design(upload(legacy))
    assert loaded is not no_update and "geometry" not in loaded
    output = transition("result_store", design["results"], design["state"], loaded)
    assert all(not row["parameters"]["overrides"] for row in output[0]["rows"])
    assert output[0]["_loaded_geometry_timestamp"] == 200
    saved = yaml.safe_load(download(design, legacy=True)["content"])
    assert saved["geometry"] == export_geometry(design["results"])


@pytest.mark.parametrize("malformation", [
    "missing_rotor", "extra_stage", "unknown_version", "fractional_blades",
    "invalid_sampling", "unknown_parameter", "wrong_surface", "hidden_relief",
])
def test_invalid_geometry_rejects_entire_upload_atomically(design, notices, malformation):
    document = dict(design["config"], geometry=export_geometry(design["results"], design["state"]))
    geometry = document["geometry"]
    # Corrupt the last row to prove all rows are validated before any load.
    row = geometry["stages"][-1]["rotor"]
    if malformation == "missing_rotor":
        del geometry["stages"][-1]["rotor"]
    elif malformation == "extra_stage":
        geometry["stages"].append(deepcopy(geometry["stages"][0]))
    elif malformation == "unknown_version":
        geometry["schema_version"] = 900
    else:
        path, value = {
            "fractional_blades": ("cascade.blade_count", 31.5),
            "invalid_sampling": ("numerics.number_of_points", 5),
            "unknown_parameter": ("thickness.misspelled_option", .2),
            "wrong_surface": ("geometry.type", "radial" if design["config"]["inputs"]["turbine_type"] == "axial" else "axial"),
            "hidden_relief": ("leading_edge.pressure_side", "minus"),
        }[malformation]
        row["overrides"][path] = value
    original = deepcopy(document)
    assert application.load_design(upload(document)) is no_update
    assert "Could not load YAML" in notices[-1][1]["children"]
    assert "geometry" in notices[-1][1]["children"].lower()
    assert document == original
    with pytest.raises(ValueError):
        validate_geometry(geometry, 2, design["config"]["inputs"]["turbine_type"])


def test_save_folds_unacknowledged_edits_across_rows_before_serializing(design, notices):
    state = design["state"]
    pending = journal(state, [
        dict(kind="set", path="cascade.blade_count", value=43),
        dict(kind="set", row="stage_02/rotor", path="flow_domain.outlet_chords", value=2.7),
        dict(kind="set", row="stage_02/rotor", path="numerics.number_of_points", value=361),
    ])
    saved = yaml.safe_load(download(design, pending=pending)["content"])["geometry"]
    assert saved["stages"][0]["stator"]["overrides"]["cascade.blade_count"] == 43
    assert saved["stages"][1]["rotor"]["overrides"]["flow_domain.outlet_chords"] == 2.7
    assert saved["stages"][1]["rotor"]["overrides"]["numerics.number_of_points"] == 361
    assert state["rows"][0]["parameters"]["overrides"]["cascade.blade_count"] == 31


def test_invalid_pending_edit_blocks_save_and_old_epoch_edit_is_ignored(design, notices):
    pending = journal(design["state"], [dict(kind="set", path="cascade.blade_count", value=2.5)])
    assert download(design, pending=pending) is no_update
    assert "pending invalid geometry edit" in notices[-1][1]["children"]
    pending["epoch"] = "superseded-calculation"
    saved = yaml.safe_load(download(design, pending=pending)["content"])
    assert saved["geometry"] == export_geometry(design["results"], design["state"])


@pytest.mark.parametrize("stale", ["timestamp", "baseline"])
def test_save_rejects_geometry_from_superseded_result(design, notices, stale):
    state = deepcopy(design["state"])
    if stale == "timestamp":
        state["_result_timestamp"] = 99
    else:
        state["baseline_id"] = "old-result"
    assert download(design, state) is no_update
    assert "Design was not saved" in notices[-1][1]["children"]


def test_production_callbacks_declare_geometry_load_and_save_dependencies():
    callbacks = {entry["callback"].__wrapped__.__name__: entry
                 for entry in application.app.callback_map.values() if entry.get("callback")}
    states = callbacks["download_yaml"]["state"]
    assert [(item["id"], item["property"]) for item in states[-3:]] == [
        ("geometry_state", "data"), ("blade_modeller_events", "data"),
        ("result_store", "modified_timestamp"),
    ]
    editor = callbacks["prepare_edit"]
    assert {"id": "loaded_cfg_store", "property": "modified_timestamp"} in editor["inputs"]
    assert {"id": "loaded_cfg_store", "property": "data"} in editor["state"]
    accept = next(entry for key, entry in application.app.callback_map.items()
                  if "geometry_state.data" in key and "blade_modeller_source.children" in key)
    assert {"id": "loaded_cfg_store", "property": "modified_timestamp"} in accept["state"]
