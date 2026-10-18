"""Workflow checks: explicit submission, stale-result guards and optional deps."""

from types import SimpleNamespace

import pytest
from dash import Dash, dcc, html, no_update
from dash.exceptions import PreventUpdate

from turbodash.apps.components import meshing as ui


@pytest.fixture
def jobs():
    submitted = []

    def submit(*args):
        submitted.append(args)
        return {"id": "a" * 32, "fingerprint": "current"}

    return SimpleNamespace(
        submitted=submitted, start_job=submit,
        fingerprint=lambda *args: "current",
        job_status=lambda handle: {"state": "running", "message": "Extruding", "fraction": 0.5},
    )


def test_input_signature_tracks_values_and_stage_count_without_order_dependency():
    a = {"scope": "overall", "key": "fluid"}
    b = {"scope": "overall", "key": "pressure"}
    signature = ui.input_signature(["Air", 1e5], [], [a, b], [], [{}])
    assert signature == ui.input_signature([1e5, "Air"], [], [b, a], [], [{}])
    assert signature != ui.input_signature(["Air", 2e5], [], [a, b], [], [{}])
    assert signature != ui.input_signature(["Air", 1e5], [], [a, b], [], [{}, {}])


@pytest.mark.parametrize("trigger", [None, "result_store", "mesh_stage", "mesh_span_layers", {"scope": "overall", "key": "fluid"}])
def test_editing_controls_does_not_submit_mesh(trigger, jobs):
    handle, status, disabled = ui.advance_mesh_job(trigger, {}, 0, "stator", {}, None, jobs=jobs)
    assert jobs.submitted == []
    assert handle is None
    assert status["state"] == "idle"
    assert disabled


def test_only_generate_submits_and_polling_retains_handle(jobs):
    handle, status, disabled = ui.advance_mesh_job("mesh_generate", {}, 0, "rotor", {"span_layers": 4}, None, jobs=jobs)
    assert len(jobs.submitted) == 1
    assert status["state"] == "running"
    assert not disabled
    polled, status, disabled = ui.advance_mesh_job("mesh_poll", {}, 0, "rotor", {"span_layers": 4}, handle, jobs=jobs)
    assert polled == handle
    assert status["fraction"] == 0.5
    assert not disabled
    assert len(jobs.submitted) == 1


def test_stale_or_unavailable_mesh_cannot_be_displayed(jobs):
    old = {"id": "a" * 32, "fingerprint": "previous"}
    handle, status, disabled = ui.advance_mesh_job("mesh_poll", {}, 0, "stator", {}, old, jobs=jobs)
    assert handle is None and disabled
    handle, status, disabled = ui.advance_mesh_job("mesh_generate", {}, 0, "stator", {}, None, reason="Pending calculation", jobs=jobs)
    assert handle is None and status["message"] == "Pending calculation"
    assert not jobs.submitted


def test_complete_status_excludes_server_artifact_paths(jobs):
    jobs.job_status = lambda handle: {
        "state": "complete", "message": "Finished", "fraction": 1,
        "manifest": {"summary": {"cells": 123}, "paths": {"mesh": "C:/private/mesh.msh"}},
    }
    _, status, disabled = ui.advance_mesh_job("mesh_poll", {}, 0, "stator", {}, {"fingerprint": "current"}, jobs=jobs)
    assert disabled
    assert status == {"state": "complete", "message": "Finished", "fraction": 1, "summary": {"cells": 123}}


def test_complete_status_keeps_only_mesh_display_metadata(jobs):
    jobs.job_status = lambda handle: {
        "state": "complete", "manifest": {"metadata": {"stage_number": 2, "row": "rotor",
            "blade_count": 35, "coordinate_units": "mm", "source_row": {"private": "details"}}},
    }
    _, status, _ = ui.advance_mesh_job("mesh_poll", {}, 0, "stator", {}, {"fingerprint": "current"}, jobs=jobs)
    assert status["metadata"] == {"stage_number": 2, "row": "rotor", "blade_count": 35, "coordinate_units": "mm"}


def test_worker_failure_is_visible(jobs):
    def fail(*args):
        raise ValueError("Inflation exceeds the passage opening")

    jobs.start_job = fail
    handle, status, disabled = ui.advance_mesh_job("mesh_generate", {}, 0, "stator", {}, None, jobs=jobs)
    assert handle is None and disabled
    assert status["state"] == "failed"
    assert "Inflation" in status["message"]


def test_missing_optional_dependencies_leave_meanline_usable(monkeypatch):
    monkeypatch.setattr(ui.meshing, "dependency_status", lambda: (False, "Install the meshing extra"))
    results = {"inputs": {"turbine_type": "axial"}, "stages_performance": [{}]}
    assert ui._unavailable(results, "same", "same") == "Install the meshing extra"
    assert "inputs have changed" in ui._unavailable(results, "previous", "current")
    app = Dash(__name__, suppress_callback_exceptions=True)
    app.layout = html.Div([ui.geometry_panel(), ui.mesh_panel()])
    ui.register_meshing_callbacks(app)
    client = app.server.test_client()
    assert client.get("/_dash-layout").status_code == 200
    assert client.get("/_dash-dependencies").status_code == 200


def test_optional_sizes_are_omitted_and_inflation_boolean_is_explicit():
    values = [item[2] for item in ui._SETTINGS] + [["enabled"]]
    options = ui._options(values)
    assert "blade_size_mm" not in options
    assert "first_height_mm" not in options
    assert options["inflation_enabled"] is True
    assert options["span_layers"] == 12


@pytest.fixture
def callbacks():
    app = Dash(__name__, suppress_callback_exceptions=True)
    ui.register_meshing_callbacks(app)
    return {entry["callback"].__wrapped__.__name__: entry["callback"].__wrapped__
            for entry in app.callback_map.values() if "callback" in entry}


def test_stale_design_hides_geometry_and_blocks_download(callbacks, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("A stale result must not build or export geometry")

    monkeypatch.setattr(ui.meshing, "preview_blade_row", forbidden)
    monkeypatch.setattr(ui.meshing, "geometry_zip", forbidden)
    results = {"inputs": {"turbine_type": "axial"}, "stages_performance": [{}]}
    live = ([], [], [], [], [{}])
    _, _, message, disabled, _ = callbacks["update_geometry"](results, "previous", 0, "stator", {"display": "block"}, *live)
    assert disabled and "inputs have changed" in message
    _, error = callbacks["download_geometry"](1, results, "previous", 0, "stator", *live)
    assert "inputs have changed" in error


def test_mesh_download_rechecks_settings_on_server(callbacks, monkeypatch):
    monkeypatch.setattr(ui.meshing, "dependency_status", lambda: (True, ""))
    monkeypatch.setattr(ui, "vtk_assets_available", lambda: True)
    monkeypatch.setattr(ui.mesh_jobs, "fingerprint", lambda *args: "current")
    monkeypatch.setattr(ui.mesh_jobs, "mesh_archive", lambda handle: pytest.fail("Stale mesh must not download"))
    results = {"inputs": {"turbine_type": "axial"}, "stages_performance": [{}]}
    live = ([], [], [], [], [{}])
    values = [item[2] for item in ui._SETTINGS] + [[]]
    _, error = callbacks["download_mesh"](1, {"fingerprint": "old"}, results, ui.input_signature(*live), 0, "stator", *values, *live)
    assert "does not match" in error


def test_irrelevant_inspection_fields_do_not_break_surface_view(callbacks, monkeypatch):
    monkeypatch.setattr(ui.mesh_jobs, "job_directory", lambda handle: "trusted")
    calls = []

    def inspect(directory, **options):
        calls.append(options)
        return {"surfaces": {}}

    monkeypatch.setattr(ui.meshing, "inspect_mesh", inspect)
    _, error = callbacks["inspect_mesh"]({"id": "handle"}, {"state": "complete"}, "surface", "skewness", None, None, None, None, None, {"display": "block"})
    assert calls == [{"mode": "surface", "metric": "skewness"}]
    assert not isinstance(error, str)


def test_mesh_summary_explains_counts_and_quality():
    text = ui._summary_text({"volume_nodes": 120, "cells": {"hex": 50, "wedge": 10},
        "quality": {"min_determinant_jacobian": 0.1}}, {"skewness": {"min": 0.1, "max": 0.8, "mean": 0.3}})
    assert "Nodes: 120" in text
    assert "Volume cells: 60" in text
    assert "Minimum Jacobian: 0.1" in text
    assert "Skewness: 0.1 to 0.8" in text


def test_three_tab_layout_keeps_controls_and_stores_mounted():
    from turbodash.apps.turbine_experimental import app, switch_tab

    def components(component):
        yield component
        children = getattr(component, "children", None)
        for child in children if isinstance(children, (list, tuple)) else [children]:
            if hasattr(child, "to_plotly_json"):
                yield from components(child)

    all_components = list(components(app.layout))
    tabs = [component for component in all_components if isinstance(component, dcc.Tab)]
    assert [tab.label for tab in tabs] == ["Meanline", "Turbine geometry", "Mesh"]
    assert all(not getattr(tab, "children", None) for tab in tabs)
    ids = [component.id for component in all_components if isinstance(getattr(component, "id", None), str)]
    assert len(ids) == len(set(ids))
    assert {"result_store", "result_input_signature", "stage_data_store", "mesh_job", "mesh_figure", "geometry_overview", "turbogrid_export_button"} <= set(ids)
    assert {"geometry-scene", "overview-scene", "mesh-scene", "geometry-vtk-data", "overview-vtk-data", "mesh-vtk-data"} <= set(ids)
    graphs = {component.id for component in all_components if isinstance(component, dcc.Graph)}
    assert {"geometry_profile", "geometry_meridional", "meridional_plot"} <= graphs
    assert not {"geometry_model", "geometry_overview", "mesh_figure"} & graphs
    assert [style["display"] for style in switch_tab("mesh")] == ["none", "none", "block"]


def test_geometry_cache_reuses_numerical_preview_and_hidden_tabs_preserve_scene(callbacks, monkeypatch):
    ui._cached_preview.cache_clear()
    calls = []
    monkeypatch.setattr(ui, "_unavailable", lambda *args: "")
    monkeypatch.setattr(ui, "plot_profile_preview", lambda payload: "2D profile")
    monkeypatch.setattr(ui, "plot_geometry_meridional", lambda payload: "2D meridional")
    monkeypatch.setattr(ui.vtk, "geometry_data", lambda payload: {"vtk": True})
    monkeypatch.setattr(ui.meshing, "preview_blade_row", lambda *args, **kwargs: calls.append(kwargs) or {"metadata": {}})
    results = {"stages_performance": [{}]}
    live = ([], [], [], [], [{}])
    try:
        with pytest.raises(PreventUpdate):
            callbacks["update_geometry"](results, "signature", 0, "stator", {"display": "none"}, *live)
        assert not calls
        for _ in range(2):
            _, data, _, disabled, meridional = callbacks["update_geometry"](results, "signature", 0, "stator", {"display": "block"}, *live)
            assert data == {"vtk": True}
            assert meridional == "2D meridional"
            assert not disabled
        assert len(calls) == 1
    finally:
        ui._cached_preview.cache_clear()


def test_hidden_mesh_preserves_view_without_reloading_artifacts(callbacks, monkeypatch):
    monkeypatch.setattr(ui.meshing, "inspect_mesh", lambda *args, **kwargs: pytest.fail("Hidden view reloaded mesh"))
    with pytest.raises(PreventUpdate):
        callbacks["inspect_mesh"]({"id": "handle"}, {"state": "complete"}, "surface", "skewness", None, None, None, None, None, {"display": "none"})


def test_missing_vtk_assets_are_actionable(monkeypatch):
    monkeypatch.setattr(ui.meshing, "dependency_status", lambda: (True, ""))
    monkeypatch.setattr(ui, "vtk_assets_available", lambda: False)
    results = {"inputs": {"turbine_type": "axial"}, "stages_performance": [{}]}
    assert "VTK renderer" in ui._unavailable(results, "same", "same")


def _components(component):
    if isinstance(component, (list, tuple)):
        for item in component:
            yield from _components(item)
        return
    yield component
    children = getattr(component, "children", None)
    for child in children if isinstance(children, (list, tuple)) else [children]:
        if hasattr(child, "to_plotly_json"):
            yield from _components(child)


@pytest.mark.parametrize("turbine_type", ["axial", "radial"])
def test_geometry_editors_match_every_mounted_meanline_parameter(turbine_type):
    from turbodash.apps.turbine_experimental import geometry_stage_panel, make_stage_accordion_item, stage_defaults_for

    values = stage_defaults_for(turbine_type)
    meanline = make_stage_accordion_item(1, values, turbine_type)
    geometry = geometry_stage_panel(1, values, turbine_type)

    def parameters(component, element):
        return {item.id["key"]: item.value for item in _components(component)
                if isinstance(getattr(item, "id", None), dict) and item.id.get("elem") == element}

    assert parameters(meanline, "input") == parameters(geometry, "geometry")
    assert ("stator.aspect_ratio" in parameters(geometry, "geometry")) == (turbine_type == "axial")


@pytest.mark.parametrize("element,value", [("geometry", 0.45), ("slider", 0.25), ("input", 0.35)])
def test_geometry_meanline_and_slider_use_one_value(monkeypatch, element, value):
    from turbodash.apps import turbine_experimental as app_experimental

    monkeypatch.setattr(app_experimental, "ctx", SimpleNamespace(triggered_id={"elem": element}))
    assert app_experimental.sync_stage_slider_input(.25, .35, .45) == (value, value, value)
    monkeypatch.setattr(app_experimental, "ctx", SimpleNamespace(triggered_id={"elem": "geometry"}))
    assert app_experimental.sync_stage_slider_input(.25, .35, None) == (no_update, None, None)


def test_row_mesh_settings_persist_independently_and_ignore_inactive_edits(callbacks):
    identifiers = [{"scope": "mesh-row", "stage": 0, "row": row, "key": "span_layers"} for row in ("stator", "rotor")]
    defaults = [item[2] for item in ui._SETTINGS] + [[]]
    select = callbacks["select_mesh_settings"]
    assert select(0, "stator", [3, 7], identifiers, *defaults)[ui._OPTION_KEYS.index("span_layers")] == 3
    assert select(0, "rotor", [3, 7], identifiers, *defaults)[ui._OPTION_KEYS.index("span_layers")] == 7
    current = list(defaults)
    current[ui._OPTION_KEYS.index("span_layers")] = 3
    assert all(value is no_update for value in select(0, "stator", [3, 9], identifiers, *current))
    panels = ui.mesh_row_panels([{}, {}], [3, 7], identifiers)
    fields = {(item.id["stage"], item.id["row"]): item.value for item in _components(panels)
              if isinstance(getattr(item, "id", None), dict) and item.id.get("key") == "span_layers"}
    assert fields == {(0, "stator"): 3, (0, "rotor"): 7, (1, "stator"): 12, (1, "rotor"): 12}


def test_workspace_pairs_2d_views_above_single_visible_3d_view():
    geometry = list(_components(ui.geometry_panel()))
    mesh = list(_components(ui.mesh_panel()))
    geometry_ids = {item.id: item for item in geometry if isinstance(getattr(item, "id", None), str)}
    mesh_ids = {item.id: item for item in mesh if isinstance(getattr(item, "id", None), str)}
    assert geometry_ids["geometry_model"].style == {"display": "none"}
    assert {"geometry_profile", "geometry_meridional", "geometry_overview"} <= set(geometry_ids)
    assert {"mesh_profile", "mesh_meridional", "mesh_figure"} <= set(mesh_ids)
    assert geometry_ids["geometry_profile"].responsive is True
    assert mesh_ids["mesh_profile"].responsive is True
