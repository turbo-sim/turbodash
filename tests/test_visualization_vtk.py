"""VTK adaptation preserves physical geometry, original cells and stable hosts."""

from copy import deepcopy
import json

from dash import Dash, html
import numpy as np
import pytest

from turbodash.visualization import vtk


@pytest.fixture
def surface():
    return {"points": [10., 0., 0., 11., 0., 0., 11., 1., 2., 10., 1., 2.],
            "polys": [4, 0, 1, 2, 3], "normals": [0., 0., 1.] * 4}


@pytest.fixture
def preview(surface):
    return {"metadata": {"row": "stator", "stage_number": 1, "blade_count": 8,
                         "profile_chord_mm": 2., "coordinate_units": "mm"},
            "surfaces": {"blade": surface, "hub": deepcopy(surface)}}


def descendants(component):
    yield component
    children = getattr(component, "children", [])
    if not isinstance(children, (list, tuple)):
        children = [children]
    for child in children:
        if hasattr(child, "to_plotly_json"):
            yield from descendants(child)


def text_content(component):
    if isinstance(component, (str, int, float)):
        return str(component)
    children = getattr(component, "children", [])
    if not isinstance(children, (list, tuple)):
        children = [children]
    return " ".join(text_content(child) for child in children if child is not None)


def test_geometry_preserves_raw_vtk_quads_normals_and_mm(preview):
    before = deepcopy(preview)
    data = vtk.geometry_data(preview)
    row = data["rows"][0]
    assert row["num_blades"] == 8
    assert row["geometry_type"] == "axial"
    assert row["surfaces"] is preview["surfaces"]
    assert row["surfaces"]["blade"]["polys"] == [4, 0, 1, 2, 3]
    assert row["surfaces"]["blade"]["points"][0] == 10.
    assert row["surfaces"]["blade"]["normals"] == [0., 0., 1.] * 4
    assert preview == before
    assert json.loads(json.dumps(data))["key"] == data["key"]


def test_turbine_placement_is_display_only_and_rows_remain_independent(preview):
    rotor = deepcopy(preview)
    rotor["metadata"]["row"] = "rotor"
    before = deepcopy([preview, rotor])
    data = vtk.turbine_data([preview, rotor])
    assert [row["id"] for row in data["rows"]] == ["stage_1_stator", "stage_1_rotor"]
    assert [row["placement"]["streamwise_offset_mm"] for row in data["rows"]] == [0., 2.5]
    assert data["rows"][1]["surfaces"]["blade"]["points"][2] == 0.
    assert "illustrative" in data["note"]
    assert [preview, rotor] == before


def test_inspection_preserves_volume_cell_id_and_quality_per_original_face(surface):
    inspection = {**surface, "cell_ids": [42], "cell_values": [100.], "color_values": [0.5]}
    payload = {"surfaces": {"inspection": inspection}, "mode": "worst", "metric": "aspect_ratio",
               "metric_label": "Edge aspect ratio", "color_range": [1., 10000.],
               "logarithmic": True, "note": "Original volume quality."}
    boundary = {"surfaces": {"blade": surface}}
    data = vtk.mesh_data(payload, geometry=boundary)
    row = data["rows"][0]
    selected = data["inspection"]
    assert selected["row_id"] == row["id"]
    assert selected["job"] == row["job"]
    assert selected["surfaces"]["inspection"] is inspection
    assert inspection["polys"] == [4, 0, 1, 2, 3]
    assert inspection["cell_ids"] == [42]
    assert inspection["cell_values"] == [100.]
    assert inspection["color_values"] == [0.5]
    assert data["quality"]["color_range"] == [1., 10000.]


def test_inspection_change_preserves_artifact_identity_and_camera(surface):
    context = {"surfaces": {"blade": surface}}
    first = {"mode": "worst", "surfaces": {"inspection": {**surface,
             "cell_ids": [5], "cell_values": [.8], "color_values": [.8]}}}
    second = deepcopy(first)
    second["surfaces"]["inspection"]["cell_ids"] = [9]
    old, new = [vtk.mesh_data(payload, context) for payload in (first, second)]
    assert old["key"] == new["key"]
    assert old["inspection"]["key"] != new["inspection"]["key"]


def test_surface_mesh_passes_original_polygons_to_upstream_edge_shader(surface):
    data = vtk.mesh_data({"mode": "surface", "surfaces": {"hub": surface}})
    assert data["inspection"] is None
    assert data["rows"][0]["surfaces"]["hub"]["polys"] == [4, 0, 1, 2, 3]


@pytest.mark.parametrize("metric,low,high,labels", [
    ("skewness", 0., 1., ["0", "0.25", "0.5", "0.75", "1"]),
    ("orthogonality", 0., 1., ["1", "0.75", "0.5", "0.25", "0"]),
    ("aspect_ratio", 1., 10000., ["1", "10", "100", "1e+03", "1e+04"]),
])
def test_legend_matches_upstream_green_good_red_bad_color_order(metric, low, high, labels):
    legend = vtk.quality_legend({"metric": metric, "color_range": [low, high],
                               "logarithmic": metric == "aspect_ratio"})
    assert [span.children for span in legend.children[2].children] == labels
    assert all(color in legend.children[1].style["background"] for color in vtk.QUALITY_COLORS)
    assert "Better" in text_content(legend) and "Poorer" in text_content(legend)


def test_panel_exposes_stable_host_and_store_without_python_vtk_or_plotly_tree(monkeypatch):
    monkeypatch.setattr(vtk, "vtk_available", lambda: True)
    panel = vtk.vtk_panel("mesh")
    components = list(descendants(panel))
    ids = [getattr(component, "id", None) for component in components]
    assert "mesh-scene" in ids
    assert "mesh-vtk-data" in ids
    assert "mesh-vtk-fit" in ids
    assert not any(component.to_plotly_json()["type"] in ("Graph", "View") for component in components)
    assert len([id for id in ids if id]) == len(set(id for id in ids if id))


def test_missing_optional_dependency_keeps_stable_panel_and_explains_install(monkeypatch):
    monkeypatch.setattr(vtk, "vtk_available", lambda: False)
    assert "Install the Turbodash mesh extra" in text_content(vtk.vtk_panel("geometry"))


def test_callbacks_update_render_token_and_note_without_replacing_host():
    app = Dash(__name__)
    app.layout = html.Div([vtk.vtk_panel(pane) for pane in vtk.PANES])
    vtk.register_vtk_callbacks(app)
    assert len(app.callback_map) == 12
    assert all("-scene.children" not in key for key in app.callback_map)
    assert "studio.render" in vtk._RENDER
    assert "studio.commit" in vtk._RENDER


def test_sidebar_has_rows_surfaces_counts_and_stable_viewport(monkeypatch):
    monkeypatch.setattr(vtk, "vtk_available", lambda: True)
    components = list(descendants(vtk.vtk_panel("overview")))
    ids = {getattr(component, "id", None) for component in components}
    assert {"overview-scene", "overview-vtk-rows", "overview-vtk-copy-controls",
            "overview-vtk-display-state", "overview-vtk-surfaces", "overview-vtk-fit"} <= ids
    classes = {getattr(component, "className", None) for component in components}
    assert {"td-vtk-viewport", "td-vtk-frame", "td-vtk-display", "td-vtk-host", "td-display-section"} <= classes
    assert "Configure display" in text_content(vtk.vtk_panel("overview"))


def test_mesh_inspection_controls_can_live_in_display_sidebar():
    panel = vtk.vtk_panel("mesh", extra_controls=html.Div("Inspection", id="inspection-controls"))
    sidebar = next(component for component in descendants(panel)
                   if getattr(component, "className", None) == "td-vtk-display")
    assert sidebar.children[-1].id == "inspection-controls"


def test_visibility_and_per_row_counts_survive_geometry_recalculation(preview):
    rotor = deepcopy(preview)
    rotor["metadata"]["row"] = "rotor"
    data = vtk.turbine_data([preview, rotor])
    preferences = vtk.display_preferences(["stage_1_stator"], [7, 2], [
        {"row": "stage_1_stator"}, {"row": "stage_1_rotor"},
    ])
    changed = deepcopy(data)
    changed["key"] = "new-calculation"
    _, visible, controls = vtk.display_rows(changed, preferences)
    assert visible == ["stage_1_stator"]
    assert [control["value"] for control in controls] == [7, 2]
    assert all(control["max"] == 8 for control in controls)


def test_new_rows_start_visible_without_unhiding_existing_hidden_rows(preview):
    data = vtk.turbine_data([preview])
    preferences = vtk.display_preferences([], [3], [{"row": "stage_1_stator"}])
    rotor = deepcopy(preview)
    rotor["metadata"]["row"] = "rotor"
    changed = vtk.turbine_data([preview, rotor])
    _, visible, controls = vtk.display_rows(changed, preferences)
    assert visible == ["stage_1_rotor"]
    assert [control["value"] for control in controls] == [3, 3]
    assert vtk.display_rows(data, preferences)[1] == []


def test_temporarily_absent_rows_keep_visibility_and_count_preferences(preview):
    preferences = vtk.display_preferences(["stage_1_stator"], [7], [{"row": "stage_1_stator"}])
    preferences = vtk.display_preferences([], [2], [{"row": "stage_1_rotor"}], preferences)
    assert preferences["visible_rows"] == ["stage_1_stator"]
    assert preferences["copies"] == {"stage_1_stator": 7, "stage_1_rotor": 2}
    assert vtk.display_rows(vtk.turbine_data([preview]), preferences)[1] == ["stage_1_stator"]


def test_copy_controls_use_physical_blade_count_and_triangle_budget(preview, monkeypatch):
    monkeypatch.setattr(vtk, "MAX_DISPLAY_TRIANGLES", 32)
    data = vtk.turbine_data([preview])  # Two original quads: 4 triangles/copy.
    assert data["copy_limit"] == 8
    _, _, controls = vtk.display_rows(data, {"copies": {"stage_1_stator": 100}})
    assert controls[0]["max"] == controls[0]["value"] == 8
    data["copy_limit"] = 2
    assert vtk.display_rows(data)[2][0]["value"] == 2


def test_stale_dataset_does_not_reset_controls_or_preferences():
    app = Dash(__name__)
    vtk.register_vtk_callbacks(app)
    callback = next(item["callback"].__wrapped__ for name, item in app.callback_map.items()
                    if "overview-vtk-copy-controls.children" in name)
    preferences = {"known_rows": ["stage_1_stator"], "visible_rows": [],
                   "copies": {"stage_1_stator": 4}}
    from dash import no_update
    assert callback(vtk.empty_data("Recalculating"), preferences) == (no_update,) * 3


def test_shared_renderer_receives_visibility_without_changing_dataset_key():
    assert "studio.render(data, settings, pane, shown" in vtk._RENDER
    assert "shown.includes(inspection.row_id)" in vtk._RENDER
    assert "data.key =" not in vtk._RENDER


def test_render_limit_does_not_sample_cells(preview, monkeypatch):
    monkeypatch.setattr(vtk, "MAX_DISPLAY_TRIANGLES", 1)
    data = vtk.mesh_data({"mode": "surface", "surfaces": preview["surfaces"]})
    assert data["rows"] == []
    assert "No cells were sampled" in data["note"]


@pytest.mark.parametrize("polys", [[4, 0, 1], [3, 0, 1, 99], [3., 0., 1., 2.]])
def test_invalid_polygon_connectivity_is_rejected(preview, polys):
    preview["surfaces"]["blade"]["polys"] = polys
    with pytest.raises(ValueError, match="polygon|indices|node"):
        vtk.geometry_data(preview)


def test_quality_arrays_must_match_original_face_count(surface):
    with pytest.raises(ValueError, match="one finite value per original polygon"):
        vtk.mesh_data({"surfaces": {"inspection": {**surface,
                      "cell_ids": [5, 6], "cell_values": [.8], "color_values": [.8]}}})
