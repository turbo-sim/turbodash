"""Opt-in mesh editor presentation preserves row settings and shared defaults."""

from copy import deepcopy
from types import SimpleNamespace

import dash_bootstrap_components as dbc
from dash import Dash, dcc, html
import pytest

from turbodash.apps.shared.components import meshing as ui
from turbodash.visualization import plotly_mesh


def components(component):
    if isinstance(component, (list, tuple)):
        for item in component:
            yield from components(item)
        return
    yield component
    children = getattr(component, "children", None)
    if children is not None:
        yield from components(children)


def callbacks(*, precise_inputs):
    app = Dash(__name__, suppress_callback_exceptions=True)
    ui.register_meshing_callbacks(app, precise_inputs=precise_inputs)
    return {
        entry["callback"].__wrapped__.__name__: entry["callback"].__wrapped__
        for entry in app.callback_map.values()
        if "callback" in entry
    }


def test_precise_row_fields_keep_full_values_and_discrete_counts():
    stages = [{"name": "High-pressure"}, {"name": "Low-pressure"}]
    ids = [
        {"scope": "mesh-row", "stage": 0, "row": row, "key": "span_stretch"}
        for row in ("stator", "rotor")
    ]
    values = [1.23456789123, 2.98765432198]
    panels = ui.mesh_row_panels(stages, values, ids, precise_inputs=True)
    assert [panel.children[0].children for panel in panels] == [
        "High-pressure",
        "Low-pressure",
    ]
    fields = [
        component
        for component in components(panels)
        if isinstance(component, dbc.Input)
    ]
    assert len(fields) == 4 * len(ui._SETTINGS)
    for field in fields:
        assert field.step == (
            1 if field.id["key"] in {"span_layers", "profile_points"} else "any"
        )
        assert field.debounce is True
    assert [
        field.value
        for field in fields
        if field.id["stage"] == 0 and field.id["key"] == "span_stretch"
    ] == values
    assert all(
        component.open is False
        for component in components(panels)
        if isinstance(component, html.Details)
    )
    assert stages == [{"name": "High-pressure"}, {"name": "Low-pressure"}]
    # Other app editions keep their existing increments and initial expansion.
    legacy = ui.mesh_row_panels(stages)
    assert legacy[0].open is True
    assert legacy[0].children[0].children == "Stage 1"
    assert (
        next(
            field
            for field in components(legacy)
            if isinstance(field, dbc.Input) and field.id["key"] == "span_stretch"
        ).step
        == 0.1
    )


def test_opt_in_callbacks_display_names_without_changing_selected_row(monkeypatch):
    monkeypatch.setattr(ui, "ctx", SimpleNamespace(triggered_id=None))
    results = {
        "inputs": {"stages": [{"name": "Input first"}, {"name": "Input second"}]},
        "stages_performance": [{"name": "Solved first"}, {}],
    }
    updated = callbacks(precise_inputs=True)["update_stages"](
        results, None, 1, 0, "rotor"
    )
    expected = [
        {"label": "Solved first", "value": 0},
        {"label": "Input second", "value": 1},
    ]
    assert updated == (expected, 1, expected, 0, "rotor")
    legacy = callbacks(precise_inputs=False)["update_stages"](
        results, None, 1, 0, "rotor"
    )
    assert [item["label"] for item in legacy[0]] == ["Stage 1", "Stage 2"]


def test_named_stage_fallback_supports_yaml_mapping_and_missing_names():
    assert ui._named_stage_options(
        {"inputs": {"stages": {"HP": {}, "LP": {}}}, "stages_performance": [{}, {}, {}]}
    ) == [
        {"label": "HP", "value": 0},
        {"label": "LP", "value": 1},
        {"label": "Stage 3", "value": 2},
    ]


def test_workbench_panels_keep_callback_ids_but_remove_slice_slider():
    legacy = list(components([ui.geometry_panel(), ui.mesh_panel()]))
    refined = list(
        components([ui.geometry_panel(workbench=True), ui.mesh_panel(workbench=True)])
    )

    def indexed(nodes):
        return {
            node.id: node
            for node in nodes
            if isinstance(getattr(node, "id", None), str)
        }

    old, new = indexed(legacy), indexed(refined)
    assert set(old) <= set(new)
    assert isinstance(old["mesh_fraction"], dcc.Slider)
    assert isinstance(new["mesh_fraction"], dbc.Input)
    assert new["mesh_fraction"].step == "any"
    assert new["mesh_fraction"].value == old["mesh_fraction"].value == 50
    assert not any(isinstance(component, dcc.Slider) for component in refined)
    tooltip_text = " ".join(
        str(node.children) for node in refined if isinstance(node, dbc.Tooltip)
    )
    assert "cylindrical" in tooltip_text and "flared" in tooltip_text
    app = Dash(__name__, suppress_callback_exceptions=True)
    app.layout = html.Div(
        [ui.geometry_panel(workbench=True), ui.mesh_panel(workbench=True)]
    )
    ui.register_meshing_callbacks(app, precise_inputs=True)
    assert app.server.test_client().get("/_dash-layout").status_code == 200


@pytest.mark.parametrize("mesh", [False, True])
def test_meridional_zero_origin_is_opt_in_and_preserves_physical_geometry(mesh):
    preview = {
        "meridional_figure": {
            "data": [{"x": [0, 10], "y": [80, 100], "type": "scatter"}]
        },
        "metadata": {"hub_radius_mm": 80, "shroud_radius_mm": 100},
    }
    before = deepcopy(preview)
    default = plotly_mesh.plot_geometry_meridional(preview)
    figure = plotly_mesh._meridional_style(
        plotly_mesh.go.Figure(preview["meridional_figure"]),
        preview["metadata"],
        mesh=mesh,
        start_at_zero=True,
        show_note=False,
    )
    assert default.layout.yaxis.range[0] > 0
    assert figure.layout.yaxis.range[0] == 0
    assert figure.layout.yaxis.rangemode == "tozero"
    assert figure.layout.yaxis.fixedrange is False
    assert figure.layout.yaxis.scaleanchor == "x"
    assert not figure.layout.annotations
    assert list(figure.data[0].y) == [80, 100]
    assert preview == before
