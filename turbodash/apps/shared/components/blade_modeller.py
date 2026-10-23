"""Shared blade editor: one baseline with sparse, per-row geometry overrides.

Only successful ``result_store`` changes replace the baseline. Widgets project
that state; neither viewing a row nor editing it calls the meanline solver.
Live slider events update the lightweight plots; the committed snapshot drives
3D, CAD and mesh consumers. Browser-side token checks reject stale plot work.
"""
from __future__ import annotations

import base64
import binascii
from copy import deepcopy
import json
import math
import re
import uuid

import yaml
import dash_bootstrap_components as dbc
from dash import ALL, MATCH, ClientsideFunction, Input, Output, State, ctx, dcc, html, no_update
from dash.exceptions import PreventUpdate

from turbodash.geometry.blade_parametrization.parameters import (
    FIELD_SPECS, resolve_parameters, state_from_dict,
)
from turbodash.geometry.blade_parametrization.geometry import build_section
from turbodash.geometry.blade_parametrization.plots import (
    build_figures, passage_traces,
)
from turbodash.visualization import vtk
from turbodash.visualization.blade_preview import section_preview
from turbodash.geometry.meanline import (
    get_row_section, initialize_geometry, reset_geometry_overrides,
    row_parameter_state, update_geometry_override,
)
from turbodash.geometry.serialization import restore_geometry
from .meshing import _engineering_plot, geometry_panel as _mesh_geometry_panel
from .turbogrid import turbogrid_export_panel
from .blade_help import PARAMETER_HELP
from .meshing import _hint


PATHS = tuple(FIELD_SPECS)
HIDDEN_PATHS = {"leading_edge.pressure_side", "trailing_edge.relief_strength", "trailing_edge.pressure_side"}
SLIDER_RANGES = {
    "camberline.inlet_metal_angle_deg": (-85., 85., .25),
    "camberline.outlet_metal_angle_deg": (-85., 85., .25),
    "camberline.turning_location": (.01, .99, .005),
    "camberline.turning_concentration": (1., 5., .05),
    "thickness.maximum_thickness_ratio": (.001, .65, .0025),
    "thickness.maximum_thickness_location": (.01, .99, .005),
    "thickness.leading_edge_thickness_ratio": (0., .3, .0025),
    "thickness.trailing_edge_thickness_ratio": (0., .3, .0025),
    "thickness.leading_edge_fraction": (0., 1., .005),
    "thickness.trailing_edge_fraction": (0., 1., .005),
    "thickness.leading_edge_fullness": (-1., 1., .01),
    "thickness.trailing_edge_fullness": (-1., 1., .01),
    "thickness.shape_exponent": (1., 8., .05),
    "thickness.leading_edge_rounding_exponent": (2., 8., .05),
    "thickness.trailing_edge_rounding_exponent": (2., 8., .05),
    "leading_edge.relief_strength": (0., .95, .005),
}
SLIDER_PATHS = tuple(SLIDER_RANGES)


def _lookup(values, path):
    group, name = path.split(".", 1)
    return values[group][name]


def to_control(path, value):
    """Display dimensions in mm, angles in degrees and ratios without units."""
    spec = FIELD_SPECS[path]
    if spec["type"] == "bool":
        return ["enabled"] if value else []
    if spec["units"] == "m" and value is not None:
        return value * 1000
    return value


def from_control(path, value):
    spec = FIELD_SPECS[path]
    if spec["type"] == "bool":
        return "enabled" in (value or [])
    if spec["type"] == "enum" or value is None:
        return value
    if spec["type"] == "int":
        return int(value) if isinstance(value, (int, float)) and float(value).is_integer() else value
    return value / 1000 if spec["units"] == "m" else value


def slider_settings(path, value):
    """Slider steps aid exploration; they never quantize a typed value."""
    low, high, step = SLIDER_RANGES[path]
    if isinstance(value, (int, float)) and math.isfinite(value):
        low, high = min(low, value), max(high, value)
    return {"min": low, "max": high, "step": step}


def selected_row(state, stage, row):
    return next((item for item in (state or {}).get("rows", [])
                 if item["stage_index"] == (stage or 0) and item["row"] == (row or "stator")), None)


def state_token(state, row_id):
    """View acceptance includes the meanline epoch, row and exact edit revision."""
    if not state or not row_id:
        return None
    return [state.get("baseline_id"), state.get("_epoch"), state.get("revision", 0), row_id]


def input_view(state, row_id):
    resolved = resolve_parameters(row_parameter_state(state, row_id))
    kind = resolved.values["geometry"]["type"]
    values, labels, resets, styles = [], [], [], []
    for path in PATHS:
        values.append(to_control(path, _lookup(resolved.values, path)))
        origin = resolved.provenance[path]
        labels.append({"override": "Edited", "inherited": "Inherited", "default": "Default"}[origin])
        resets.append(origin != "override")
        hidden = path in HIDDEN_PATHS or (kind == "axial" and path in {"geometry.inlet_radius", "geometry.outlet_radius"}) or (
            kind == "radial" and path in {"geometry.mean_radius", "geometry.meridional_chord"})
        styles.append({"display": "none"} if hidden else {})
    return values, labels, resets, styles


def import_row_yaml(contents, state, row_id):
    """Apply a YAML row atomically as overrides of its current meanline source."""
    try:
        payload = contents.split(",", 1)[1]
        if len(payload) > 1_400_000:
            raise ValueError("Choose a YAML file smaller than 1 MB.")
        parsed = yaml.safe_load(base64.b64decode(payload, validate=True).decode("utf-8-sig"))
        imported = resolve_parameters(state_from_dict(parsed)).values
        if imported["leading_edge"]["pressure_side"] != "auto" or imported["trailing_edge"]["relief_strength"] != 0:
            raise ValueError("This editor uses automatic pressure-side leading relief and no trailing-edge relief. Set leading_edge.pressure_side to auto and trailing_edge.relief_strength to 0 in the YAML.")
        current = row_parameter_state(state, row_id)
        inherited = resolve_parameters(state_from_dict({**current.to_dict(), "overrides": {}})).values
        if imported["geometry"]["type"] != inherited["geometry"]["type"]:
            raise ValueError("The row stream surface must match its turbine. Change axial/radial configuration in Meanline first.")
        overrides = {path: _lookup(imported, path) for path in PATHS
                     if _lookup(imported, path) != _lookup(inherited, path)}
        candidate = state_from_dict({**current.to_dict(), "overrides": overrides})
        build_section(resolve_parameters(candidate).values)
        updated = deepcopy(state)
        next(item for item in updated["rows"] if item["id"] == row_id)["parameters"] = candidate.to_dict()
        updated["revision"] = state.get("revision", 0) + 1
        return updated
    except (ValueError, TypeError, IndexError, AttributeError, UnicodeError, binascii.Error, yaml.YAMLError) as error:
        raise ValueError(f"Could not import geometry: {error}") from error


def fold_edit_journal(current, event):
    """Replay pending client actions against its last acknowledged valid state.

    Every request contains the accepted baseline-plus-overrides and *all*
    unacknowledged actions. Two rapid fields therefore never depend on a previous
    response having arrived. Acknowledgments bound the live drag history; reset
    and import retain their order. Rejected actions preserve effective geometry.
    """
    if not event or event.get("epoch") != current.get("_epoch"):
        raise PreventUpdate
    if event.get("sequence", -1) <= current.get("revision", 0):
        raise PreventUpdate
    base = event.get("base")
    if not base or base.get("_epoch") != current.get("_epoch") or base.get("baseline_id") != current.get("baseline_id"):
        raise PreventUpdate
    state = deepcopy(base)
    errors = []
    rejected = set()
    for action in event.get("actions", []):
        if action.get("sequence", 0) <= base.get("revision", 0):
            continue
        row_id, kind = action["row"], action["kind"]
        try:
            if kind == "set":
                if action["path"] in HIDDEN_PATHS:
                    raise ValueError("This relief setting is not editable in the geometry workspace.")
                candidate = update_geometry_override(state, row_id, action["path"],
                                                     from_control(action["path"], action.get("value")))
            elif kind == "reset":
                candidate = reset_geometry_overrides(state, row_id, action.get("path"))
            elif kind == "import":
                candidate = import_row_yaml(action["contents"], state, row_id)
            else:
                raise ValueError("Unknown geometry edit action.")
            get_row_section(candidate, row_id, check_intersections=False)
            state = candidate
        except (ValueError, KeyError, TypeError) as error:
            errors.append(f"Edit not applied: {error} Previous valid geometry is shown.")
            rejected.add((row_id, action.get("path")))
    state["revision"] = int(event["sequence"])
    return state, errors, rejected


def _field(path):
    spec = FIELD_SPECS[path]
    identifier = {"type": "blade-parameter", "path": path}
    if spec["type"] == "enum":
        control = dcc.Dropdown(id=identifier, options=[{"label": item.capitalize(), "value": item}
                               for item in spec["options"]], value=None, clearable=False,
                               disabled=path == "geometry.type")
    elif spec["type"] == "bool":
        control = dcc.Checklist(id=identifier, options=[{"label": "", "value": "enabled"}], value=[])
    else:
        # No float step: arbitrary precision is independent of the slider grid.
        control = dcc.Input(id=identifier, type="number", value=None,
                            step=1 if spec["type"] == "int" else "any", debounce=True,
                            className="td-blade-number")
    units = "mm" if spec["units"] == "m" else "°" if spec["units"] == "deg" else ""
    label = spec["label"] + (f" [{units}]" if units else "")
    children = [html.Div([html.Div([
                          html.Label(label, htmlFor=json.dumps(identifier, sort_keys=True, separators=(",", ":"))),
                          _hint(PARAMETER_HELP[path], "blade-help-" + path.replace(".", "-")),
                          ], className="td-blade-label"),
                          control], className="td-blade-input-line")]
    if path == "geometry.type":
        children.append(html.Span("Set by the Meanline turbine configuration.", className="td-blade-caption"))
    if path in SLIDER_RANGES:
        children.append(dcc.Slider(id={"type": "blade-slider", "path": path}, value=None,
                                   marks={value: f"{value:g}" for value in SLIDER_RANGES[path][:2]},
                                   updatemode="mouseup", className="td-blade-slider",
                                   **slider_settings(path, None)))
    children.append(html.Div([
        html.Span("Inherited", id={"type": "blade-origin", "path": path}),
        html.Button("Reset", id={"type": "blade-reset", "path": path}, n_clicks=0,
                    disabled=True, title=f"Reset {spec['label'].lower()} to the calculated baseline"),
    ], className="td-blade-provenance"))
    return html.Div(children, id={"type": "blade-field", "path": path}, className="td-blade-field",
                    hidden=path in HIDDEN_PATHS,
                    **{"data-parameter-label": label, "data-parameter-aliases": path})


def controls():
    def fields(*prefixes):
        return [_field(path) for path in PATHS if path not in HIDDEN_PATHS and any(path.startswith(prefix) for prefix in prefixes)]

    def submenu(title, children, *, opened=False):
        return html.Details([html.Summary(title), *children], open=opened, className="td-subsection td-blade-subsection")

    def section(number, title, children):
        return html.Details([
            html.Summary([html.Span(f"{number:02}", className="td-section-number"),
                          html.Span(title, className="td-section-title")]),
            html.Div(children, className="td-section-body"),
        ], id=f"blade-section-{number}", className="td-section td-blade-group", open=False)

    body = [path for path in PATHS if path.startswith("thickness.") and not any(
        word in path for word in ("leading_edge_thickness", "trailing_edge_thickness", "edge_fraction", "rounding_exponent"))]
    return html.Div([
        html.Div(id="blade_modeller_context", **{"data-row": "", "data-epoch": ""}),
        html.Div("Calculate a turbine to prepare its blade rows.", id="blade_modeller_source", role="status"),
        html.P("Edit this blade row without changing the meanline calculation.",
               className="td-blade-caption"),
        html.Div([
            html.Button("Row YAML", id="blade_yaml_open", n_clicks=0, className="td-action",
                        title="View, copy, paste, or load this row's YAML parameters"),
            html.Button("Restore preset defaults", id="blade_modeller_reset", n_clicks=0, className="td-action",
                        title="Clear this row's overrides and restore its preset shape with the calculated meanline dimensions"),
        ], className="td-blade-actions"),
        html.Div(id="blade_modeller_error", role="alert", className="td-blade-error"),
        section(1, "Meridional channel", fields("geometry.", "cascade.")),
        section(2, "Camberline and turning", fields("camberline.")),
        section(3, "Thickness distribution", [
            submenu("Body and shoulders", [_field(path) for path in body], opened=True),
            submenu("Leading edge", fields("thickness.leading_edge_thickness", "thickness.leading_edge_fraction",
                                           "thickness.leading_edge_rounding", "leading_edge.relief_strength") + [
                html.P("Relief is applied to the pressure side using the blade's turning direction.", className="td-blade-caption")]),
            submenu("Trailing-edge rounding", fields("thickness.trailing_edge_thickness", "thickness.trailing_edge_fraction",
                                                      "thickness.trailing_edge_rounding")),
            submenu("Physical thickness correction", fields("correction.")),
        ]),
        section(4, "Flow domain", fields("flow_domain.") + [
            submenu("Advanced: resolution", fields("numerics.") + [
                html.P("Sampling controls numerical resolution, not the blade shape law.", className="td-blade-caption")]),
        ]),
        *[_field(path) for path in PATHS if path in HIDDEN_PATHS],
    ], id="blade_modeller_controls", className="td-blade-modeller")


def geometry_panel(*, workbench=False):
    """Preserve the shared workspace shape and export IDs used by all editions."""
    panel = _mesh_geometry_panel(workbench=workbench)
    side = next(child for child in panel.children if isinstance(child, html.Aside))
    side.children = [
        html.Div([
            html.Label("Stage", htmlFor="geometry_stage", className="td-blade-selector-label"),
            dcc.Dropdown(id="geometry_stage", options=[], value=0, clearable=False,
                         className="td-blade-stage", searchable=False),
            html.Label("Blade row", className="td-blade-selector-label"),
            dcc.RadioItems(id="geometry_row", options=[{"label": "Stator", "value": "stator"},
                                                       {"label": "Rotor", "value": "rotor"}],
                           value="stator", className="td-blade-row-switch", inline=True),
        ], className="td-blade-selector"),
        html.Div(id="geometry_rows_container"), controls(),
    ]
    main = next(child for child in panel.children if isinstance(child, html.Main))
    main.children = [
        yaml_dialog(),
        *[dcc.Store(id=name) for name in (
            "geometry_state", "geometry_committed", "geometry_results_store", "blade_modeller_drag",
            "blade_modeller_events", "blade_modeller_event_sink", "blade_modeller_ack_sink",
            "blade_modeller_2d_proposal", "blade_modeller_3d_proposal", "blade_modeller_edit_proposal", "blade_yaml_context", "blade_yaml_submission")],
        html.Div(id="geometry_message", role="status", className="td-blade-caption"),
        html.Section([
            html.Div([
                dcc.Checklist(id="blade_modeller_crosscuts", options=[{"label": "Passage crosscuts", "value": "show"}], value=[]),
                html.Span("Geometric opening between neighboring blades.", className="td-blade-caption"),
            ], className="td-blade-toolbar"),
            html.Div([_engineering_plot("Blade-to-blade view", "geometry_profile"),
                      _engineering_plot("Meridional channel", "geometry_meridional")], className="td-plot-pair"),
            html.Div(id="blade_modeller_passage_note", className="td-blade-caption"),
        ], className="td-plot-group"),
        html.Section(html.Div([_engineering_plot("Camberline", "blade_modeller_camber"),
                               _engineering_plot("Thickness distribution", "blade_modeller_thickness")],
                              className="td-plot-pair"), className="td-plot-group"),
        html.Section([
            html.H5("3D blade preview"),
            html.P("The preview refreshes when an edit is committed or a slider is released.", className="td-blade-caption"),
            html.Div(vtk.vtk_panel("geometry", "Calculate a turbine to view its blades."),
                     id="blade_modeller_preview", className="td-blade-preview"),
            html.Div(id="blade_modeller_preview_note", role="status", className="td-blade-caption"),
        ], className="td-plot-group"),
        html.Details([html.Summary("TurboGrid export"), turbogrid_export_panel()], className="td-row-settings"),
    ]
    return panel


def yaml_dialog():
    return dbc.Modal([
        dcc.Download(id="blade_modeller_download"),
        dbc.ModalHeader(dbc.ModalTitle("Row parameters", id="blade_yaml_title"), close_button=False),
        dbc.ModalBody([
            html.P("Copy this row's parameters, paste another YAML definition, or choose a file. Changes take effect only when you select Apply to row.",
                   className="td-blade-caption"),
            html.Div([
                dcc.Upload(id="blade_modeller_upload", accept=".yaml,.yml", multiple=False,
                           children=html.Button("Choose YAML file", className="td-yaml-button")),
                html.Div([dcc.Clipboard(id="blade_yaml_copy", target_id="blade_yaml_text", title="Copy YAML", style={"display": "inline-block"}),
                          html.Span("Copy YAML")], className="td-yaml-copy"),
                html.Button("Download YAML", id="blade_modeller_save", n_clicks=0, className="td-yaml-button"),
            ], className="td-yaml-toolbar"),
            html.Div(id="blade_yaml_file_notice", role="status", className="td-blade-caption"),
            html.Label("Row YAML", htmlFor="blade_yaml_text", className="visually-hidden"),
            dcc.Textarea(id="blade_yaml_text", value="", spellCheck=False, className="td-yaml-code"),
            html.Div(id="blade_yaml_feedback", role="status", className="td-yaml-feedback"),
        ]),
        dbc.ModalFooter([
            html.Button("Close", id="blade_yaml_close", n_clicks=0, className="td-yaml-button"),
            html.Button("Apply to row", id="blade_yaml_apply", n_clicks=0, disabled=True, className="td-yaml-button td-yaml-primary"),
        ]),
    ], id="blade_yaml_modal", is_open=False, size="lg", centered=True,
       backdrop="static", className="td-blade-yaml")


def row_yaml_document(state, stage, row_name, pending, result_timestamp):
    """Capture accepted row intent, including the edit blurred by opening YAML."""
    if state and state.get("_result_timestamp") != result_timestamp:
        raise PreventUpdate
    if state and pending and pending.get("epoch") == state.get("_epoch") and pending.get("sequence", 0) > state.get("revision", 0):
        state, errors, _ = fold_edit_journal(state, pending)
        if errors:
            raise ValueError("Resolve the pending invalid geometry edit before opening YAML.")
    selected = selected_row(state, stage, row_name)
    if selected is None:
        raise PreventUpdate
    name = re.sub(r"[^\w-]+", "_", selected["label"]).strip("_")
    return {"content": yaml.safe_dump(row_parameter_state(state, selected["id"]).to_dict(), sort_keys=False, allow_unicode=True),
            "filename": f"{name}_geometry.yaml", "type": "application/yaml"}


def _theme_figure(figure, row, row_id):
    """Use the workbench's semantic trace roles, preserving scientific data."""
    figure.update_layout(font={"family": "Segoe UI, sans-serif"}, uirevision=f"blade:{row_id}")
    for trace in figure.data:
        color = getattr(getattr(trace, "line", None), "color", None)
        role = "stator" if color == "#c77b32" else "rotor" if color == "#3478ac" else "endwall"
        trace.meta = {"theme_role": role}
        if hasattr(trace, "line"):
            trace.line.simplify = False
            if color == "#c77b32":
                trace.line.color = "#ff7f0e"
            elif color == "#3478ac":
                trace.line.color = "#1f77b4"
    return figure


def evaluate_2d(state, row_id, *, crosscuts=False):
    """Cached authoritative section, with expensive postprocessing opt-in."""
    section = get_row_section(state, row_id, check_intersections=False)
    figures = build_figures(section)
    note = ""
    if crosscuts:
        from turbodash.geometry.blade_parametrization.passage import section_passage
        try:
            passage = section_passage(section, number_of_cuts=25)
            figures["blade_to_blade"].add_traces(passage_traces(passage))
            if passage.get("valid_count"):
                widths = passage["widths_mm"]
                note = f"Opening {min(widths):.3f}–{max(widths):.3f} mm; {passage['valid_count']} geometric crosscuts."
            else:
                note = passage.get("message", "No valid crosscuts for this geometry.")
        except ValueError as error:
            note = f"Crosscuts unavailable: {error}"
    row = next(item for item in state["rows"] if item["id"] == row_id)
    for figure in figures.values():
        _theme_figure(figure, row["row"], row_id)
    figures["thickness"].update_layout(margin={"b": 95}, legend={"y": -.32})
    diagnostics = section.diagnostics
    message = (f"{row['label']} · maximum paired thickness {diagnostics['actual_maximum_thickness_mm']:.3f} mm"
               f" · pitch {diagnostics['physical_pitch_mm']:.3f} mm")
    warnings = list(dict.fromkeys([*row.get("warnings", []), *section.warnings]))
    if warnings:
        message += ". " + " ".join(warnings)
    return {"token": state_token(state, row_id), "figures": figures, "message": message,
            "passage_note": note, "crosscuts": bool(crosscuts)}


def evaluate_3d(state, row_id):
    section = get_row_section(state, row_id, check_intersections=False)
    row = next(item for item in state["rows"] if item["id"] == row_id)
    preview = section_preview(section, row=row["row"], stage_number=row["stage_index"] + 1)
    data = vtk.geometry_data(preview)
    try:
        get_row_section(state, row_id, check_intersections=True)
        note = "Current committed geometry; sampled intersection checks passed."
    except ValueError as error:
        note = f"Geometry requires review before CFD: {error}"
    if preview.get("diagnostic"):
        note += " " + preview["diagnostic"]
    return {"token": state_token(state, row_id), "data": data, "message": note}


def register_callbacks(app, *, results_store="result_store", loaded_config_store=None,
                       normalize_inputs=None):
    """Register the sole geometry-state writer and separate live/commit views."""
    app.clientside_callback(
        "function(low, high) { return {[low]: String(low), [high]: String(high)}; }",
        Output({"type": "blade-slider", "path": MATCH}, "marks"),
        Input({"type": "blade-slider", "path": MATCH}, "min"),
        Input({"type": "blade-slider", "path": MATCH}, "max"),
    )
    vtk.register_vtk_callbacks(app, panes=("geometry",))
    app.clientside_callback(
        ClientsideFunction(namespace="bladeModeller", function_name="sampleDrag"),
        Output("blade_modeller_drag", "data"), Input({"type": "blade-slider", "path": ALL}, "drag_value"),
        State({"type": "blade-slider", "path": ALL}, "id"), prevent_initial_call=True)
    app.clientside_callback(
        ClientsideFunction(namespace="bladeModeller", function_name="acknowledge"),
        Output("blade_modeller_ack_sink", "data"), Input("geometry_state", "data"), prevent_initial_call=True)
    app.clientside_callback(
        ClientsideFunction(namespace="bladeModeller", function_name="forwardControls"),
        Output("blade_modeller_event_sink", "data"),
        Input({"type": "blade-parameter", "path": ALL}, "value"),
        Input({"type": "blade-reset", "path": ALL}, "n_clicks"),
        Input("blade_modeller_reset", "n_clicks"),
        State({"type": "blade-parameter", "path": ALL}, "id"),
        State({"type": "blade-reset", "path": ALL}, "id"), prevent_initial_call=True)

    def edit(results, stage, row_name, event, identifiers, slider_ids, current, committed, result_timestamp,
             loaded_cfg=None, loaded_timestamp=None):
        trigger = ctx.triggered_id
        dragging = trigger == "blade_modeller_events" and bool((event or {}).get("dragging"))
        notice, edited_path = "", None
        state_changed = True
        rejected = set()
        refresh_fields = set()
        # A queued edit can carry the previous state while the newest accepted
        # meanline result is already in the browser. Its result timestamp wins,
        # including a repeat calculation with identical geometric dimensions.
        loading = loaded_cfg is not None and (
            not current or current.get("_loaded_geometry_timestamp") != loaded_timestamp
            or current.get("_loaded_geometry_pending_result", False))
        result_after_load = (result_timestamp is not None and loaded_timestamp is not None
                             and result_timestamp >= loaded_timestamp)
        loaded_inputs_match = loading and normalize_inputs is not None and results and (
            normalize_inputs(loaded_cfg) == normalize_inputs(results))
        if loaded_config_store and trigger == loaded_config_store and not loaded_inputs_match:
            # The upload arrives before its overall/stage controls and solve.
            # Keep the previous geometry until the matching result is accepted.
            raise PreventUpdate
        refresh = trigger in (results_store, loaded_config_store) or (results and
                  (not current or current.get("_result_timestamp") != result_timestamp))
        if refresh:
            if not results or not results.get("stages_performance"):
                raise PreventUpdate
            try:
                if loaded_inputs_match and "geometry" in loaded_cfg:
                    state = restore_geometry(results, loaded_cfg["geometry"])
                else:
                    state = initialize_geometry(results)
                if loaded_inputs_match:
                    state["_loaded_geometry_timestamp"] = loaded_timestamp
                    # Identical meanline controls need no new solve. Restore
                    # against that accepted result immediately, and retain the
                    # load if an already queued matching solve arrives later.
                    state["_loaded_geometry_pending_result"] = not result_after_load
                elif current and "_loaded_geometry_timestamp" in current:
                    state["_loaded_geometry_timestamp"] = current["_loaded_geometry_timestamp"]
                state["_epoch"] = uuid.uuid4().hex
                state["_result_timestamp"] = result_timestamp
                dragging = False
            except (ValueError, KeyError, TypeError) as error:
                fields, slider_fields = [no_update] * len(PATHS), [no_update] * len(SLIDER_PATHS)
                return (no_update, no_update, fields, slider_fields, slider_fields, slider_fields,
                        fields, fields, fields, no_update, f"Could not initialize blade geometry: {error}", no_update, no_update)
        else:
            if not current:
                raise PreventUpdate
            state = current
        selected = selected_row(state, stage, row_name)
        if selected is None:
            if not state.get("rows"):
                raise PreventUpdate
            selected = state["rows"][0]
        row_id = selected["id"]
        try:
            if refresh:
                pass
            elif trigger in ("geometry_stage", "geometry_row"):
                state_changed = False
            elif trigger == "blade_modeller_events":
                state, errors, rejected = fold_edit_journal(current, event)
                latest = (event or {}).get("latest", {})
                edited_path = latest.get("path")
                if latest.get("row") != row_id:
                    edited_path = "__other_row__"
                # A reset/import response can be superseded by a rapid later
                # field edit. Project every affected value from the cumulative
                # transaction, not only that final edited field.
                for action in event.get("actions", []):
                    if action["row"] == row_id and action["kind"] in {"reset", "import"}:
                        refresh_fields.update([action["path"]] if action.get("path") else PATHS)
                if errors:
                    notice = " ".join(dict.fromkeys(errors))
                elif latest.get("kind") == "import":
                    notice = "YAML applied to this row. Reset restores the current meanline baseline."
            else:
                raise PreventUpdate
        except (ValueError, KeyError, TypeError) as error:
            fields, slider_fields = [no_update] * len(PATHS), [no_update] * len(SLIDER_PATHS)
            return (no_update, no_update, fields, slider_fields, slider_fields, slider_fields,
                    fields, fields, fields, no_update, f"Edit not applied: {error} Previous valid geometry is shown.", no_update, no_update)
        projected, labels, disabled, styles = input_view(state, row_id)
        by_path = dict(zip(PATHS, projected))
        slider_paths = [item["path"] for item in slider_ids]
        field_paths = [item["path"] for item in identifiers]
        settings = [slider_settings(path, by_path[path]) for path in slider_paths]
        # Preserve the DOM value/selection in every unrelated numeric field.
        shown_values = [value if not edited_path or path == edited_path or path in refresh_fields or (row_id, path) in rejected else no_update
                        for path, value in zip(PATHS, projected)]
        shown_values = dict(zip(PATHS, shown_values))
        labels, disabled, styles = (dict(zip(PATHS, items)) for items in (labels, disabled, styles))
        slider_values = [by_path[path] for path in slider_paths]
        if dragging:
            slider_values = [no_update] * len(SLIDER_PATHS)
        selected = next(item for item in state["rows"] if item["id"] == row_id)
        count = len(selected["parameters"].get("overrides", {}))
        label = f"{selected['label']} · {count} edited parameter{'s' if count != 1 else ''}"
        return (state if state_changed else no_update, state if state_changed and not dragging else no_update,
                [shown_values[path] for path in field_paths], slider_values,
                [item["min"] for item in settings], [item["max"] for item in settings],
                [labels[path] for path in field_paths], [disabled[path] for path in field_paths],
                [styles[path] for path in field_paths], label, notice, row_id, state.get("_epoch", ""))

    @app.callback(
        Output("blade_modeller_edit_proposal", "data"),
        Input(results_store, "data"), Input("geometry_stage", "value"), Input("geometry_row", "value"),
        Input("blade_modeller_events", "data"),
        State({"type": "blade-parameter", "path": ALL}, "id"), State({"type": "blade-slider", "path": ALL}, "id"),
        State("geometry_state", "data"), State("geometry_committed", "data"),
        State(results_store, "modified_timestamp"),
        *([State(loaded_config_store, "data"), Input(loaded_config_store, "modified_timestamp")]
          if loaded_config_store else []),
        prevent_initial_call=True,
    )
    def prepare_edit(*args):
        values = edit(*args)
        proposed = values[0] if values[0] is not no_update else args[6]
        proposal = {"values": values, "stage": args[1] or 0, "row": args[2] or "stator",
                    "epoch": (proposed or {}).get("_epoch"), "revision": (proposed or {}).get("revision", 0),
                    "result_timestamp": args[8]}
        if loaded_config_store:
            proposal["loaded_timestamp"] = args[10]
        return proposal

    # Expose the pure transition for unit tests; only acceptEdit writes browser state.
    prepare_edit.transition = edit
    app.clientside_callback(
        ClientsideFunction(namespace="bladeModeller", function_name="acceptEdit"),
        Output("geometry_state", "data"), Output("geometry_committed", "data"),
        Output({"type": "blade-parameter", "path": ALL}, "value"),
        Output({"type": "blade-slider", "path": ALL}, "value"),
        Output({"type": "blade-slider", "path": ALL}, "min"), Output({"type": "blade-slider", "path": ALL}, "max"),
        Output({"type": "blade-origin", "path": ALL}, "children"),
        Output({"type": "blade-reset", "path": ALL}, "disabled"), Output({"type": "blade-field", "path": ALL}, "style"),
        Output("blade_modeller_source", "children"), Output("blade_modeller_error", "children"),
        Output("blade_modeller_context", "data-row"), Output("blade_modeller_context", "data-epoch"),
        Input("blade_modeller_edit_proposal", "data"), State("geometry_state", "data"),
        State("geometry_stage", "value"), State("geometry_row", "value"),
        State(results_store, "modified_timestamp"),
        *([State(loaded_config_store, "modified_timestamp")] if loaded_config_store else []),
        prevent_initial_call=True)

    @app.callback(Output("blade_modeller_2d_proposal", "data"), Input("geometry_state", "data"),
                  Input("geometry_stage", "value"), Input("geometry_row", "value"),
                  Input("blade_modeller_crosscuts", "value"))
    def prepare_2d(state, stage, row_name, crosscuts):
        selected = selected_row(state, stage, row_name)
        if selected is None:
            raise PreventUpdate
        try:
            return evaluate_2d(state, selected["id"], crosscuts="show" in (crosscuts or []))
        except (ValueError, TypeError, KeyError) as error:
            return {"token": state_token(state, selected["id"]), "error": f"Geometry could not be displayed: {error}"}

    app.clientside_callback(
        ClientsideFunction(namespace="bladeModeller", function_name="accept2d"),
        Output("geometry_profile", "figure"), Output("geometry_meridional", "figure"),
        Output("blade_modeller_camber", "figure"), Output("blade_modeller_thickness", "figure"),
        Output("geometry_message", "children"), Output("blade_modeller_passage_note", "children"),
        Input("blade_modeller_2d_proposal", "data"), State("geometry_state", "data"),
        State("geometry_stage", "value"), State("geometry_row", "value"),
        State("blade_modeller_crosscuts", "value"), prevent_initial_call=True)

    @app.callback(Output("blade_modeller_3d_proposal", "data"), Input("geometry_committed", "data"),
                  Input("geometry_stage", "value"), Input("geometry_row", "value"))
    def prepare_3d(state, stage, row_name):
        selected = selected_row(state, stage, row_name)
        if selected is None:
            raise PreventUpdate
        try:
            return evaluate_3d(state, selected["id"])
        except (ValueError, TypeError, KeyError, ImportError, RuntimeError) as error:
            message = f"3D preview unavailable: {error}"
            return {"token": state_token(state, selected["id"]), "error": message,
                    "data": vtk.empty_data(message)}

    app.clientside_callback(
        ClientsideFunction(namespace="bladeModeller", function_name="accept3d"),
        Output("geometry-vtk-data", "data"), Output("blade_modeller_preview_note", "children"),
        Input("blade_modeller_3d_proposal", "data"), State("geometry_committed", "data"),
        State("geometry_stage", "value"), State("geometry_row", "value"), prevent_initial_call=True)

    @app.callback(
        Output("blade_yaml_modal", "is_open"), Output("blade_yaml_text", "value"),
        Output("blade_yaml_context", "data"), Output("blade_yaml_title", "children"),
        Output("blade_yaml_file_notice", "children"),
        Input("blade_yaml_open", "n_clicks"), Input("blade_yaml_close", "n_clicks"),
        Input("blade_modeller_upload", "contents"),
        State("geometry_state", "data"), State("geometry_stage", "value"), State("geometry_row", "value"),
        State("blade_modeller_events", "data"), State(results_store, "modified_timestamp"),
        State("blade_modeller_upload", "filename"), prevent_initial_call=True)
    def edit_yaml(open_clicks, close_clicks, contents, state, stage, row_name, pending, result_timestamp, filename):
        trigger = ctx.triggered_id
        if trigger == "blade_yaml_close":
            return False, no_update, no_update, no_update, ""
        if trigger == "blade_modeller_upload":
            if not contents:
                raise PreventUpdate
            try:
                payload = contents.split(",", 1)[1]
                if len(payload) > 1_400_000:
                    raise ValueError("Choose a YAML file smaller than 1 MB.")
                draft = base64.b64decode(payload, validate=True).decode("utf-8-sig")
                return no_update, draft, no_update, no_update, f"Loaded {filename or 'YAML file'} into the editor. Select Apply to row to use it."
            except (ValueError, IndexError, UnicodeError, binascii.Error) as error:
                return no_update, no_update, no_update, no_update, f"Could not read file: {error}"
        if trigger != "blade_yaml_open" or not open_clicks:
            raise PreventUpdate
        try:
            document = row_yaml_document(state, stage, row_name, pending, result_timestamp)
        except (PreventUpdate, ValueError) as error:
            return True, "", None, "Row parameters", str(error) or "Wait for a successful calculation before opening row parameters."
        selected = selected_row(state, stage, row_name)
        captured = {"epoch": state["_epoch"], "row": selected["id"], "filename": document["filename"],
                    "label": selected["label"], "id": uuid.uuid4().hex}
        return True, document["content"], captured, selected["label"] + " — YAML", ""

    app.clientside_callback(
        ClientsideFunction(namespace="bladeModeller", function_name="applyYaml"),
        Output("blade_yaml_submission", "data"), Input("blade_yaml_apply", "n_clicks"),
        State("blade_yaml_text", "value"), State("blade_yaml_context", "data"), prevent_initial_call=True)

    app.clientside_callback(
        "function(captured, state) { return !captured || !state || captured.epoch !== state._epoch; }",
        Output("blade_yaml_apply", "disabled"),
        Input("blade_yaml_context", "data"), Input("geometry_state", "data"))

    @app.callback(Output("blade_yaml_feedback", "children"),
                  Input("blade_yaml_submission", "data"), Input("geometry_state", "data"),
                  Input("blade_modeller_error", "children"), Input("blade_yaml_context", "data"))
    def yaml_feedback(submission, state, error, captured):
        if not captured:
            return ""
        if not state or state.get("_epoch") != captured["epoch"]:
            return "The calculated baseline changed. Close and reopen Row YAML before applying."
        if not submission or submission.get("dialog") != captured["id"]:
            return ""
        if submission.get("error"):
            return submission["error"]
        if state.get("revision", 0) < submission["sequence"]:
            return "Validating row parameters…"
        return error or f"Applied to {captured['label']}."

    @app.callback(Output("blade_modeller_download", "data"), Input("blade_modeller_save", "n_clicks"),
                  State("blade_yaml_text", "value"), State("blade_yaml_context", "data"), prevent_initial_call=True)
    def save_yaml(clicks, draft, captured):
        if not clicks or not draft:
            raise PreventUpdate
        return {"content": draft, "filename": (captured or {}).get("filename", "blade_row.yaml"), "type": "application/yaml"}
