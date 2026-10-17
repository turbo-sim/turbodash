"""Geometry and mesh tabs, with optional TurboMOC integration.

Only opaque job identifiers enter browser stores. Gmsh runs in a separate
process; changing controls invalidates its result and never starts a new mesh.
"""

import hashlib
import json
import logging
from functools import lru_cache

import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import ALL, Input, Output, State, ctx, dcc, html, no_update
from dash.exceptions import PreventUpdate
from plotly.utils import PlotlyJSONEncoder

from . import meshing, mesh_jobs
from .plotting_mesh import plot_profile_preview, plot_geometry_meridional, mesh_figures
from . import visualization_vtk as vtk
from .vtk_assets import available as vtk_assets_available
from .app_turbogrid import turbogrid_export_panel


_LOG = logging.getLogger(__name__)
_SETTINGS = (
    ("blade_size_mm", "Blade element size [mm]", None, 0.000001, None, "any"),
    ("domain_size_mm", "Passage element size [mm]", None, 0.000001, None, "any"),
    ("span_layers", "Spanwise layers", 12, 1, 200, 1),
    ("span_stretch", "Spanwise stretching (0 = uniform)", 1.5, 0, 8, 0.1),
    ("profile_points", "Profile points", 200, 40, 1000, 10),
    ("inlet_chords", "Inlet extension / chord", 0.5, 0.05, 10, 0.05),
    ("outlet_chords", "Outlet extension / chord", 1.0, 0.05, 10, 0.05),
    ("first_height_mm", "First inflation layer height [mm]", None, 0.000001, None, "any"),
    ("inflation_thickness_mm", "Inflation thickness [mm]", None, 0.000001, None, "any"),
    ("inflation_growth", "Inflation growth ratio", 1.2, 1.001, 3, 0.05),
)
_OPTION_KEYS = [item[0] for item in _SETTINGS] + ["inflation_enabled"]
_LIVE = [
    ({"scope": "overall", "key": ALL}, "value"),
    ({"scope": "stage", "stage": ALL, "key": ALL, "elem": "input"}, "value"),
    ({"scope": "overall", "key": ALL}, "id"),
    ({"scope": "stage", "stage": ALL, "key": ALL, "elem": "input"}, "id"),
    ("stage_data_store", "data"),
]


def input_signature(overall_values, stage_values, overall_ids, stage_ids, stages):
    """Identify the exact input state that produced a successful calculation."""
    def pairs(ids, values):
        return sorted(
            (json.dumps(key, sort_keys=True), value)
            for key, value in zip(ids or [], values or [])
        )

    payload = [pairs(overall_ids, overall_values), pairs(stage_ids, stage_values), len(stages or [])]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _empty(message):
    figure = go.Figure()
    figure.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
    figure.update_layout(template="plotly_white", height=470, xaxis_visible=False, yaxis_visible=False)
    return figure


@lru_cache(maxsize=12)
def _cached_preview(serialized_results, stage, row):
    """Retain a bounded set of read-only previews across tab/camera changes."""
    return meshing.preview_blade_row(json.loads(serialized_results), stage_index=stage, row=row)


def _preview_key(results):
    return json.dumps(results, sort_keys=True, cls=PlotlyJSONEncoder)


def _visible(style):
    return isinstance(style, dict) and style.get("display") != "none"


def _unavailable(results, calculated_signature, live_signature):
    if not results or not results.get("stages_performance"):
        return "Calculate a turbine in the Meanline tab first."
    if calculated_signature != live_signature:
        return "The design inputs have changed. Wait for a successful meanline calculation."
    if results.get("inputs", {}).get("turbine_type") != "axial":
        return "3D models and meshes currently support axial turbines only."
    available, message = meshing.dependency_status()
    if not available:
        return message
    if not vtk_assets_available():
        return "3D viewing requires the compatible TurboMoC VTK renderer. Install TurboDash with its mesh extra."
    return ""


def _row_selector(prefix):
    return dbc.Row([
        dbc.Col([dbc.Label("Stage"), dcc.Dropdown(id=f"{prefix}_stage", options=[], value=0, clearable=False)], md=6),
        dbc.Col([dbc.Label("Blade row"), dcc.Dropdown(id=f"{prefix}_row", options=[
            {"label": "Stator", "value": "stator"}, {"label": "Rotor", "value": "rotor"},
        ], value="stator", clearable=False)], md=6),
    ], className="mb-3")


def _card(title, children):
    return dbc.Card([dbc.CardHeader(title), dbc.CardBody(children)], className="mb-3")


def geometry_panel():
    """Keep these components mounted while the tab is hidden."""
    return html.Div([
        html.Aside([
            html.H4("Turbine geometry"),
            html.P("Edit the stage and blade parameters here or in Meanline. Physical geometry rounds blade counts up and recomputes pitch and profile dimensions. Meanline performance uses continuous counts.", className="small text-muted"),
            html.Details([
                html.Summary("Inspect and export a row"),
                _row_selector("geometry"),
                dbc.Button("Download 3D model ZIP", id="geometry_download_button", n_clicks=0, disabled=True, color="primary"),
                dcc.Download(id="geometry_download"),
                html.Div(id="geometry_download_error", role="alert", className="text-danger mt-2"),
            ], open=True, className="td-row-settings"),
            html.Div(id="geometry_rows_container"),
        ], className="td-sidebar"),
        html.Main([
            html.Div(id="geometry_message", role="status", className="small text-muted mb-3"),
            html.Section([
                html.Div([
                    _engineering_plot("Blade-to-blade view", "geometry_profile"),
                    _engineering_plot("Meridional channel", "geometry_meridional"),
                ], className="td-plot-pair"),
            ], className="td-plot-group"),
            html.Section([
                html.H5("3D turbine"),
                html.Div(id="geometry_overview", children=vtk.vtk_panel("overview", message="Calculate a turbine to view its blade rows.")),
            ], className="td-plot-group"),
            html.Details([html.Summary("TurboGrid export"), turbogrid_export_panel()], className="td-row-settings"),
            # Retain the legacy row host without presenting a duplicate viewport.
            html.Div(id="geometry_model", children=vtk.vtk_panel("geometry"), style={"display": "none"}),
        ], className="td-workspace-main"),
    ], className="td-workspace")


def _engineering_plot(title, graph_id):
    return html.Div([
        html.H5(title),
        dcc.Graph(id=graph_id, figure=_empty("Calculate a turbine to view this section."),
                  responsive=True, style={"height": "100%", "minHeight": 0},
                  config={"responsive": True, "displaylogo": False}),
    ], className="td-engineering-plot")


def mesh_row_panels(stages, values=None, ids=None):
    """Each mounted row owns its settings, independently of the active row."""
    existing = {(item["stage"], item["row"], item["key"]): value
                for item, value in zip(ids or [], values or [])}
    panels = []
    for stage in range(len(stages or [])):
        rows = []
        for row in ("stator", "rotor"):
            fields = []
            for key, label, default, minimum, maximum, step in _SETTINGS:
                fields.append(html.Div([
                    html.Label(label),
                    dbc.Input(id={"scope": "mesh-row", "stage": stage, "row": row, "key": key},
                              type="number", value=existing.get((stage, row, key), default), min=minimum,
                              max=maximum, step=step, debounce=True, placeholder="Automatic" if default is None else None),
                ], className="td-parameter"))
            fields.extend([
                dbc.Checklist(id={"scope": "mesh-row", "stage": stage, "row": row, "key": "inflation_enabled"},
                    options=[{"label": "Blade boundary-layer inflation", "value": "enabled"}],
                    value=existing.get((stage, row, "inflation_enabled"), []), switch=True, className="my-3"),
                dbc.Button("Use this row", id={"scope": "mesh-row-select", "stage": stage, "row": row},
                           n_clicks=0, size="sm", color="primary", outline=True),
            ])
            rows.append(html.Details([html.Summary(row.title()), *fields],
                                     open=stage == 0 and row == "stator", className="td-row-settings"))
        panels.append(html.Details([html.Summary(f"Stage {stage + 1}"), *rows],
                                  open=stage == 0, className="td-row-settings"))
    return panels


def _inspection_controls_panel():
    return html.Details([
        html.Summary("Mesh inspection"),
        dbc.Row([
            dbc.Col([dbc.Label("View"), dcc.Dropdown(id="mesh_view", options=[
                {"label": "Boundary surfaces", "value": "surface"},
                {"label": "Slice", "value": "slice"},
                {"label": "Spanwise layer", "value": "layer"},
                {"label": "Worst quality cells", "value": "worst"},
                {"label": "Quality threshold", "value": "threshold"},
            ], value="surface", clearable=False)], width=12),
            dbc.Col([dbc.Label("Quality metric"), dcc.Dropdown(id="mesh_metric", options=[
                {"label": "Skewness", "value": "skewness"},
                {"label": "Orthogonality", "value": "orthogonality"},
                {"label": "Aspect ratio", "value": "aspect_ratio"},
            ], value="skewness", clearable=False)], width=12),
        ], className="mb-3"),
        dbc.Row([
            dbc.Col([dbc.Label("Slice axis"), dcc.Dropdown(id="mesh_axis", options=["X", "Y", "Z"], value="Z", clearable=False)], width=6),
            dbc.Col([dbc.Label("Spanwise layer"), dbc.Input(id="mesh_layer", type="number", min=1, step=1, value=1, debounce=True)], width=6),
            dbc.Col([dbc.Label("Worst-cell count"), dbc.Input(id="mesh_worst_count", type="number", min=1, max=500, step=1, value=20, debounce=True)], width=6),
            dbc.Col([dbc.Label("Threshold"), dbc.Input(id="mesh_threshold", type="number", value=0.8, step="any", debounce=True)], width=6),
        ], className="mb-3"),
        dbc.Label("Slice position [% of extent]"),
        dcc.Slider(id="mesh_fraction", min=0, max=100, step=1, value=50, marks={0: "0%", 50: "50%", 100: "100%"}, tooltip={"placement": "bottom"}),
        html.P("High skewness and aspect ratio, or low orthogonality, indicate poorer cells. Inspection changes do not regenerate the mesh.", className="small text-muted mt-3"),
        html.Div(id="mesh_inspection_message", className="small", role="status"),
    ], open=False, className="td-row-settings")


def mesh_panel():
    # Selected-row values feed the established generation callbacks. Editors
    # themselves are the persistent per-row controls in the sidebar.
    controls = [dbc.Input(id=f"mesh_{key}", type="number", value=default)
                for key, _, default, *_ in _SETTINGS]
    controls.append(dbc.Checklist(id="mesh_inflation_enabled", options=[{"label": "Inflation", "value": "enabled"}], value=[]))
    return html.Div([
        dcc.Store(id="mesh_job"),
        dcc.Store(id="mesh_status", data={"state": "idle", "message": "Choose a blade row, then generate a mesh."}),
        dcc.Interval(id="mesh_poll", interval=1500, disabled=True, n_intervals=0),
        dcc.Download(id="mesh_download"),
        dcc.Store(id="mesh_row_choice"),
        html.Div(controls, style={"display": "none"}),
        html.Aside([
                html.H4("Mesh generation"),
                html.P("Set each row independently, select the active row, then generate its mesh. Editing settings does not start a mesh job.", className="small text-muted"),
                html.Details([
                    html.Summary("Active row"),
                    _row_selector("mesh"),
                    dbc.Button("Generate mesh", id="mesh_generate", color="primary", n_clicks=0, disabled=True),
                    html.Div(id="mesh_availability", className="small mt-2", role="status"),
                ], open=True, className="td-row-settings"),
                html.Div(id="mesh_rows_container"),
                html.P("Blank sizes use chord-based defaults. Rows use constant radius and span; endwall flaring is omitted.", className="small text-muted"),
                html.Details([
                    html.Summary("Generation status and download"),
                    html.Div(id="mesh_status_text", role="status"),
                    dbc.Progress(id="mesh_progress", value=0, className="mt-3", style={"height": "8px"}),
                    html.Pre(id="mesh_summary", className="small mt-3", style={"whiteSpace": "pre-wrap"}),
                    dbc.Button("Download mesh ZIP", id="mesh_download_button", color="primary", outline=True, n_clicks=0, disabled=True),
                    html.Div(id="mesh_download_error", role="alert", className="text-danger mt-2"),
                ], open=True, className="td-row-settings"),
        ], className="td-sidebar"),
        html.Main([
            html.Div(id="mesh_2d_message", className="small text-muted mb-2", role="status"),
            html.Section([
                html.Div([_engineering_plot("Blade-to-blade mesh", "mesh_profile"),
                          _engineering_plot("Meridional mesh", "mesh_meridional")], className="td-plot-pair"),
            ], className="td-plot-group"),
            html.Section([html.H5("3D mesh"), html.Div(id="mesh_figure", children=vtk.vtk_panel("mesh", message="Generate a mesh to inspect it.", extra_controls=_inspection_controls_panel()))], className="td-plot-group"),
        ], className="td-workspace-main"),
    ], className="td-workspace")


def _options(values):
    options = {key: value for key, value in zip(_OPTION_KEYS, values) if value is not None}
    options["inflation_enabled"] = bool(options.get("inflation_enabled"))
    return options


def advance_mesh_job(trigger, results, stage, row, options, handle, *, reason="", jobs=None):
    """One explicit Generate event submits work; all edits invalidate it."""
    jobs = jobs or mesh_jobs
    idle = {"state": "idle", "message": reason or "Settings changed. Select Generate mesh to create a new mesh."}
    if reason:
        return None, idle, True
    if trigger == "mesh_generate":
        try:
            handle = jobs.start_job(results, stage, row, options)
            return handle, {"state": "running", "message": "Generating mesh…", "fraction": 0.0}, False
        except Exception as error:
            _LOG.exception("Mesh submission failed")
            return None, {"state": "failed", "message": str(error)}, True
    if trigger != "mesh_poll" or not handle:
        return None, idle, True
    try:
        if handle.get("fingerprint") != jobs.fingerprint(results, stage, row, options):
            return None, idle, True
        status = jobs.job_status(handle)
        safe_status = {key: status[key] for key in ("state", "message", "fraction") if key in status}
        if status.get("state") == "complete":
            manifest = status.get("manifest", {})
            metadata = manifest.get("metadata", {})
            if metadata:
                safe_status["metadata"] = {key: metadata[key] for key in ("stage_number", "row", "blade_count", "coordinate_units") if key in metadata}
            summary = manifest.get("summary", {})
            safe_status["summary"] = {key: summary[key] for key in ("volume_nodes", "cells", "quality", "elapsed_s") if key in summary}
            metrics = manifest.get("report", {}).get("metrics", {})
            if metrics:
                safe_status["metrics"] = {name: {key: values[key] for key in ("min", "mean", "max") if key in values} for name, values in metrics.items()}
        return handle, safe_status, status.get("state") not in ("queued", "running")
    except Exception as error:
        _LOG.exception("Mesh status failed")
        return None, {"state": "failed", "message": str(error)}, True


def _summary_text(summary, metrics=None):
    lines = []
    if "volume_nodes" in summary:
        lines.append(f"Nodes: {summary['volume_nodes']:,}")
    cells = summary.get("cells", {})
    if isinstance(cells, dict):
        lines.append(f"Volume cells: {sum(cells.values()):,}")
        lines.extend(f"  {name.replace('_', ' ').capitalize()}: {value:,}" for name, value in cells.items())
    elif cells:
        lines.append(f"Volume cells: {cells:,}")
    quality = summary.get("quality", {})
    for key, label in (("min_determinant_jacobian", "Minimum Jacobian"), ("min_sicn", "Minimum scaled inverse condition number"), ("max_periodic_error_mm", "Periodic mismatch [mm]")):
        if key in quality:
            lines.append(f"{label}: {quality[key]:.4g}")
    for name, values in (metrics or {}).items():
        lines.append(f"{name.replace('_', ' ').capitalize()}: {values['min']:.4g} to {values['max']:.4g} (mean {values['mean']:.4g})")
    return "\n".join(lines)


def register_meshing_callbacks(app):
    vtk.register_vtk_callbacks(app)

    @app.callback(
        Output("mesh_rows_container", "children"), Input("stage_data_store", "data"),
        State({"scope": "mesh-row", "stage": ALL, "row": ALL, "key": ALL}, "value"),
        State({"scope": "mesh-row", "stage": ALL, "row": ALL, "key": ALL}, "id"),
    )
    def render_mesh_rows(stages, values, ids):
        return mesh_row_panels(stages, values, ids)

    @app.callback(
        Output("mesh_row_choice", "data"),
        Input({"scope": "mesh-row-select", "stage": ALL, "row": ALL}, "n_clicks"),
        prevent_initial_call=True,
    )
    def choose_mesh_row(clicks):
        trigger = ctx.triggered_id
        if not isinstance(trigger, dict) or not any(clicks or []):
            raise PreventUpdate
        return {"stage": trigger["stage"], "row": trigger["row"]}

    @app.callback(
        Output("geometry_stage", "options"), Output("geometry_stage", "value"),
        Output("mesh_stage", "options"), Output("mesh_stage", "value"),
        Output("mesh_row", "value"),
        Input("result_store", "data"), Input("mesh_row_choice", "data"),
        State("geometry_stage", "value"), State("mesh_stage", "value"), State("mesh_row", "value"),
    )
    def update_stages(results, choice, geometry_stage, mesh_stage, mesh_row):
        count = len((results or {}).get("stages_performance", []))
        options = [{"label": f"Stage {i + 1}", "value": i} for i in range(count)]
        if ctx.triggered_id == "mesh_row_choice" and choice and choice.get("stage") in range(count):
            mesh_stage, mesh_row = choice["stage"], choice["row"]
        return options, geometry_stage if geometry_stage in range(count) else 0, options, mesh_stage if mesh_stage in range(count) else 0, mesh_row or "stator"

    @app.callback(
        *[Output(f"mesh_{key}", "value") for key in _OPTION_KEYS],
        Input("mesh_stage", "value"), Input("mesh_row", "value"),
        Input({"scope": "mesh-row", "stage": ALL, "row": ALL, "key": ALL}, "value"),
        State({"scope": "mesh-row", "stage": ALL, "row": ALL, "key": ALL}, "id"),
        *[State(f"mesh_{key}", "value") for key in _OPTION_KEYS],
    )
    def select_mesh_settings(stage, row, values, ids, *current):
        selected = {item["key"]: value for item, value in zip(ids or [], values or [])
                    if item["stage"] == stage and item["row"] == row}
        defaults = {item[0]: item[2] for item in _SETTINGS} | {"inflation_enabled": []}
        return tuple(no_update if selected.get(key, defaults[key]) == old else selected.get(key, defaults[key])
                     for key, old in zip(_OPTION_KEYS, current))

    @app.callback(
        Output("geometry_profile", "figure"), Output("geometry-vtk-data", "data"),
        Output("geometry_message", "children"), Output("geometry_download_button", "disabled"),
        Output("geometry_meridional", "figure"),
        Input("result_store", "data"), Input("result_input_signature", "data"),
        Input("geometry_stage", "value"), Input("geometry_row", "value"),
        Input("geometry_pane", "style"),
        *[Input(*item) for item in _LIVE],
    )
    def update_geometry(results, signature, stage, row, pane_style, *live):
        reason = _unavailable(results, signature, input_signature(*live))
        if reason:
            return _empty(reason), vtk.empty_data(reason), reason, True, _empty(reason)
        if not _visible(pane_style):
            raise PreventUpdate
        try:
            preview = _cached_preview(_preview_key(results), stage, row)
            metadata = preview.get("metadata", {})
            message = " ".join(metadata.get("notes", [])) or "Constant-radius, constant-span blade row."
            return plot_profile_preview(preview), vtk.geometry_data(preview), str(message), False, plot_geometry_meridional(preview)
        except Exception as error:
            _LOG.exception("Geometry preview failed")
            message = f"Could not build this blade row: {error}"
            return _empty(message), vtk.empty_data(message), message, True, _empty(message)

    @app.callback(
        Output("overview-vtk-data", "data"), Input("result_store", "data"),
        Input("result_input_signature", "data"), Input("geometry_pane", "style"),
        *[Input(*item) for item in _LIVE],
    )
    def update_overview(results, signature, pane_style, *live):
        reason = _unavailable(results, signature, input_signature(*live))
        if reason:
            return vtk.empty_data(reason)
        if not _visible(pane_style):
            raise PreventUpdate
        try:
            key = _preview_key(results)
            previews = [_cached_preview(key, stage, row)
                        for stage in range(len(results["stages_performance"])) for row in ("stator", "rotor")]
            return vtk.turbine_data(previews)
        except Exception as error:
            _LOG.exception("Whole-turbine preview failed")
            return vtk.empty_data(f"Could not build the turbine overview: {error}")

    @app.callback(
        Output("mesh_profile", "figure"), Output("mesh_meridional", "figure"), Output("mesh_2d_message", "children"),
        Input("mesh_job", "data"), Input("mesh_status", "data"), Input("result_store", "data"),
        Input("result_input_signature", "data"), Input("mesh_stage", "value"), Input("mesh_row", "value"),
        Input("mesh_pane", "style"), *[Input(*item) for item in _LIVE],
    )
    def update_mesh_sections(handle, status, results, signature, stage, row, pane_style, *live):
        reason = _unavailable(results, signature, input_signature(*live))
        if reason:
            return _empty(reason), _empty(reason), reason
        if not _visible(pane_style):
            raise PreventUpdate
        try:
            if handle and (status or {}).get("state") == "complete":
                profile, meridional = mesh_figures(mesh_jobs.job_directory(handle))
                return profile, meridional, "Generated reference-section cells and physical spanwise mesh."
            preview = _cached_preview(_preview_key(results), stage, row)
            return plot_profile_preview(preview), plot_geometry_meridional(preview), "Geometry preview. Generate this row to display its 2D mesh."
        except Exception as error:
            message = f"Could not display the row sections: {error}"
            return _empty(message), _empty(message), message

    @app.callback(
        Output("geometry_download", "data"), Output("geometry_download_error", "children"),
        Input("geometry_download_button", "n_clicks"), State("result_store", "data"),
        State("result_input_signature", "data"), State("geometry_stage", "value"), State("geometry_row", "value"),
        *[State(*item) for item in _LIVE], prevent_initial_call=True,
    )
    def download_geometry(clicks, results, signature, stage, row, *live):
        if not clicks:
            raise PreventUpdate
        reason = _unavailable(results, signature, input_signature(*live))
        if reason:
            return no_update, reason
        try:
            data = meshing.geometry_zip(results, stage_index=stage, row=row)
            return dcc.send_bytes(data, f"turbodash_stage_{stage + 1}_{row}_geometry.zip", type="application/zip"), ""
        except Exception as error:
            _LOG.exception("Geometry export failed")
            return no_update, f"Geometry export failed: {error}"

    @app.callback(
        Output("mesh_job", "data"), Output("mesh_status", "data"), Output("mesh_poll", "disabled"),
        Input("mesh_generate", "n_clicks"), Input("mesh_poll", "n_intervals"),
        Input("result_store", "data"), Input("result_input_signature", "data"),
        Input("mesh_stage", "value"), Input("mesh_row", "value"),
        *[Input(f"mesh_{key}", "value") for key in _OPTION_KEYS],
        *[Input(*item) for item in _LIVE], State("mesh_job", "data"),
    )
    def manage_mesh(clicks, ticks, results, signature, stage, row, *values):
        options = _options(values[:len(_OPTION_KEYS)])
        live = values[len(_OPTION_KEYS):-1]
        reason = _unavailable(results, signature, input_signature(*live))
        return advance_mesh_job(ctx.triggered_id, results, stage, row, options, values[-1], reason=reason)

    @app.callback(
        Output("mesh_generate", "disabled"), Output("mesh_availability", "children"),
        Output("mesh_status_text", "children"), Output("mesh_progress", "value"),
        Output("mesh_summary", "children"), Output("mesh_download_button", "disabled"),
        Input("mesh_status", "data"), Input("result_store", "data"), Input("result_input_signature", "data"),
        *[Input(*item) for item in _LIVE],
    )
    def display_status(status, results, signature, *live):
        reason = _unavailable(results, signature, input_signature(*live))
        status = status or {}
        state = status.get("state", "idle")
        summary = status.get("summary", {})
        summary_text = _summary_text(summary, status.get("metrics")) if summary else ""
        fraction = status.get("fraction") or 0
        return bool(reason) or state in ("running", "queued"), reason, status.get("message", ""), 100 * max(0, min(1, fraction)), summary_text, bool(reason) or state != "complete"

    @app.callback(
        Output("mesh_axis", "disabled"), Output("mesh_fraction", "disabled"),
        Output("mesh_layer", "disabled"), Output("mesh_worst_count", "disabled"),
        Output("mesh_threshold", "disabled"), Output("mesh_metric", "disabled"),
        Input("mesh_view", "value"),
    )
    def inspection_controls(mode):
        return mode != "slice", mode != "slice", mode != "layer", mode != "worst", mode != "threshold", mode == "surface"

    @app.callback(
        Output("mesh_threshold", "min"), Output("mesh_threshold", "max"), Output("mesh_threshold", "value"),
        Input("mesh_metric", "value"),
    )
    def threshold_range(metric):
        return (1, None, 10) if metric == "aspect_ratio" else (0, 1, 0.2 if metric == "orthogonality" else 0.8)

    @app.callback(
        Output("mesh-vtk-data", "data"), Output("mesh_inspection_message", "children"),
        Input("mesh_job", "data"), Input("mesh_status", "data"),
        Input("mesh_view", "value"), Input("mesh_metric", "value"), Input("mesh_axis", "value"),
        Input("mesh_fraction", "value"), Input("mesh_layer", "value"),
        Input("mesh_worst_count", "value"), Input("mesh_threshold", "value"),
        Input("mesh_pane", "style"),
    )
    def inspect_mesh(handle, status, mode, metric, axis, fraction, layer, count, threshold, pane_style):
        if not handle or (status or {}).get("state") != "complete":
            return vtk.empty_data("Generate a mesh to inspect it."), ""
        if not _visible(pane_style):
            raise PreventUpdate
        try:
            options = {"metric": metric}
            if mode == "slice":
                options.update(axis=axis, fraction=float(fraction) / 100)
            elif mode == "layer":
                options["layer"] = layer
            elif mode == "worst":
                options["n"] = count
            elif mode == "threshold":
                options["threshold"] = threshold
            directory = mesh_jobs.job_directory(handle)
            payload = meshing.inspect_mesh(directory, mode=mode, **options)
            context = meshing.inspect_mesh(directory, mode="surface") if mode != "surface" else None
            metadata = status.get("metadata", status.get("manifest", {}).get("metadata", {}))
            viewer = vtk.mesh_data(payload, geometry=context, metadata=metadata)
            details = payload.get("details", [])
            message = f"Selected cells: {payload['selected_count']}. " if "selected_count" in payload else ""
            detail_table = html.Table([
                html.Thead(html.Tr([html.Th("Cell"), html.Th(payload.get("metric_label", "Quality"))])),
                html.Tbody([html.Tr([html.Td(item["cell_id"]), html.Td(f"{item['value']:.5g}")]) for item in details[:10]]),
            ], className="table table-sm mt-2") if details else None
            return viewer, html.Div([html.P(message + payload.get("note", "")), detail_table])
        except Exception as error:
            _LOG.exception("Mesh inspection failed")
            message = f"Could not inspect the mesh: {error}"
            return vtk.empty_data(message), message

    @app.callback(
        Output("mesh_download", "data"), Output("mesh_download_error", "children"),
        Input("mesh_download_button", "n_clicks"), State("mesh_job", "data"),
        State("result_store", "data"), State("result_input_signature", "data"),
        State("mesh_stage", "value"), State("mesh_row", "value"),
        *[State(f"mesh_{key}", "value") for key in _OPTION_KEYS], *[State(*item) for item in _LIVE],
        prevent_initial_call=True,
    )
    def download_mesh(clicks, handle, results, signature, stage, row, *values):
        if not clicks:
            raise PreventUpdate
        reason = _unavailable(results, signature, input_signature(*values[len(_OPTION_KEYS):]))
        options = _options(values[:len(_OPTION_KEYS)])
        if reason:
            return no_update, reason
        try:
            if not handle or handle.get("fingerprint") != mesh_jobs.fingerprint(results, stage, row, options):
                return no_update, "The mesh does not match the current settings. Generate it again."
            archive = mesh_jobs.mesh_archive(handle)
            return dcc.send_bytes(archive, f"turbodash_stage_{stage + 1}_{row}_mesh.zip", type="application/zip"), ""
        except Exception as error:
            _LOG.exception("Mesh download failed")
            return no_update, f"Mesh download failed: {error}"
