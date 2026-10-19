"""Dash optimization editors and job delivery, separate from numerical solving."""

from uuid import uuid4
import time

from dash import ALL, Input, Output, State, ctx, dcc, html, no_update
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

from . import jobs
from .model import (
    build_settings, flat_config, nested_config, output_catalog, variable_catalog,
)


DEFAULT_OBJECTIVE = "$overall_performance.efficiency_ts"
DEFAULT_CONSTRAINT = "$stages_performance.*.geometry_meanline.*.flaring_angle"


def field(label, component):
    return html.Label([html.Span(label), component], className="td-opt-field")


def panel():
    return html.Details([
        html.Summary(["Optimization", html.Span("SLSQP", className="td-opt-badge")]),
        html.Div([
            html.P("Choose the inputs to vary and the limits your design must meet.", className="td-opt-hint"),
            dbc.Tabs(id="opt-tabs", active_tab="settings", children=[
                dbc.Tab(label="Settings", tab_id="settings", children=html.Div([
                    field("Objective", dcc.Dropdown(id="opt-objective", value=DEFAULT_OBJECTIVE, clearable=False)),
                    field("Goal", dbc.RadioItems(id="opt-direction", value="maximize", inline=True,
                                                options=[{"label": "Maximize", "value": "maximize"},
                                                         {"label": "Minimize", "value": "minimize"}])),
                    html.Div([
                        field("Solver tolerance", dbc.Input(id="opt-tolerance", type="number", value=1e-8, min=1e-14, step="any")),
                        field("Iteration limit", dbc.Input(id="opt-iterations", type="number", value=150, min=1, max=2000, step=1)),
                        field("Feasibility tolerance", dbc.Input(id="opt-feasibility", type="number", value=1e-6, min=1e-14, step="any")),
                    ], className="td-opt-settings-grid"),
                ], className="td-opt-tab")),
                dbc.Tab(label="Design variables", tab_id="variables", children=html.Div([
                    html.P("Only selected inputs are free. A blank initial value uses the current design; values outside bounds are clipped and logged.", className="td-opt-hint"),
                    html.Div(id="opt-variable-rows"),
                    dbc.Button("+ Add variable", id="opt-add-variable", outline=True, color="primary", size="sm", n_clicks=0),
                ], className="td-opt-tab")),
                dbc.Tab(label="Constraints", tab_id="constraints", children=html.Div([
                    html.P("Scale divides the residual into dimensionless units. Use a positive physical scale, even for a zero target.", className="td-opt-hint"),
                    html.Div(id="opt-constraint-rows"),
                    dbc.Button("+ Add constraint", id="opt-add-constraint", outline=True, color="primary", size="sm", n_clicks=0),
                ], className="td-opt-tab")),
            ]),
            dbc.Button("Start optimization", id="opt-start", color="primary", className="td-opt-start", n_clicks=0),
            html.Div(id="opt-status", role="status", children="Ready · optimization uses your current design."),
            html.Div(id="opt-progress", className="td-opt-progress", **{"aria-live": "polite"}),
        ], className="td-opt-body"),
        dcc.Store(id="opt-launching", data=False), dcc.Store(id="opt-catalog"), dcc.Store(id="opt-editor-state", data={"variables": [], "constraints": []}),
        dcc.Store(id="opt-job", storage_type="session"), dcc.Store(id="opt-applied-result"),
        dcc.Store(id="opt-progress-snapshot"),
        dcc.Interval(id="opt-poll", interval=750),
    ], className="td-opt-panel", id="opt-panel")


def terminal():
    return html.Details([
        html.Summary(["Optimization terminal", html.Span("PySolverView", className="td-opt-badge")]),
        html.Pre("Convergence output will appear here when you start a run.", id="opt-log", tabIndex=0,
                 **{"aria-label": "Optimization convergence output"}),
    ], id="opt-terminal", className="td-opt-terminal")


def _dropdown_options(catalog):
    return [dict(value=path, label=item["label"]) for path, item in catalog.items()]


def _numeric(kind, row, key, label, value=None, placeholder=None):
    return field(label, dbc.Input(id={"type": f"opt-{kind}-number", "row": row, "field": key},
                                 type="number", value=value, step="any", placeholder=placeholder,
                                 size="sm", debounce=True))


def variable_row(row, catalog):
    key, path = row["id"], row.get("path")
    choice = catalog.get(path, {})
    options = _dropdown_options(catalog)
    if path and path not in catalog:
        options.append(dict(value=path, label=f"Unavailable · {path}"))
    return html.Div([
        html.Div([
            dcc.Dropdown(id={"type": "opt-variable-path", "row": key}, options=options, value=path,
                         searchable=True, clearable=False, placeholder="Choose a model input…"),
            html.Button("×", id={"type": "opt-remove-variable", "row": key}, n_clicks=0, className="btn btn-link",
                        title="Remove variable", **{"aria-label": "Remove variable"}),
        ], className="td-opt-row-head"),
        html.Div([
            _numeric("variable", key, "lower", "Lower", row.get("lower")),
            _numeric("variable", key, "initial", "Initial", row.get("initial"), f"Current: {choice['value']:g}" if choice else "Current design"),
            _numeric("variable", key, "upper", "Upper", row.get("upper")),
        ], className="td-opt-bounds"),
    ], className="td-opt-row")


def constraint_row(row, catalog):
    key, path = row["id"], row.get("path")
    options = _dropdown_options(catalog)
    if path and path not in catalog:
        options.append(dict(value=path, label=f"Unavailable · {path}"))
    return html.Div([
        html.Div([
            dcc.Dropdown(id={"type": "opt-constraint-path", "row": key}, options=options, value=path,
                         searchable=True, clearable=False, placeholder="Choose a calculated quantity…"),
            html.Button("×", id={"type": "opt-remove-constraint", "row": key}, n_clicks=0, className="btn btn-link",
                        title="Remove constraint", **{"aria-label": "Remove constraint"}),
        ], className="td-opt-row-head"),
        html.Div([
            field("Comparison", dcc.Dropdown(id={"type": "opt-comparison", "row": key}, value=row.get("comparison", "<="),
                                            clearable=False, searchable=False,
                                            options=[{"label": label, "value": value} for label, value in (("≤", "<="), ("=", "="), ("≥", ">="))])),
            _numeric("constraint", key, "target", "Target", row.get("target", 0)),
            _numeric("constraint", key, "scale", "Scale", row.get("scale", 1)),
        ], className="td-opt-bounds"),
    ], className="td-opt-row")


def collect_rows(path_values, path_ids, number_values, number_ids, comparisons=None, comparison_ids=None):
    rows = {identity["row"]: dict(id=identity["row"], path=value)
            for identity, value in zip(path_ids or [], path_values or [])}
    for identity, value in zip(number_ids or [], number_values or []):
        if identity["row"] in rows:
            rows[identity["row"]][identity["field"]] = value
    for identity, value in zip(comparison_ids or [], comparisons or []):
        if identity["row"] in rows:
            rows[identity["row"]]["comparison"] = value
    return list(rows.values())


def reconcile_editor(rows, previous, catalog, kind, trigger, loaded=None):
    """Preserve live numeric edits while refreshing choices and path defaults."""
    if loaded is not None:
        rows = loaded
    elif trigger == f"opt-add-{kind}":
        chosen = {row.get("path") for row in rows}
        path = next((path for path in catalog if path not in chosen), None)
        if kind == "constraint":
            path = DEFAULT_CONSTRAINT if DEFAULT_CONSTRAINT in catalog else path
        rows.append(dict(id=uuid4().hex, path=path))
    elif isinstance(trigger, dict) and trigger.get("type") == f"opt-remove-{kind}":
        rows = [row for row in rows if row["id"] != trigger["row"]]
    old = {row["id"]: row for row in previous}
    for row in rows:
        choice = catalog.get(row.get("path"), {})
        before = old.get(row["id"], {})
        changed = before.get("path") != row.get("path") or (
            row.get("path") == "rotational_speed" and before.get("label") != choice.get("label"))
        if changed and loaded is None:
            if kind == "variable":
                row.update(lower=choice.get("lower"), upper=choice.get("upper"), initial=None)
            else:
                row.update(target=0, comparison="=", scale=choice.get("scale", 1))
        row["label"] = choice.get("label")
    return rows


def register_callbacks(app, live_config, render_results):
    """Bind the editors to this app's existing control and rendering functions."""
    overall_values = State({"scope": "overall", "key": ALL}, "value")
    stage_values = State({"scope": "stage", "stage": ALL, "key": ALL, "elem": "input"}, "value")
    overall_ids = State({"scope": "overall", "key": ALL}, "id")
    stage_ids = State({"scope": "stage", "stage": ALL, "key": ALL, "elem": "input"}, "id")
    var_path = {"type": "opt-variable-path", "row": ALL}
    con_path = {"type": "opt-constraint-path", "row": ALL}
    var_num = {"type": "opt-variable-number", "row": ALL, "field": ALL}
    con_num = {"type": "opt-constraint-number", "row": ALL, "field": ALL}
    comparison = {"type": "opt-comparison", "row": ALL}
    editor_states = [State(var_path, "value"), State(var_path, "id"), State(var_num, "value"), State(var_num, "id"),
                     State(con_path, "value"), State(con_path, "id"), State(con_num, "value"), State(con_num, "id"),
                     State(comparison, "value"), State(comparison, "id")]

    @app.callback(Output("opt-catalog", "data"), Output("opt-objective", "options"),
                  Input("result_store", "data"))
    def choices(results):
        if not results:
            raise PreventUpdate
        config = dict(inputs=results["inputs"])
        variables, outputs = variable_catalog(config), output_catalog(config, results)
        return dict(variables=variables, constraints=outputs, fingerprint=jobs.fingerprint(config)), _dropdown_options({k: v for k, v in outputs.items() if v["objective"]})

    @app.callback(Output("opt-variable-rows", "children"), Output("opt-constraint-rows", "children"),
                  Output("opt-editor-state", "data"),
                  Input("opt-add-variable", "n_clicks"), Input("opt-add-constraint", "n_clicks"),
                  Input({"type": "opt-remove-variable", "row": ALL}, "n_clicks"),
                  Input({"type": "opt-remove-constraint", "row": ALL}, "n_clicks"),
                  Input(var_path, "value"), Input(con_path, "value"),
                  Input("opt-catalog", "data"), Input("loaded_cfg_store", "data"),
                  State("opt-editor-state", "data"), *editor_states)
    def edit_rows(_addv, _addc, _removev, _removec, _vp, _cp, catalog, loaded_config, previous, *values):
        if not catalog:
            raise PreventUpdate
        trigger = ctx.triggered_id
        # A children replacement can report synthetic pattern events alongside
        # the new catalog. Always process the configuration/catalog update.
        for preferred in ("loaded_cfg_store", "opt-catalog", "opt-add-variable", "opt-add-constraint"):
            if preferred in ctx.triggered_prop_ids.values():
                trigger = preferred
                break
        if isinstance(trigger, dict) and trigger["type"].startswith("opt-remove") and not ctx.triggered[0]["value"]:
            raise PreventUpdate
        variables = collect_rows(*values[:4])
        constraints = collect_rows(*values[4:])
        previous = previous or dict(variables=[], constraints=[])
        loaded_variables = loaded_constraints = None
        loaded_signature = jobs.fingerprint(loaded_config) if loaded_config else None
        load_pending = loaded_config and (trigger == "loaded_cfg_store" or previous.get("loaded_signature") != loaded_signature)
        if load_pending:
            # Loading also changes turbine/speed type. Derive choices from that
            # configuration immediately, without waiting on downstream plots.
            catalog = dict(variables=variable_catalog(loaded_config), constraints=output_catalog(loaded_config))
            settings = loaded_config.get("optimization", {})
            loaded_variables = [dict(id=uuid4().hex, path=spec.get("path", path).removeprefix("$"),
                                     lower=spec.get("lower_value"), upper=spec.get("upper_value"), initial=spec.get("initial_value"))
                                for path, spec in settings.get("design_variables", {}).items()]
            loaded_constraints = [dict(id=uuid4().hex, path=spec["variable"], comparison=spec.get("type", "<="),
                                       target=spec.get("value", 0),
                                       scale=(abs(spec.get("value", 0)) if spec.get("normalize") is True else
                                              1 if spec.get("normalize") in (False, None) else spec["normalize"]))
                                  for spec in settings.get("constraints", [])]
        variables = reconcile_editor(variables, previous["variables"], catalog["variables"], "variable", trigger, loaded_variables)
        constraints = reconcile_editor(constraints, previous["constraints"], catalog["constraints"], "constraint", trigger, loaded_constraints)
        # New children can trigger the path inputs once more; stop the cycle.
        if isinstance(trigger, dict) and trigger["type"].endswith("-path"):
            if variables == previous["variables"] and constraints == previous["constraints"]:
                raise PreventUpdate
        state = dict(variables=variables, constraints=constraints, loaded_signature=loaded_signature)
        return ([variable_row(row, catalog["variables"]) for row in variables] or html.P("No free inputs yet. Add a variable to begin.", className="td-opt-empty"),
                [constraint_row(row, catalog["constraints"]) for row in constraints] or html.P("No constraints · bounds still apply to every free input.", className="td-opt-empty"), state)

    @app.callback(Output("opt-objective", "value"), Output("opt-direction", "value"),
                  Output("opt-tolerance", "value"), Output("opt-iterations", "value"), Output("opt-feasibility", "value"),
                  Input("loaded_cfg_store", "data"))
    def load_settings(config):
        settings = (config or {}).get("optimization", {})
        objective = settings.get("objective_function", {})
        options = settings.get("solver_options", {})
        return (objective.get("variable", DEFAULT_OBJECTIVE), objective.get("type", "maximize"),
                options.get("tolerance", 1e-8), options.get("max_iterations", 150), settings.get("feasibility_tolerance", 1e-6))

    job_outputs = [
        Output("opt-job", "data"), Output("opt-status", "children"), Output("opt-log", "children"),
        Output("opt-progress-snapshot", "data"), Output("opt-progress", "children"),
        Output("loaded_cfg_store", "data", allow_duplicate=True), Output("opt-applied-result", "data"),
        Output("opt-terminal", "open"),
    ]
    job_states = [
        State("opt-job", "data"), overall_values, stage_values, overall_ids, stage_ids,
        State("stage_data_store", "data"), *editor_states,
        State("opt-objective", "value"), State("opt-direction", "value"),
        State("opt-tolerance", "value"), State("opt-iterations", "value"), State("opt-feasibility", "value"),
        State("result_store", "data"),
    ]

    @app.callback(*job_outputs, Input("opt-poll", "n_intervals"), *job_states, prevent_initial_call=True)
    def poll_job(tick, *values):
        return run_or_poll(None, tick, *values)

    @app.callback(*[Output(output.component_id, output.component_property, allow_duplicate=True)
                    for output in job_outputs], Input("opt-start", "n_clicks"), *job_states,
                  running=[(Output("opt-launching", "data"), True, False)], prevent_initial_call=True)
    def launch_job(click, *values):
        # Separate callbacks keep an interval response from superseding a
        # slower launch response. The numerical job itself remains isolated.
        return run_or_poll(click, None, *values)

    def run_or_poll(_click, _tick, handle, ov, sv, oi, si, stages, *values):
        owner = jobs.session_owner()
        if ctx.triggered_id == "opt-start":
            if not _click:
                raise PreventUpdate
            try:
                if handle and (handle.get("applying") or jobs.job_status(handle, owner)["state"] == "running"):
                    raise PreventUpdate
                config = live_config(ov, sv, oi, si, stages)
                variables = collect_rows(*values[:4])
                constraints = collect_rows(*values[4:10])
                settings = build_settings(config, variables, constraints, *values[10:15], results=values[15])
                handle = jobs.start_job(config, settings, owner)
                return handle, "Optimizing turbine…", "Starting a fresh optimization run…", None, "Preparing initial design…", no_update, no_update, True
            except PreventUpdate:
                raise
            except Exception as error:
                return None, f"Could not start: {error}", f"ERROR: {error}", None, "", no_update, no_update, True
        if not handle:
            raise PreventUpdate
        try:
            if handle.get("applying"):
                current = live_config(ov, sv, oi, si, stages)
                displayed = values[15]
                if (displayed and jobs.fingerprint(displayed) == handle["target"]
                        and jobs.fingerprint(current) == handle["target"]):
                    return None, handle["message"], no_update, no_update, no_update, no_update, no_update, no_update
                if time.time() - handle["applied_at"] > 45:
                    source = jobs.source_config(handle, owner)
                    return None, "Could not apply optimized controls. Previous inputs restored.", no_update, no_update, "", nested_config(source), None, True
                raise PreventUpdate
            status, log, progress, result = jobs.read_job(handle, owner)
            summary = (f"Update {progress['iteration']} · objective {progress['objective']:.7g} · violation {progress['max_constraint_violation']:.3g}" if progress else "Preparing initial design…")
            if status["state"] == "running":
                return no_update, "Optimizing turbine…", log, progress, summary, no_update, no_update, no_update
            config, applied = no_update, no_update
            message = status["message"]
            if result and result["success"]:
                current = live_config(ov, sv, oi, si, stages)
                if jobs.fingerprint(current) != result["fingerprint"]:
                    message = "Design changed during the run. Result was not applied; start again from the current inputs."
                else:
                    # Validate all plots/tables before publishing new inputs.
                    render_results(result["turbine_results"], None)
                    config, applied = nested_config(result["config"]), result
                    message = f"Optimization applied · objective {result['objective']:.7g}. {result['message']}"
                    pending = dict(handle, applying=True, target=jobs.fingerprint(result["config"]),
                                   applied_at=time.time(), message=message)
                    return pending, "Applying optimized design…", log, progress, summary, config, applied, no_update
            elif result:
                message = f"Result not applied · {result['message']}"
            return None, message, log, progress, summary, config, applied, no_update
        except PreventUpdate:
            raise
        except Exception as error:
            return None, f"Optimization stopped: {error}", f"ERROR: {error}", no_update, "", no_update, no_update, True

    app.clientside_callback(
        """function(job, launching, style) {
            return [!!job || !!launching, job ? 'td-opt-panel td-opt-busy' : 'td-opt-panel',
                    Object.assign({}, style, {pointerEvents: job ? 'none' : 'auto'})];
        }""",
        Output("opt-start", "disabled"), Output("opt-panel", "className"), Output("design-controls", "style"),
        Input("opt-job", "data"), Input("opt-launching", "data"), State("design-controls", "style"))
