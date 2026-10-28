"""Build the optimization controls and connect them to background solver runs.

This module creates the Configuration, Design variables, and Constraints tabs,
the Start optimization button, and the expandable convergence terminal. Its
Dash callbacks preserve edits and selections, restore settings from loaded
designs, and collect the current inputs when the user starts an optimization.

Runs are launched and monitored through ``opt_jobs``. Progress and outcomes
update the interface, and successful results are applied to the workbench when
the input design still matches the one used to start the run. Failed runs keep
the previously displayed design available for further editing.
"""

from uuid import uuid4
import time

from dash import ALL, Input, Output, State, ctx, dcc, html, no_update
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

from . import opt_jobs as jobs
from .opt_model import (
    build_settings,
    flat_config,
    nested_config,
    output_catalog,
    variable_catalog,
    variable_groups,
    constraint_groups,
    SLSQP_EXTRA_CONTROLS,
    solver_editor_options,
    extra_solver_options,
    editor_optimizer,
    editor_solver_arguments,
)
from turbodash.optimization import OPTIMIZER_LABELS
from .opt_selection import (
    apply_selection_operations,
    picker_signature,
    picker_tree,
    selector_menu,
)

DEFAULT_OBJECTIVE = "$overall_performance.efficiency_ts_with_recovery"
DEFAULT_CONSTRAINT = "$stages_performance.*.geometry_meanline.*.flaring_angle"


def field(label, component, help_text=None):
    return html.Label([
        html.Span(label, className="td-opt-field-label", title=help_text,
                  **({"tabIndex": 0} if help_text else {})),
        component,
    ], className="td-opt-field")


def panel():
    defaults = solver_editor_options()
    slsqp_defaults = solver_editor_options({"solver": "slsqp"})
    return html.Details(
        [
            html.Summary(
                [
                    "Optimization",
                    html.Span(
                        OPTIMIZER_LABELS["slsqp"],
                        id="opt-solver-badge",
                        className="td-opt-badge",
                    ),
                ]
            ),
            html.Div(
                [
                    dbc.Tabs(
                        id="opt-tabs",
                        active_tab="settings",
                        children=[
                            dbc.Tab(
                                label="Configuration",
                                tab_id="settings",
                                label_class_name="td-opt-help-configuration",
                                children=html.Div(
                                    [
                                        field(
                                            "Objective",
                                            dcc.Dropdown(
                                                id="opt-objective",
                                                value=DEFAULT_OBJECTIVE,
                                                clearable=False,
                                            ),
                                        ),
                                        field(
                                            "Goal",
                                            dcc.RadioItems(
                                                id="opt-direction",
                                                value="maximize",
                                                className="td-segmented",
                                                options=[
                                                    {
                                                        "label": "Maximize",
                                                        "value": "maximize",
                                                    },
                                                    {
                                                        "label": "Minimize",
                                                        "value": "minimize",
                                                    },
                                                ],
                                            ),
                                        ),
                                        field(
                                            "Optimizer",
                                            dcc.Dropdown(
                                                id="opt-solver",
                                                value="slsqp",
                                                clearable=False,
                                                searchable=False,
                                                options=[
                                                    dict(label=label, value=name)
                                                    for name, label in OPTIMIZER_LABELS.items()
                                                ],
                                            ),
                                            "IPM (Pounce): interior-point method for constrained turbine design. "
                                            "SQP (Pounce): sequential quadratic programming with an active-set approach; an alternative when IPM struggles. "
                                            "SLSQP (SciPy, through PySolverView): the default sequential least-squares method, with adjustable first-step and invalid-trial recovery controls. "
                                            "All three use the same feasibility and first-order optimality checks.",
                                        ),
                                        field(
                                            "Iteration limit",
                                            dbc.Input(
                                                id="opt-iterations",
                                                type="number",
                                                value=defaults["maxiter"],
                                                min=1,
                                                max=2000,
                                                step=1,
                                            ),
                                        ),
                                        html.Div(
                                            [
                                                field(
                                                    "Solver tolerance",
                                                    dbc.Input(
                                                        id="opt-tolerance",
                                                        type="number",
                                                        value=defaults["ftol"],
                                                        min=1e-14,
                                                        step="any",
                                                        className="td-scientific",
                                                    ),
                                                    "Native stopping tolerance. Separate feasibility and KKT optimality limits still apply; expert settings let you inspect or change them.",
                                                ),
                                                html.Div(
                                                    [
                                                        field(
                                                            "Feasibility tolerance",
                                                            dbc.Input(
                                                                id="opt-feasibility",
                                                                type="number",
                                                                value=1e-6,
                                                                min=1e-14,
                                                                step="any",
                                                                className="td-scientific",
                                                            ),
                                                            "Maximum allowed normalized constraint and bound violation before a result can be applied.",
                                                        ),
                                                        field(
                                                            "Optimality tolerance",
                                                            dbc.Input(
                                                                id="opt-optimality-tolerance",
                                                                type="number",
                                                                value=defaults[
                                                                    "optimality_tol"
                                                                ],
                                                                min=1e-14,
                                                                step="any",
                                                                className="td-scientific",
                                                            ),
                                                            "Maximum first-order KKT residual in normalized coordinates, including bounds. Passing the complete check can stop Pounce; an earlier native exit also ends the run.",
                                                        ),
                                                    ],
                                                    id="opt-expert-tolerances",
                                                    hidden=True,
                                                    className="td-opt-settings-grid",
                                                ),
                                            ],
                                            className="td-opt-settings-grid td-opt-tolerances",
                                        ),
                                        html.Div(
                                            [
                                                field(
                                                    "First step fraction",
                                                    dbc.Input(
                                                        id="opt-first-step",
                                                        type="number",
                                                        value=slsqp_defaults[
                                                            "first_step_fraction"
                                                        ],
                                                        min=1e-8,
                                                        max=1,
                                                        step="any",
                                                    ),
                                                    "Fraction of the first proposed SLSQP step to try. Default 0.25 starts conservatively, then the growth factor restores larger steps.",
                                                ),
                                                field(
                                                    "Step growth factor",
                                                    dbc.Input(
                                                        id="opt-step-growth-factor",
                                                        type="number",
                                                        value=slsqp_defaults[
                                                            "step_growth_factor"
                                                        ],
                                                        min=1,
                                                        step="any",
                                                        placeholder="Blank disables",
                                                    ),
                                                    "Default 1.5 increases the starting step fraction by 50% each iteration. After recovery, growth restarts from the accepted fraction. A factor of 1 holds it; blank disables gradual restoration.",
                                                ),
                                                field(
                                                    "Step reduction factor",
                                                    dbc.Input(
                                                        id="opt-backtrack-factor",
                                                        type="number",
                                                        value=0.5,
                                                        min=1e-8,
                                                        max=0.99999999,
                                                        step="any",
                                                    ),
                                                    "Multiply an invalid trial step by this factor before retrying. Default 0.5 halves the step. Recovery is always enabled in the app.",
                                                ),
                                                field(
                                                    "Trial retry limit",
                                                    dbc.Input(
                                                        id="opt-max-backtracks",
                                                        type="number",
                                                        value=20,
                                                        min=0,
                                                        step=1,
                                                    ),
                                                    "Maximum reductions of an invalid trial. Exhausting the limit stops the run with the physical failure details.",
                                                ),
                                            ],
                                            id="opt-expert-steps",
                                            hidden=True,
                                            className="td-opt-settings-grid td-opt-expert-steps",
                                        ),
                                        html.Div(
                                            [
                                                field(
                                                    "Barrier strategy",
                                                    dcc.Dropdown(
                                                        id="opt-mu-strategy",
                                                        value="adaptive",
                                                        clearable=False,
                                                        searchable=False,
                                                        options=[
                                                            dict(
                                                                label="Adaptive",
                                                                value="adaptive",
                                                            ),
                                                            dict(
                                                                label="Monotone",
                                                                value="monotone",
                                                            ),
                                                        ],
                                                    ),
                                                    "Adaptive was fastest at the requested accuracy in the turbine comparison. Monotone is an alternative trajectory.",
                                                ),
                                            ],
                                            id="opt-expert-ipm",
                                            hidden=True,
                                            className="td-opt-settings-grid",
                                        ),
                                        html.Div(
                                            dbc.Checkbox(
                                                id="opt-expert-settings",
                                                label="Show expert settings",
                                                value=False,
                                                persistence=True,
                                                persistence_type="local",
                                            ),
                                            className="td-opt-expert-footer",
                                        ),
                                    ],
                                    className="td-opt-tab",
                                ),
                            ),
                            dbc.Tab(
                                label="Design variables",
                                tab_id="variables",
                                label_class_name="td-opt-help-variables",
                                children=html.Div(
                                    [
                                        html.Div(
                                            [
                                                selector_menu("variable"),
                                                selector_menu("variable", groups=True),
                                            ],
                                            className="td-opt-picker-actions",
                                        ),
                                        html.Div(id="opt-variable-rows"),
                                    ],
                                    className="td-opt-tab",
                                ),
                            ),
                            dbc.Tab(
                                label="Constraints",
                                tab_id="constraints",
                                label_class_name="td-opt-help-constraints",
                                children=html.Div(
                                    [
                                        html.Div(
                                            [
                                                selector_menu("constraint"),
                                                selector_menu(
                                                    "constraint", groups=True
                                                ),
                                            ],
                                            className="td-opt-picker-actions",
                                        ),
                                        html.Div(id="opt-constraint-rows"),
                                    ],
                                    className="td-opt-tab",
                                ),
                            ),
                        ],
                    ),
                    *[
                        html.Span(
                            message,
                            id=f"opt-tab-help-{name}",
                            className="visually-hidden",
                            **{"data-opt-help-tab": f"td-opt-help-{name}"},
                        )
                        for name, message in (
                            (
                                "configuration",
                                "Choose what to optimize and whether to maximize or minimize it. Set the iteration limit and solver tolerance; expert settings reveal the separate acceptance tolerances and step controls. Hiding expert settings keeps their values.",
                            ),
                            (
                                "variables",
                                "Select the inputs the optimizer may change, individually or in groups. Set each lower and upper bound. Leave Initial blank to use the current design; out-of-bounds initial values are clipped and logged.",
                            ),
                            (
                                "constraints",
                                "Select the limits the design must satisfy, individually or in groups. Choose a comparison and target. Scale divides the physical residual into a normalized violation; it must be positive, even for a zero target.",
                            ),
                        )
                    ],
                ],
                className="td-opt-body",
            ),
            dcc.Store(id="opt-launching", data=False),
            dcc.Store(id="opt-catalog"),
            dcc.Store(id="opt-editor-state", data={"variables": [], "constraints": []}),
            dcc.Store(id="opt-editor-load"),
            dcc.Store(id="opt-solver-options", data=solver_editor_options()),
            dcc.Store(id="opt-last-solver"),
            dcc.Store(
                id="opt-selection-request", data={"revision": 0, "operations": []}
            ),
            dcc.Store(id="opt-picker-signature"),
            dcc.Store(id="opt-job", storage_type="session"),
            dcc.Store(id="opt-applied-result"),
            dcc.Store(id="opt-progress-snapshot"),
            dcc.Store(id="opt-outcome-state", data="ready"),
            dcc.Interval(id="opt-poll", interval=750),
        ],
        className="td-opt-panel",
        id="opt-panel",
    )


def terminal():
    """Keep the expandable run log and final report in the workbench's dock."""
    return html.Details(
        [
            html.Summary(
                [
                    "Optimization terminal",
                    html.Span(
                        OPTIMIZER_LABELS["slsqp"],
                        id="opt-terminal-solver",
                        className="td-opt-badge",
                    ),
                ]
            ),
            html.Pre(
                "Convergence output and the final optimization report will appear here when you start a run.",
                id="opt-log",
                tabIndex=0,
                **{"aria-label": "Optimization output and final report"},
            ),
        ],
        id="opt-terminal",
        className="td-opt-terminal",
        open=False,
    )


def actions():
    """Keep the run action and its status outside the collapsible editor."""
    return html.Footer([
        html.Div([
            html.Strong("Optimization"),
            html.Span("Ready", id="opt-state-label", className="td-opt-outcome td-opt-outcome-neutral",
                      role="status", **{"aria-live": "polite", "aria-atomic": "true"}),
            html.Div(id="opt-status", role="status", children="Optimization uses your current design.",
                     **{"aria-live": "polite", "aria-atomic": "true"}),
            html.Div(id="opt-progress", className="td-opt-progress", **{"aria-live": "polite"}),
        ], className="td-optimization-status"),
        html.Div([
            dbc.Button("Start optimization", id="opt-start", color="primary", className="td-opt-start", n_clicks=0),
            dbc.Button("Stop", id="opt-stop", color="danger", outline=True, disabled=True, n_clicks=0,
                       title="Stop the running optimization and retain the current design."),
        ], className="td-opt-run-buttons"),
    ], id="opt-actions", className="td-optimization-dock td-opt-state-ready",
        **{"aria-label": "Optimization controls", "aria-busy": "false"})


def recovery_summary(result):
    """Give successful recoveries a compact status outside the solver report."""
    count = result.get("domain_rejections", 0)
    return f" Recovered {count} invalid trial{'s' if count != 1 else ''}." if count else ""


def outcome_presentation(state, launching=False):
    """Present explicit job outcomes without interpreting solver messages."""
    states = {
        "ready": ("Ready", "neutral", "primary"),
        "running": ("Running", "info", "primary"),
        "applying": ("Applying design", "info", "primary"),
        "success": ("Applied successfully", "success", "success"),
        "design_changed": ("Design changed · result not applied", "warning", "warning"),
        "unconfirmed": ("Application unconfirmed", "warning", "warning"),
        "failed": ("Optimization failed", "danger", "danger"),
        "cancelled": ("Stopped by user", "neutral", "primary"),
    }
    state = "running" if launching else state
    if state not in states:
        state = "ready"
    label, tone, color = states[state]
    button = {
        "running": "Optimizing…",
        "applying": "Applying design…",
    }.get(state, "Start optimization")
    return (
        f"td-optimization-dock td-opt-state-{state.replace('_', '-')}",
        "true" if state in ("running", "applying") else "false",
        label,
        f"td-opt-outcome td-opt-outcome-{tone}",
        button,
        color,
    )


def _dropdown_options(catalog):
    return [dict(value=path, label=item["label"]) for path, item in catalog.items()]


def _numeric(kind, row, key, label, value=None, placeholder=None, help_text=None):
    return field(label, dbc.Input(id={"type": f"opt-{kind}-number", "row": row, "field": key},
                                 type="number", value=value, step="any", placeholder=placeholder,
                                 size="sm", debounce=True), help_text)


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
            html.Button("×", id={"type": "opt-remove-variable", "row": key}, n_clicks=0, className="btn btn-link td-opt-remove",
                        title="Remove variable", **{"aria-label": "Remove variable", "data-opt-kind": "variable", "data-opt-row": key}),
        ], className="td-opt-row-head"),
        html.Div([
            _numeric("variable", key, "lower", "Lower", row.get("lower"), help_text="Minimum allowed value, in the quantity's displayed units."),
            _numeric("variable", key, "initial", "Initial", row.get("initial"), f"Current: {choice['value']:g}" if choice else "Current design",
                     help_text="Starting value. Leave blank to use the current design; values outside the bounds are clipped and logged."),
            _numeric("variable", key, "upper", "Upper", row.get("upper"), help_text="Maximum allowed value, greater than the lower bound."),
        ], className="td-opt-bounds"),
    ], className="td-opt-row", key=key)


def constraint_row(row, catalog):
    key, path = row["id"], row.get("path")
    options = _dropdown_options(catalog)
    if path and path not in catalog:
        options.append(dict(value=path, label=f"Unavailable · {path}"))
    return html.Div([
        html.Div([
            dcc.Dropdown(id={"type": "opt-constraint-path", "row": key}, options=options, value=path,
                         searchable=True, clearable=False, placeholder="Choose a calculated quantity…"),
            html.Button("×", id={"type": "opt-remove-constraint", "row": key}, n_clicks=0, className="btn btn-link td-opt-remove",
                        title="Remove constraint", **{"aria-label": "Remove constraint", "data-opt-kind": "constraint", "data-opt-row": key}),
        ], className="td-opt-row-head"),
        html.Div([
            field("Comparison", dcc.Dropdown(id={"type": "opt-comparison", "row": key}, value=row.get("comparison", "<="),
                                            clearable=False, searchable=False,
                                            options=[{"label": label, "value": value} for label, value in (("≤", "<="), ("=", "="), ("≥", ">="))]),
                  "Choose whether the calculated quantity must be below, equal to or above the target."),
            _numeric("constraint", key, "target", "Target", row.get("target", 0), help_text="Required value, in the quantity's displayed units."),
            _numeric("constraint", key, "scale", "Scale", row.get("scale", 1), help_text="Positive physical scale that divides the constraint residual. Use a positive scale even when the target is zero."),
        ], className="td-opt-bounds"),
    ], className="td-opt-row", key=key)


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


def reconcile_editor(rows, previous, catalog, kind, trigger, loaded=None, *,
                     add_count=None, removed=(), edited_fields=None):
    """Preserve live numeric edits while refreshing choices and path defaults."""
    rows = [dict(row) for row in rows]
    if loaded is not None:
        rows = [dict(row) for row in loaded]
    else:
        removed = set(removed)
        if isinstance(trigger, dict) and trigger.get("type") == f"opt-remove-{kind}":
            removed.add(trigger["row"])
        rows = [row for row in rows if row["id"] not in removed]
        if add_count is None:
            add_count = int(trigger == f"opt-add-{kind}")
        for _ in range(add_count):
            chosen = {row.get("path") for row in rows}
            path = next((path for path in catalog if path not in chosen), None)
            if kind == "constraint":
                path = DEFAULT_CONSTRAINT if DEFAULT_CONSTRAINT in catalog else path
            rows.append(dict(id=uuid4().hex, path=path))
    old = {row["id"]: row for row in previous}
    for row in rows:
        choice = catalog.get(row.get("path"), {})
        before = old.get(row["id"], {})
        changed = before.get("path") != row.get("path") or (
            row.get("path") == "rotational_speed" and before.get("label") != choice.get("label"))
        if changed and loaded is None:
            fields = (edited_fields or {}).get(row["id"], set())
            # A path response can still be pending when the user commits new
            # bounds. Keep those later edits, while resetting untouched fields
            # to defaults for the newly selected quantity and its units.
            kept = {key: row.get(key) for key in
                    (("lower", "initial", "upper") if kind == "variable" else ("target", "scale", "comparison"))
                    if before and fields and (key in fields or row.get(key) != before.get(key))}
            if kind == "variable":
                row.update(lower=choice.get("lower"), upper=choice.get("upper"), initial=None)
            else:
                row.update(target=choice.get("target", 0), comparison=choice.get("comparison", "="),
                           scale=choice.get("scale", 1))
            row.update(kept)
        row["label"] = choice.get("label")
    return rows


def register_callbacks(app, live_config, render_results):
    """Bind the editors to this app's existing control and rendering functions."""
    overall_values = State({"scope": "overall", "key": ALL}, "value")
    stage_values = State(
        {"scope": "stage", "stage": ALL, "key": ALL, "elem": "input"}, "value"
    )
    overall_ids = State({"scope": "overall", "key": ALL}, "id")
    stage_ids = State(
        {"scope": "stage", "stage": ALL, "key": ALL, "elem": "input"}, "id"
    )
    var_path = {"type": "opt-variable-path", "row": ALL}
    con_path = {"type": "opt-constraint-path", "row": ALL}
    var_num = {"type": "opt-variable-number", "row": ALL, "field": ALL}
    con_num = {"type": "opt-constraint-number", "row": ALL, "field": ALL}
    comparison = {"type": "opt-comparison", "row": ALL}
    editor_states = [
        State(var_path, "value"),
        State(var_path, "id"),
        State(var_num, "value"),
        State(var_num, "id"),
        State(con_path, "value"),
        State(con_path, "id"),
        State(con_num, "value"),
        State(con_num, "id"),
        State(comparison, "value"),
        State(comparison, "id"),
    ]

    @app.callback(Output("opt-catalog", "data"), Output("opt-objective", "options"),
                  Input("result_store", "data"))
    def choices(results):
        if not results:
            raise PreventUpdate
        config = dict(inputs=results["inputs"])
        variables, outputs = variable_catalog(config), output_catalog(config, results)
        return dict(
            variables=variables, constraints=outputs, fingerprint=jobs.fingerprint(config),
            groups=dict(variable=variable_groups(config, variables),
                        constraint=constraint_groups(config, outputs)),
        ), sorted(
            _dropdown_options({k: v for k, v in outputs.items() if v["objective"]}),
            key=lambda option: option["value"] != DEFAULT_OBJECTIVE,
        )

    app.clientside_callback(
        """function(config, selection, previous) {
            return config ? {config: config, revision: (previous?.revision || 0) + 1,
                selection_revision: selection?.revision || 0} : null;
        }""",
        Output("opt-editor-load", "data"),
        Input("loaded_cfg_store", "data"),
        State("opt-selection-request", "data"),
        State("opt-editor-load", "data"),
    )

    @app.callback(
        Output("opt-variable-picker", "children"), Output("opt-constraint-picker", "children"),
        Output("opt-variable-group-picker", "children"), Output("opt-constraint-group-picker", "children"),
        Output("opt-picker-signature", "data"),
        Input("opt-catalog", "data"), Input("opt-editor-state", "data"),
        State("opt-picker-signature", "data"),
    )
    def render_pickers(catalog, editor, signature):
        if not catalog:
            raise PreventUpdate
        editor = editor or dict(variables=[], constraints=[])
        current = picker_signature(catalog, editor)
        if current == signature:
            raise PreventUpdate
        revision = editor.get("selection_revision", 0)
        groups = catalog.get("groups", {})
        return (
            picker_tree(catalog["variables"], editor["variables"], "variable", revision),
            picker_tree(catalog["constraints"], editor["constraints"], "constraint", revision),
            picker_tree(catalog["variables"], editor["variables"], "variable", revision, groups.get("variable", {})),
            picker_tree(catalog["constraints"], editor["constraints"], "constraint", revision, groups.get("constraint", {})),
            current,
        )

    @app.callback(Output("opt-variable-rows", "children"), Output("opt-constraint-rows", "children"),
                  Output("opt-editor-state", "data"),
                  Input("opt-add-variable", "n_clicks"), Input("opt-add-constraint", "n_clicks"),
                  Input({"type": "opt-remove-variable", "row": ALL}, "n_clicks"),
                  Input({"type": "opt-remove-constraint", "row": ALL}, "n_clicks"),
                  Input(var_path, "value"), Input(con_path, "value"),
                  Input(var_num, "value"), Input(con_num, "value"), Input(comparison, "value"),
                  Input("opt-catalog", "data"), Input("opt-editor-load", "data"),
                  Input("opt-selection-request", "data"),
                  State({"type": "opt-remove-variable", "row": ALL}, "id"),
                  State({"type": "opt-remove-constraint", "row": ALL}, "id"),
                  State("opt-editor-state", "data"), *editor_states)
    def edit_rows(_addv, _addc, _removev, _removec, _vp, _cp, _vn, _cn, _comparisons, catalog, load_transition, selection_request,
                  remove_variable_ids, remove_constraint_ids, previous, *values):
        if not catalog:
            raise PreventUpdate
        trigger = ctx.triggered_id
        if "opt-editor-load" in ctx.triggered_prop_ids.values():
            trigger = "opt-editor-load"
        loaded_config = (load_transition or {}).get("config")
        variables = collect_rows(*values[:4])
        constraints = collect_rows(*values[4:])
        previous = previous or dict(variables=[], constraints=[])
        # Committed numbers and comparisons are Inputs as well as shared job
        # State: a new edit must supersede any older in-flight children rebuild,
        # otherwise a delayed Add/catalog response can restore stale bounds.
        loaded_variables = loaded_constraints = None
        loaded_revision = (load_transition or {}).get("revision")
        load_pending = loaded_config and (trigger == "opt-editor-load" or previous.get("loaded_revision") != loaded_revision)
        if load_pending:
            # Loading also changes turbine/speed type. Derive choices from that
            # configuration immediately, without waiting on downstream plots.
            variables_catalog = variable_catalog(loaded_config)
            constraints_catalog = output_catalog(loaded_config)
            catalog = dict(
                variables=variables_catalog, constraints=constraints_catalog,
                groups=dict(variable=variable_groups(loaded_config, variables_catalog),
                            constraint=constraint_groups(loaded_config, constraints_catalog)),
            )
            settings = loaded_config.get("optimization", {})
            loaded_variables = [dict(id=uuid4().hex, path=spec.get("path", path).removeprefix("$"),
                                     lower=spec.get("lower_value"), upper=spec.get("upper_value"), initial=spec.get("initial_value"))
                                for path, spec in settings.get("design_variables", {}).items()]
            loaded_constraints = [dict(id=uuid4().hex, path=spec["variable"], comparison=spec.get("type", "<="),
                                       target=spec.get("value", 0),
                                       scale=(abs(spec.get("value", 0)) if spec.get("normalize") is True else
                                              1 if spec.get("normalize") in (False, None) else spec["normalize"]))
                                  for spec in settings.get("constraints", [])]
        # The browser journals selections cumulatively so rapid toggles and
        # overlapping replies cannot lose actions. A load replaces earlier
        # selections; only operations after its captured baseline are replayed.
        baseline = (load_transition if load_pending else previous).get("selection_revision", 0)
        operations = sorted(
            (operation for operation in (selection_request or {}).get("operations", [])
             if operation.get("revision", 0) > baseline),
            key=lambda operation: operation["revision"],
        )
        selection_revision = max(
            baseline, (selection_request or {}).get("revision", 0),
        )
        if (
            not load_pending and not operations
            and all(identity in ("opt-add-variable", "opt-add-constraint")
                    for identity in ctx.triggered_prop_ids.values())
        ):
            # Native menu opening changes no editor state or numeric focus.
            raise PreventUpdate
        edited_fields = {}
        for identity in ctx.triggered_prop_ids.values():
            if isinstance(identity, dict) and identity.get("type") in (
                    "opt-variable-number", "opt-constraint-number", "opt-comparison"):
                edited_fields.setdefault(identity["row"], set()).add(identity.get("field", "comparison"))
        variables = reconcile_editor(
            variables, previous["variables"], catalog["variables"], "variable", None,
            loaded_variables, add_count=0,
            removed=[identity["row"] for count, identity in zip(_removev or [], remove_variable_ids or []) if count],
            edited_fields=edited_fields,
        )
        constraints = reconcile_editor(
            constraints, previous["constraints"], catalog["constraints"], "constraint", None,
            loaded_constraints, add_count=0,
            removed=[identity["row"] for count, identity in zip(_removec or [], remove_constraint_ids or []) if count],
            edited_fields=edited_fields,
        )
        groups = catalog.get("groups", {})
        variables = apply_selection_operations(
            variables, catalog["variables"], groups.get("variable", {}), "variable", operations)
        constraints = apply_selection_operations(
            constraints, catalog["constraints"], groups.get("constraint", {}), "constraint", operations)
        # New children can trigger the path inputs once more; stop the cycle.
        if isinstance(trigger, dict):
            if variables == previous["variables"] and constraints == previous["constraints"]:
                raise PreventUpdate
        state = dict(variables=variables, constraints=constraints, loaded_revision=loaded_revision,
                     selection_revision=selection_revision)
        return ([variable_row(row, catalog["variables"]) for row in variables] or html.P("No free inputs yet. Add a variable to begin.", className="td-opt-empty"),
                [constraint_row(row, catalog["constraints"]) for row in constraints] or html.P("No constraints · bounds still apply to every free input.", className="td-opt-empty"), state)

    app.clientside_callback(
        """function(expert, solver) {
            return [!expert, !expert || solver !== 'slsqp', !expert || solver !== 'pounce_ipm'];
        }""",
        Output("opt-expert-tolerances", "hidden"),
        Output("opt-expert-steps", "hidden"),
        Output("opt-expert-ipm", "hidden"),
        Input("opt-expert-settings", "value"),
        Input("opt-solver", "value"),
    )

    @app.callback(
        Output("opt-solver-badge", "children"),
        Output("opt-terminal-solver", "children"),
        Output("opt-last-solver", "data"),
        Input("opt-solver", "value"),
        Input("opt-job", "data"),
        Input("opt-applied-result", "data"),
        State("opt-last-solver", "data"),
    )
    def display_solver(solver, job, applied, previous):
        selected = OPTIMIZER_LABELS.get(solver, OPTIMIZER_LABELS["slsqp"])
        # Keep the terminal tied to the actual run when the next selection changes.
        executed = (
            (job or {}).get("solver") or previous or (applied or {}).get("solver")
        )
        return selected, OPTIMIZER_LABELS.get(executed, selected), executed

    @app.callback(
        Output("opt-objective", "value"),
        Output("opt-direction", "value"),
        Output("opt-tolerance", "value"),
        Output("opt-iterations", "value"),
        Output("opt-feasibility", "value"),
        *[Output(component_id, "value") for _, component_id in SLSQP_EXTRA_CONTROLS],
        Output("opt-solver", "value"),
        Output("opt-mu-strategy", "value"),
        Output("opt-solver-options", "data"),
        Input("loaded_cfg_store", "data"),
    )
    def load_settings(config):
        settings = (config or {}).get("optimization")
        objective = (settings or {}).get("objective_function", {})
        options = solver_editor_options(settings)
        controls = dict(solver_editor_options({"solver": "slsqp"}), **options)
        return (
            objective.get("variable", DEFAULT_OBJECTIVE),
            objective.get("type", "maximize"),
            options["ftol"],
            options["maxiter"],
            (settings or {}).get("feasibility_tolerance", 1e-6),
            *[controls[name] for name, _ in SLSQP_EXTRA_CONTROLS],
            editor_optimizer(settings),
            options.get("mu_strategy", "adaptive"),
            options,
        )

    job_outputs = [
        Output("opt-job", "data"),
        Output("opt-status", "children"),
        Output("opt-log", "children"),
        Output("opt-progress-snapshot", "data"),
        Output("opt-progress", "children"),
        Output("loaded_cfg_store", "data", allow_duplicate=True),
        Output("opt-applied-result", "data"),
        Output("opt-terminal", "open"),
        Output("opt-outcome-state", "data"),
    ]
    job_states = [
        State("opt-job", "data"),
        overall_values,
        stage_values,
        overall_ids,
        stage_ids,
        State("stage_data_store", "data"),
        *editor_states,
        State("opt-objective", "value"),
        State("opt-direction", "value"),
        State("opt-tolerance", "value"),
        State("opt-iterations", "value"),
        State("opt-feasibility", "value"),
        State("result_store", "data"),
        *[State(component_id, "value") for _, component_id in SLSQP_EXTRA_CONTROLS],
        State("opt-solver-options", "data"),
        State("opt-solver", "value"),
        State("opt-mu-strategy", "value"),
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

    @app.callback(*[Output(output.component_id, output.component_property, allow_duplicate=True)
                    for output in job_outputs], Input("opt-stop", "n_clicks"), *job_states,
                  prevent_initial_call=True)
    def stop_job(click, *values):
        handle = values[0]
        if not click or not handle or handle.get("applying"):
            raise PreventUpdate
        jobs.cancel_job(handle, jobs.session_owner())
        return run_or_poll(None, None, *values)

    def run_or_poll(_click, _tick, handle, ov, sv, oi, si, stages, *values):
        owner = jobs.session_owner()
        if ctx.triggered_id == "opt-start":
            if not _click:
                raise PreventUpdate
            try:
                if handle and (
                    handle.get("applying")
                    or jobs.job_status(handle, owner)["state"] == "running"
                ):
                    raise PreventUpdate
                config = live_config(ov, sv, oi, si, stages)
                variables = collect_rows(*values[:4])
                constraints = collect_rows(*values[4:10])
                settings = build_settings(
                    config,
                    variables,
                    constraints,
                    *values[10:15],
                    results=values[15],
                    **editor_solver_arguments(values, offset=16),
                )
                handle = jobs.start_job(config, settings, owner)
                handle = dict(handle, solver=settings.get("solver", "slsqp"))
                return (
                    handle,
                    "Optimizing turbine…",
                    "Starting a fresh optimization run…",
                    None,
                    "Preparing initial design…",
                    no_update,
                    no_update,
                    no_update,
                    "running",
                )
            except PreventUpdate:
                raise
            except Exception as error:
                return (
                    None,
                    f"Could not start: {error}",
                    f"ERROR: {error}",
                    None,
                    "",
                    no_update,
                    no_update,
                    no_update,
                    "failed",
                )
        if not handle:
            raise PreventUpdate
        try:
            if handle.get("applying"):
                current = live_config(ov, sv, oi, si, stages)
                displayed = values[15]
                if (
                    displayed
                    and jobs.fingerprint(displayed) == handle["target"]
                    and jobs.fingerprint(current) == handle["target"]
                ):
                    return (
                        None,
                        handle["message"],
                        no_update,
                        no_update,
                        no_update,
                        no_update,
                        no_update,
                        no_update,
                        "success",
                    )
                if time.time() - handle["applied_at"] > 45:
                    # The inspector remains editable while jobs run. A user may
                    # have changed the design after the optimized values arrived;
                    # never overwrite those edits with the pre-run configuration.
                    return (
                        None,
                        "Optimization application was not confirmed. Current inputs retained.",
                        no_update,
                        no_update,
                        "",
                        no_update,
                        no_update,
                        no_update,
                        "unconfirmed",
                    )
                raise PreventUpdate
            status, log, progress, result = jobs.read_job(handle, owner)
            summary = (
                f"Update {progress['iteration']} · objective {progress['objective']:.7g} · violation {progress['max_constraint_violation']:.3g}"
                if progress
                else "Preparing initial design…"
            )
            if status["state"] == "cancelled" and not progress:
                summary = ""
            if status["state"] == "running":
                return (
                    no_update,
                    "Optimizing turbine…",
                    log,
                    progress,
                    summary,
                    no_update,
                    no_update,
                    no_update,
                    "running",
                )
            config, applied = no_update, no_update
            message = status["message"]
            outcome = "cancelled" if status["state"] == "cancelled" else "failed"
            if result and result["success"]:
                current = live_config(ov, sv, oi, si, stages)
                if jobs.fingerprint(current) != result["fingerprint"]:
                    message = "Design changed during the run. Result was not applied; start again from the current inputs."
                    outcome = "design_changed"
                else:
                    # Validate all plots/tables before publishing new inputs.
                    render_results(result["turbine_results"], None)
                    config, applied = nested_config(result["config"]), result
                    message = f"Optimization applied · objective {result['objective']:.7g}. {result['message']}"
                    recovery = recovery_summary(result)
                    if recovery:
                        message = message.rstrip(".") + "." + recovery
                    pending = dict(
                        handle,
                        applying=True,
                        target=jobs.fingerprint(result["config"]),
                        applied_at=time.time(),
                        message=message,
                    )
                    return (
                        pending,
                        "Applying optimized design…",
                        log,
                        progress,
                        summary,
                        config,
                        applied,
                        no_update,
                        "applying",
                    )
            elif result:
                message = f"Result not applied · {result['message']}"
            return (
                None,
                message,
                log,
                progress,
                summary,
                config,
                applied,
                no_update,
                outcome,
            )
        except PreventUpdate:
            raise
        except Exception as error:
            return (
                None,
                f"Optimization stopped: {error}",
                f"ERROR: {error}",
                no_update,
                "",
                no_update,
                no_update,
                no_update,
                "failed",
            )

    @app.callback(
        Output("opt-actions", "className"), Output("opt-actions", "aria-busy"),
        Output("opt-state-label", "children"), Output("opt-state-label", "className"),
        Output("opt-start", "children"), Output("opt-start", "color"),
        Input("opt-outcome-state", "data"), Input("opt-launching", "data"),
    )
    def present_outcome(state, launching):
        return outcome_presentation(state, launching)

    app.clientside_callback(
        """function(job, launching, style) {
            return [!!job || !!launching, !job || !!job.applying || !!launching,
                    job ? 'td-opt-panel td-section td-opt-busy' : 'td-opt-panel td-section',
                    Object.assign({}, style, {pointerEvents: 'auto'}), !!job || !!launching];
        }""",
        Output("opt-start", "disabled"),
        Output("opt-stop", "disabled"),
        Output("opt-panel", "className"),
        Output("design-controls", "style"),
        Output("opt-solver", "disabled"),
        Input("opt-job", "data"),
        Input("opt-launching", "data"),
        State("design-controls", "style"),
    )
