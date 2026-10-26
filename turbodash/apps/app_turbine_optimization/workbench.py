"""Build the turbine workbench's page layout and reusable display components.

This module arranges the header, navigation, numbered input sections, stage
and blade-row editors, plots, performance reports, and geometry tools. It also
formats tables and headline results and places the controls and terminal
provided by ``opt_panel`` within the page.

Input controls remain mounted when sections are collapsed or the editing
context changes, preserving their values. Calculation and interaction
callbacks are registered by ``app`` and ``opt_panel``.
"""

import base64
from decimal import Decimal, ROUND_HALF_UP
import json
import math
from pathlib import Path

from dash import dcc, html, dash_table
import dash_bootstrap_components as dbc

from turbodash.apps.shared.fluid_inputs import (
    fluid_options,
    property_pair_options,
    pair_fields,
)
from turbodash.apps.shared.components.meshing import mesh_panel
from turbodash.apps.shared.components.blade_modeller import geometry_panel
from turbodash.apps.shared.loss_controls import exit_mach_control
from .opt_panel import (
    actions as optimization_actions,
    panel as optimization_panel,
    terminal as optimization_terminal,
)


EXHAUST_RECOVERY_HELP = (
    "0% recovery gives the total-to-static efficiency; 100% gives the "
    "total-to-total efficiency. A value in between represents partial recovery "
    "of the final rotor's exit kinetic energy in a downstream diffuser, "
    "providing a system-efficiency estimate between these limits. "
    "This assumption changes only the efficiency with exhaust recovery; "
    "it does not calculate a diffuser outlet state or change shaft power."
)


def identity_id(identity):
    return (
        json.dumps(identity, sort_keys=True, separators=(",", ":"))
        if isinstance(identity, dict)
        else identity
    )


def field(label, control, *, aliases="", hint=None, label_id=None):
    props = {"htmlFor": identity_id(control.id)}
    if label_id:
        props["id"] = label_id
    return html.Div(
        [
            html.Div(
                [
                    html.Label(label, **props),
                    (
                        html.Span(
                            "?",
                            title=hint,
                            tabIndex=0,
                            className="td-help",
                            **{"aria-label": hint},
                        )
                        if hint
                        else None
                    ),
                ],
                className="td-field-label",
            ),
            html.Div(control, className="td-field-control", title=hint),
        ],
        className="td-field",
        **{
            "data-parameter-label": plain_label(label),
            "data-parameter-aliases": aliases,
        },
    )


def plain_label(value):
    if isinstance(value, (list, tuple)):
        return "".join(plain_label(item) for item in value)
    if hasattr(value, "children"):
        return plain_label(value.children)
    return str(value)


def number(label, key, value, *, hint=None, label_id=None):
    return field(
        label,
        dcc.Input(
            id={"scope": "overall", "key": key},
            type="number",
            value=value,
            step="any",
            debounce=True,
        ),
        aliases=key,
        hint=hint,
        label_id=label_id,
    )


def section(number, title, children, key):
    return html.Details(
        [
            html.Summary(
                [
                    html.Span(f"{number:02}", className="td-section-number"),
                    html.Span(title, className="td-section-title"),
                ]
            ),
            html.Div(children, className="td-section-body"),
        ],
        id=f"td-section-{key}",
        className="td-section",
        open=False,
    )


def stage_editor(stages, turbine_type, stage_fields, blade_fields, defaults):
    """All stages/rows are mounted; selection changes visibility only."""
    editors = []
    for index, values in enumerate(stages):

        def stage_field(key, label, lo, hi, _step):
            value = values.get(key, defaults[key])
            if "." in key:
                row, parameter = key.split(".", 1)
                value = values.get(
                    key,
                    values.get("blade_geometry", {}).get(row, {}).get(parameter, value),
                )
            return field(
                label,
                dcc.Input(
                    id={"scope": "stage", "stage": index, "key": key, "elem": "input"},
                    type="number",
                    value=value,
                    min=lo,
                    max=hi,
                    step="any",
                    debounce=True,
                ),
                aliases=key,
            )

        design = [stage_field(*spec) for spec in stage_fields
                  if not spec[0].startswith("axial_spacing_")]
        rows = []
        for row in ("stator", "rotor"):
            controls = [
                stage_field(f"{row}.{key}", label, lo, hi, step)
                for key, label, lo, hi, step in blade_fields
                if (key != "aspect_ratio" or turbine_type != "radial")
                and not key.endswith("wedge_angle")
            ]
            if turbine_type != "radial" and (row == "rotor" or index > 0):
                controls.extend(stage_field(*spec) for spec in stage_fields
                                if spec[0] == f"axial_spacing_before_{row}")
            rows.append(
                html.Div(controls, **{"data-blade-row": row}, hidden=row != "stator")
            )
        editors.append(
            html.Div(
                [
                    html.Details(
                        [
                            html.Summary("Aerodynamic design"),
                            html.Div(design, className="td-subsection-body"),
                        ],
                        className="td-subsection",
                    ),
                    html.Details(
                        [
                            html.Summary("Blade geometry"),
                            html.Div(rows, className="td-subsection-body"),
                        ],
                        className="td-subsection",
                    ),
                ],
                **{
                    "data-stage-index": str(index),
                    "data-stage-name": values.get("name", f"Stage {index + 1}"),
                },
                hidden=index != 0,
            )
        )
    return editors


def plot_card(title, graph_id):
    hint = (
        "Station-derived hub and shroud at physical scale, with illustrative guide streamlines. Zoom to inspect; reset restores the zero radial baseline."
        if graph_id == "meridional_plot"
        else ""
    )
    return html.Section(
        [
            html.H3(title, title=hint, tabIndex=0 if hint else -1),
            dcc.Graph(
                id=graph_id,
                responsive=True,
                style={"height": "360px", "minHeight": 0},
                config={"responsive": True, "displaylogo": False},
            ),
        ],
        className="td-plot-card",
    )


def triangle_card(graph_id, figure):
    return html.Section(
        dcc.Graph(
            id=graph_id,
            figure=figure,
            responsive=True,
            style={"height": "340px", "minHeight": 0},
            config={"responsive": True, "displaylogo": False},
        ),
        className="td-plot-card",
    )


def make_data_table(columns, data, first_col_left=True):
    first = columns[0]["id"] if columns else None
    return dash_table.DataTable(
        columns=[
            {
                "name": c["name"],
                "id": c["id"],
                "type": "text" if c["id"] == first else "numeric",
            }
            for c in columns
        ],
        data=data,
        sort_action="native",
        export_format="csv",
        export_headers="display",
        style_table={"overflowX": "auto"},
        style_cell={
            "fontFamily": "inherit",
            "fontSize": "13px",
            "padding": "11px 14px",
            "textAlign": "right",
            "whiteSpace": "nowrap",
            "backgroundColor": "var(--td-surface)",
            "color": "var(--td-text)",
            "border": "none",
            "borderBottom": "1px solid var(--td-border)",
        },
        style_header={
            "fontWeight": "600",
            "backgroundColor": "var(--td-surface-alt)",
            "color": "var(--td-muted)",
        },
        style_cell_conditional=(
            [{"if": {"column_id": first}, "textAlign": "left", "fontWeight": "500"}]
            if first_col_left and first
            else []
        ),
    )


def format_power(value):
    if value is None or not math.isfinite(float(value)):
        return "—", ""
    amount = Decimal(str(value))
    scale = 0 if abs(amount) < 1000 else 1 if abs(amount) < 1000000 else 2
    rounded = (amount / (1000**scale)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    if abs(rounded) >= 1000 and scale < 2:
        scale += 1
        rounded = (amount / (1000**scale)).quantize(
            Decimal("0.01"), rounding=ROUND_HALF_UP
        )
    return f"{rounded:,.2f}", ("W", "kW", "MW")[scale]


def metric_cards(results):
    performance = (results or {}).get("overall_performance", {})

    def formatted(key, multiplier=1, places=2):
        value = performance.get(key)
        return (
            f"{value * multiplier:,.{places}f}"
            if value is not None and math.isfinite(value)
            else "—"
        )

    power, unit = format_power(performance.get("power_actual"))
    specs = [
        ("Total-to-static efficiency", formatted("efficiency_ts", 100), "%"),
        ("Total-to-total efficiency", formatted("efficiency_tt", 100), "%"),
        ("Efficiency with exhaust recovery", formatted("efficiency_ts_with_recovery", 100), "%"),
        ("Shaft speed", formatted("rotational_speed", places=0), "rpm"),
        ("Power", power, unit),
    ]
    return [
        html.Div(
            [
                html.Span(label, className="td-metric-label"),
                html.Div(
                    [html.Strong(value), html.Span(unit, className="td-metric-unit")]
                ),
            ],
            className="td-metric",
        )
        for label, value, unit in specs
    ]


def logo_data_uri(theme):
    path = Path(__file__).parent / "assets" / f"blade-ring-{theme}.svg"
    return "data:image/svg+xml;base64," + base64.b64encode(path.read_bytes()).decode(
        "ascii"
    )


def build_layout(defaults, stage_defaults, loss_models, documentation_url="/docs/"):
    pair = pair_fields(defaults["inlet_property_pair"])
    setup = [
        field(
            "Turbine type",
            dcc.RadioItems(
                id={"scope": "overall", "key": "turbine_type"},
                options=[
                    {"label": "Axial", "value": "axial"},
                    {"label": "Radial", "value": "radial"},
                ],
                value=defaults["turbine_type"],
                className="td-segmented",
            ),
        ),
        html.Div(
            [
                html.Label("Number of stages"),
                html.Div(
                    [
                        html.Button(
                            "−",
                            id="stage_minus",
                            n_clicks=0,
                            **{"aria-label": "Remove last stage"},
                        ),
                        html.Output("1", id="stage_count_label"),
                        html.Button(
                            "+",
                            id="stage_plus",
                            n_clicks=0,
                            **{"aria-label": "Add stage"},
                        ),
                    ],
                    className="td-stepper",
                ),
            ],
            className="td-field",
            **{"data-parameter-label": "Number of stages", "data-parameter-aliases": "stage count n_stages"},
        ),
        field(
            "Loss model",
            dcc.Dropdown(
                id={"scope": "overall", "key": "loss_model"},
                options=[
                    {"label": name.replace("_", " ").title(), "value": name}
                    for name in loss_models
                ],
                value=defaults["loss_model"],
                clearable=False,
            ),
        ),
        exit_mach_control(defaults),
    ]
    operating = [
        field(
            "Working fluid",
            dcc.Dropdown(
                id={"scope": "overall", "key": "fluid_name"},
                options=fluid_options(defaults["fluid_name"]),
                value=defaults["fluid_name"],
                clearable=False,
                searchable=True,
            ),
        ),
        field(
            "Inlet property pair",
            dcc.Dropdown(
                id="td-inlet-pair",
                options=property_pair_options(),
                value=defaults["inlet_property_pair"],
                clearable=False,
            ),
            aliases="thermodynamic state pressure temperature enthalpy entropy quality",
        ),
        dcc.Input(
            id={"scope": "overall", "key": "inlet_property_pair"},
            value=defaults["inlet_property_pair"],
            type="text",
            style={"display": "none"},
        ),
        *[
            number(
                f"Inlet {spec['label'].lower()} [{spec['unit']}]",
                f"inlet_property_{index}",
                defaults[f"inlet_property_{index}"],
                label_id=f"td-inlet-label-{index}",
            )
            for index, spec in enumerate(pair, 1)
        ],
        html.Div(id="td-fluid-error", role="alert", className="td-validation"),
        number("Exit pressure [Pa]", "exit_pressure", defaults["exit_pressure"]),
        number("Mass flow rate [kg/s]", "mass_flow_rate", defaults["mass_flow_rate"]),
        number(
            "Inlet flow angle [deg]", "inlet_flow_angle", defaults["inlet_flow_angle"]
        ),
    ]
    speed = [
        dcc.Input(
            id={"scope": "overall", "key": "rotational_speed_type"},
            type="hidden", value=defaults["rotational_speed_type"],
        ),
        field(
            "Speed definition",
            dcc.RadioItems(
                id="td-speed-mode",
                options=[
                    {"label": "Specific", "value": "specific"},
                    {"label": "Actual", "value": "actual"},
                ],
                value=defaults["rotational_speed_type"],
                className="td-segmented",
            ),
        ),
        number(
            "Specific speed [−]",
            "rotational_speed_value",
            defaults["rotational_speed_value"],
            label_id="td-speed-label",
        ),
        number(
            "Blade velocity ratio (ν)",
            "blade_velocity_ratio",
            defaults["blade_velocity_ratio"],
        ),
    ]
    stage = [
        html.Div(
            [
                field(
                    "Stage",
                    dcc.Dropdown(
                        id="td-stage-selector",
                        options=[{"label": "Stage 1", "value": 0}],
                        value=0,
                        clearable=False,
                    ),
                ),
                field(
                    "Blade row",
                    dcc.RadioItems(
                        id="td-row-selector",
                        options=[
                            {"label": "Stator", "value": "stator"},
                            {"label": "Rotor", "value": "rotor"},
                        ],
                        value="stator",
                        className="td-segmented",
                    ),
                ),
            ],
            className="td-stage-scope",
        ),
        html.Div(id="stages_accordion_container"),
    ]
    opt = optimization_panel()
    # Preserve the existing callback IDs and job ownership, with one disclosure.
    opt.children[0].children = [
        html.Span("06", className="td-section-number"),
        "Optimization",
        html.Span("SLSQP", className="td-opt-badge"),
    ]
    opt.className = "td-opt-panel td-section"
    opt.open = False
    geometry = geometry_panel(workbench=True)
    mesh = mesh_panel(workbench=True)
    geometry_side = next(
        child for child in geometry.children if isinstance(child, html.Aside)
    )
    mesh_side = next(child for child in mesh.children if isinstance(child, html.Aside))
    geometry.children.remove(geometry_side)
    mesh.children.remove(mesh_side)
    for child in mesh_side.children:
        if isinstance(child, html.P):
            child.children = ""
    extras = [
        html.Div(
            geometry_side.children,
            className="td-workspace-controls",
            **{"data-controls-workspace": "geometry"},
        ),
        html.Div(
            mesh_side.children,
            className="td-workspace-controls",
            **{"data-controls-workspace": "mesh"},
        ),
    ]
    controls = html.Aside(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.H2("Meanline inputs", id="td-inputs-title"),
                            html.Button("Collapse all", id="td-collapse-all"),
                        ],
                        className="td-inspector-title",
                    ),
                    html.Label(
                        "Find a parameter",
                        htmlFor="td-parameter-search",
                        className="visually-hidden",
                    ),
                    dcc.Input(
                        id="td-parameter-search",
                        type="search",
                        placeholder="Find a parameter…",
                        debounce=False,
                    ),
                    html.Div(id="td-search-results"),
                ],
                className="td-inspector-head",
            ),
            html.Div(
                [
                    html.Div([
                        section(1, "Turbine setup", setup, "setup"),
                        section(2, "Operating point", operating, "operating"),
                        section(3, "Shaft speed", speed, "speed"),
                        section(4, "Stage design", stage, "stage"),
                        section(5, "Diffuser", [field(
                            "Exhaust kinetic-energy recovery [%]",
                            dcc.Input(
                                id={"scope": "overall", "key": "diffuser_kinetic_energy_recovery_percent"},
                                type="number", min=0, max=100, step="any", debounce=True,
                                value=defaults["diffuser_kinetic_energy_recovery_percent"],
                            ),
                            aliases="diffuser exhaust recovery fraction efficiency",
                            hint=EXHAUST_RECOVERY_HELP,
                        )], "diffuser"),
                        opt,
                    ], **{"data-controls-workspace": "meanline"}),
                    *extras,
                ],
                className="td-inspector-scroll",
            ),
            html.Div(optimization_actions(), **{"data-controls-workspace": "meanline"}),
            html.Div(
                id="td-inspector-resize",
                role="separator",
                tabIndex=0,
                **{
                    "aria-label": "Resize design controls",
                    "aria-orientation": "vertical",
                },
            ),
        ],
        id="design-controls",
        className="td-inspector",
    )
    report_tabs = dbc.Tabs(
        [
            dbc.Tab(
                html.Div(id="perf_table"), label="Performance", tab_id="performance"
            ),
            dbc.Tab(html.Div(id="geom_table"), label="Geometry", tab_id="geometry"),
            dbc.Tab(
                html.Div(id="stations_table"), label="Flow stations", tab_id="flow"
            ),
        ],
        id="td-report-tabs",
        active_tab="performance",
    )
    return html.Div(
        [
            dcc.Download(id="download_yaml"),
            dcc.Store(id="result_store"),
            dcc.Store(id="result_input_signature"),
            dcc.Store(id="loaded_cfg_store"),
            dcc.Store(
                id="stage_data_store", data=[dict(stage_defaults, name="stage_1")]
            ),
            dcc.Store(id="td-stage-count-request"),
            dcc.Store(id="td-stage-model-type", data=defaults["turbine_type"]),
            dcc.Store(id="td-stage-transaction"),
            dcc.Store(id="td-applied-load-revision"),
            dcc.Store(id="td-workspace", data="meanline"),
            dcc.Store(id="td-calculation-state", data={"valid": False}),
            html.Header(
                [
                    html.Button(
                        [
                            html.Img(
                                src=logo_data_uri("light"), className="td-logo-light", alt=""
                            ),
                            html.Img(
                                src=logo_data_uri("dark"), className="td-logo-dark", alt=""
                            ),
                            html.Span("turbodash"),
                        ],
                        className="td-brand",
                        title="Return to Meanline",
                        **{"data-workspace": "meanline"},
                    ),
                    html.Button(
                        "Parameters",
                        id="td-inspector-toggle",
                        **{
                            "aria-expanded": "false",
                            "aria-controls": "design-controls",
                        },
                    ),
                    html.Div(
                        [
                            dcc.Upload(
                                id="load_button",
                                accept=".yaml,.yml",
                                children=html.Button(
                                    [
                                        html.Span("↥", **{"aria-hidden": "true"}),
                                        " Load YAML",
                                    ],
                                    className="td-action",
                                ),
                            ),
                            html.Button(
                                [
                                    html.Span("↧", **{"aria-hidden": "true"}),
                                    " Save design",
                                ],
                                id="save_button",
                                n_clicks=0,
                                className="td-action",
                            ),
                            html.A(
                                "GitHub ↗",
                                href="https://github.com/turbo-sim/turbodash",
                                target="_blank",
                                rel="noopener noreferrer",
                                className="td-project-link",
                            ),
                            html.A(
                                "Documentation ↗",
                                id="td-documentation-link",
                                href=documentation_url,
                                target="_blank",
                                rel="noopener noreferrer",
                                className="td-project-link",
                            ),
                            html.Div(
                                [
                                    html.Button(
                                        "Light",
                                        **{
                                            "data-theme": "light",
                                            "aria-pressed": "false",
                                        },
                                    ),
                                    html.Button(
                                        "Dark",
                                        **{
                                            "data-theme": "dark",
                                            "aria-pressed": "false",
                                        },
                                    ),
                                ],
                                className="td-theme-toggle",
                                role="group",
                                **{"aria-label": "Color scheme"},
                            ),
                        ],
                        className="td-header-actions",
                    ),
                ],
                className="td-header",
            ),
            html.Div(
                [
                    html.Nav(
                        [
                            html.Button(
                                [
                                    html.Span(icon, **{"aria-hidden": "true"}),
                                    html.Span(label),
                                ],
                                **{
                                    "data-workspace": value,
                                    "aria-current": (
                                        "page" if value == "meanline" else "false"
                                    ),
                                },
                            )
                            for icon, label, value in (
                                ("◉", "Meanline", "meanline"),
                                ("◇", "Geometry", "geometry"),
                                ("▦", "Mesh", "mesh"),
                            )
                        ],
                        className="td-workspace-rail",
                        **{"aria-label": "Workspaces"},
                    ),
                    controls,
                    html.Button(
                        className="td-inspector-backdrop",
                        **{"aria-label": "Close design controls"},
                    ),
                    html.Div(
                        [
                            html.Main(
                                [
                                    html.Div(
                                        id="td-load-error",
                                        role="alert",
                                        className="td-validation",
                                    ),
                                    html.Div(
                                        id="td-calculation-status",
                                        role="status",
                                        className="td-calculation-status",
                                    ),
                                    html.Div(
                                        id="td-job-status",
                                        role="status",
                                        className="td-job-status",
                                    ),
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.Div(
                                                        [
                                                            html.Span(
                                                                "TURBINE PERFORMANCE",
                                                                className="td-eyebrow",
                                                            ),
                                                            html.H1("Meanline design"),
                                                        ]
                                                    ),
                                                    html.Span(
                                                        "Waiting for results",
                                                        id="td-results-label",
                                                        className="td-result-badge",
                                                    ),
                                                ],
                                                className="td-view-heading",
                                            ),
                                            html.Div(
                                                metric_cards(None),
                                                id="td-metrics",
                                                className="td-metrics",
                                            ),
                                            html.Details(
                                                [
                                                    html.Summary("Turbine geometry"),
                                                    html.Div(
                                                        [
                                                            plot_card(
                                                                "Meridional channel view", "meridional_plot"
                                                            ),
                                                            plot_card(
                                                                "Blade-to-blade view", "blades_plot"
                                                            ),
                                                        ],
                                                        className="td-plot-pair",
                                                    ),
                                                ],
                                                id="td-geometry-plots",
                                                className="td-results-disclosure",
                                                open=True,
                                            ),
                                            html.Details(
                                                [
                                                    html.Summary("Efficiency and loss distribution"),
                                                    html.Div(
                                                        [
                                                            plot_card(
                                                                "Efficiency vs blade velocity ratio",
                                                                "efficiency_trends_plot",
                                                            ),
                                                            plot_card("Efficiency penalties by loss mechanism", "loss_plot"),
                                                        ],
                                                        className="td-plot-pair",
                                                    ),
                                                ],
                                                id="td-efficiency-plots",
                                                className="td-results-disclosure",
                                                open=True,
                                            ),
                                            html.Details(
                                                [
                                                    html.Summary("Velocity triangles"),
                                                    html.Div(
                                                        [
                                                            html.Span("Display"),
                                                            dcc.RadioItems(
                                                                id="td-triangle-mode",
                                                                options=[
                                                                    {"label": "Mach numbers", "value": "mach"},
                                                                    {"label": "Velocity [m/s]", "value": "velocity"},
                                                                ],
                                                                value="mach",
                                                                className="td-segmented",
                                                                persistence=True,
                                                                persistence_type="session",
                                                            ),
                                                        ],
                                                        className="td-plot-toolbar",
                                                    ),
                                                    html.Div(id="triangles_container"),
                                                ],
                                                id="td-triangle-plots",
                                                className="td-results-disclosure",
                                                open=True,
                                            ),
                                            html.Details(
                                                [html.Summary("Performance reports"),
                                                 html.Div(report_tabs, className="td-report-content")],
                                                id="tables_container",
                                                className="td-report-card td-results-disclosure",
                                                open=True,
                                            ),
                                        ],
                                        id="meanline_pane",
                                        style={"display": "block"},
                                    ),
                                    html.Div(
                                        geometry, id="geometry_pane", style={"display": "none"}
                                    ),
                                    html.Div(mesh, id="mesh_pane", style={"display": "none"}),
                                ],
                                className="td-canvas",
                            ),
                            optimization_terminal(),
                        ],
                        id="td-output-column", className="td-output-column",
                    ),
                ],
                className="td-shell",
            ),
        ],
        id="td-workbench",
        className="td-workbench",
        **{
            "data-stage-index": "0",
            "data-blade-row": "stator",
            "data-workspace": "meanline",
            "data-design-revision": "0",
            "data-preserve-optimization-panel": "false",
        },
    )
