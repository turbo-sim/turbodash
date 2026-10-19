"""Optimization edition of the experimental workspace; existing apps stay independent."""

from turbodash.apps.shared.paths import SHARED_ASSETS_DIRECTORY, DOCUMENTATION_FILE
from turbodash.apps.shared.cli import run_app, run_cli

from pathlib import Path
import os
import base64
import yaml
import numpy as np
from turbodash.core import turbine as core
from turbodash.visualization import plotly_turbine as _plotting
from turbodash.reporting import turbine as reporting
import dash_bootstrap_components as dbc

from datetime import datetime

from dash import (
    Dash,
    dcc,
    html,
    Input,
    Output,
    State,
    ctx,
    ALL,
    no_update,
    set_props,
)
from dash.exceptions import PreventUpdate
from turbodash.apps.shared.assets import register_app_assets
from turbodash.apps.shared.components.turbogrid import register_turbogrid_callbacks
from turbodash.apps.shared.components.meshing import (
    input_signature,
    register_meshing_callbacks,
)
from turbodash.apps.shared.fluid_inputs import (
    pair_fields,
    convert_pair,
    validate_inlet_inputs,
    fluid_options,
)
from turbodash.visualization.vtk_assets import register_vtk_assets
from .panel import register_callbacks as register_optimization_callbacks
from . import jobs as optimization_jobs
from .model import nested_config, build_settings
from .panel import collect_rows
from .workbench import (
    build_layout,
    stage_editor,
    triangle_card,
    make_data_table,
    metric_cards,
)

# =========================
# Global / setup
# =========================
app = Dash(
    __name__,
    title="Turbodash · Optimization",
    assets_folder=str(SHARED_ASSETS_DIRECTORY),
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
)
register_app_assets(
    app, Path(__file__).resolve().parent / "assets", "turbine-optimization"
)
server = app.server
optimization_jobs.configure_sessions(server)


@server.before_request
def initialize_session():
    optimization_jobs.session_owner()


register_vtk_assets(app)
register_turbogrid_callbacks(app)
register_meshing_callbacks(app, precise_inputs=True)


def configure(config_path):
    """Load a YAML startup case before serving this application's layout."""
    import sys
    from turbodash.apps.shared.startup import configure as configure_case
    return configure_case(sys.modules[__name__], config_path, "turbine")


def main(config_path=None, *, open_browser=False):
    if config_path is not None:
        configure(config_path)
    return run_app(app, open_browser=open_browser)


def cli(argv=None):
    return run_cli(main, argv)


# =========================
# Defaults
# =========================
# Overall (turbine-level) defaults.
OVERALL_DEFAULTS = dict(
    loss_model="benner",
    fluid_name="Air",
    turbine_type="axial",
    inlet_property_pair="PT_INPUTS",
    inlet_property_1=5e5,
    inlet_property_2=600.0,
    exit_pressure=1e5,
    mass_flow_rate=10.0,
    inlet_flow_angle=0.0,
    blade_velocity_ratio=0.6,
    rotational_speed_type="specific",
    rotational_speed_value=0.7,
)

BLADE_GEOMETRY_DEFAULTS = {
    "stator.solidity_chord": 1.1209982569657782,
    "rotor.solidity_chord": 1.0493781455466982,
    "stator.aspect_ratio": 2.0,
    "stator.maximum_thickness_to_chord": 0.30,
    "stator.maximum_thickness_location": 0.30,
    "stator.leading_edge_radius_to_max_thickness": 0.50,
    "stator.trailing_edge_thickness_to_opening": 0.07,
    "stator.trailing_edge_wedge_angle": 5.0,
    "stator.leading_edge_wedge_angle": 30.0,
    "rotor.aspect_ratio": 2.0,
    "rotor.maximum_thickness_to_chord": 0.30,
    "rotor.maximum_thickness_location": 0.30,
    "rotor.leading_edge_radius_to_max_thickness": 0.50,
    "rotor.trailing_edge_thickness_to_opening": 0.07,
    "rotor.trailing_edge_wedge_angle": 5.0,
    "rotor.leading_edge_wedge_angle": 30.0,
}

# Per-stage defaults. Two sets: one for axial stages and one for radial
# stages. The radial set is identical to the axial set except that all four
# radius ratios are set equal to 1.0.
STAGE_DEFAULTS_RADIAL = dict(
    work_fraction_split=1.0,
    stator_exit_angle=70.0,
    degree_reaction=0.5,
    meridional_velocity_ratio_12=1.00,
    meridional_velocity_ratio_23=1.00,
    meridional_velocity_ratio_34=1.00,
    radius_ratio_01=0.97,
    radius_ratio_12=0.85,
    radius_ratio_23=0.97,
    radius_ratio_34=0.85,
    **BLADE_GEOMETRY_DEFAULTS,
)

STAGE_DEFAULTS_AXIAL = dict(
    STAGE_DEFAULTS_RADIAL,
    **{"rotor.solidity_chord": 1.137411139662947},
    radius_ratio_01=1.0,
    radius_ratio_12=1.0,
    radius_ratio_23=1.0,
    radius_ratio_34=1.0,
)


def stage_defaults_for(turbine_type):
    """Return the per-stage default dict for the given turbine type."""
    if turbine_type == "radial":
        return dict(STAGE_DEFAULTS_RADIAL)
    return dict(STAGE_DEFAULTS_AXIAL)


# Backwards-compatible alias used throughout the module wherever a generic set
# of stage defaults / the canonical list of stage keys is needed. Points at the
# axial set (the historical default).
STAGE_DEFAULTS = STAGE_DEFAULTS_AXIAL

# Per-stage fields: (cfg_key, label, min, max, step). Labels carry the full
# variable name with symbol and unit. Order = display order. Continuous inputs
# use step="any"; the historical step column remains for shared field metadata.
STAGE_FIELDS = [
    ("work_fraction_split", ["Work fraction split"], 0.001, 10.0, 0.001),
    (
        "stator_exit_angle",
        ["Stator exit angle, α", html.Sub("2"), " [deg]"],
        0.0,
        89.0,
        0.001,
    ),
    ("degree_reaction", ["Degree of reaction, R"], -0.5, 1.0 - 1e-6, 0.001),
    (
        "meridional_velocity_ratio_12",
        ["Meridional velocity ratio, vₘ₁/vₘ₂"],
        0.1,
        2.0,
        0.001,
    ),
    (
        "meridional_velocity_ratio_23",
        ["Meridional velocity ratio, vₘ₂/vₘ₃"],
        0.1,
        2.0,
        0.001,
    ),
    (
        "meridional_velocity_ratio_34",
        ["Meridional velocity ratio, vₘ₃/vₘ₄"],
        0.1,
        2.0,
        0.001,
    ),
    ("radius_ratio_01", ["Radius ratio, r₀/r₁"], 0.10, 1.20, 0.001),
    ("radius_ratio_12", ["Radius ratio, r₁/r₂"], 0.10, 1.20, 0.001),
    ("radius_ratio_23", ["Radius ratio, r₂/r₃"], 0.10, 1.20, 0.001),
    ("radius_ratio_34", ["Radius ratio, r₃/r₄"], 0.10, 1.20, 0.001),
]

# Loss-model dropdown: internal value -> pretty label (Title Case, spaces).
LOSS_MODEL_OPTIONS = ["benner", "kacker_okapuu", "moustapha", "isentropic"]

BLADE_GEOMETRY_FIELDS = [
    ("solidity_chord", "Solidity, true chord / pitch [-]", 0.05, 5.0, 0.001),
    (
        "aspect_ratio",
        "Aspect ratio, height / meridional chord (axial) [-]",
        0.1,
        10.0,
        0.1,
    ),
    (
        "maximum_thickness_to_chord",
        "Maximum thickness / meridional chord [-]",
        0.001,
        1.0,
        0.001,
    ),
    (
        "maximum_thickness_location",
        "Maximum thickness location / chord [-]",
        0.001,
        0.999,
        0.001,
    ),
    (
        "leading_edge_radius_to_max_thickness",
        "Leading-edge radius / maximum thickness [-]",
        0.001,
        1.0,
        0.001,
    ),
    (
        "trailing_edge_thickness_to_opening",
        "Trailing-edge thickness / opening [-]",
        0.001,
        1.0,
        0.001,
    ),
    ("trailing_edge_wedge_angle", "Trailing-edge wedge angle [deg]", 0.0, 89.0, 0.1),
    ("leading_edge_wedge_angle", "Leading-edge wedge angle [deg]", 0.0, 89.0, 0.1),
]


# =========================
# Documentation layout
# =========================
docs_path = str(DOCUMENTATION_FILE)
if os.path.exists(docs_path):
    with open(docs_path, "r", encoding="utf-8") as f:
        theory_md = f.read()
else:
    theory_md = "Documentation not found."

docs_layout = html.Div(
    style={"maxWidth": "1000px", "margin": "auto", "padding": "30px"},
    children=[
        dcc.Markdown(
            theory_md,
            mathjax=True,
            style={
                "whiteSpace": "pre-wrap",
                "padding": "40px",
                "fontFamily": "Segoe UI, Roboto, Helvetica, Arial, sans-serif",
                "fontSize": "16px",
                "lineHeight": "1.6",
                "color": "var(--td-text)",
                "backgroundColor": "var(--td-surface)",
            },
        ),
    ],
)


app.layout = build_layout(
    OVERALL_DEFAULTS, STAGE_DEFAULTS, LOSS_MODEL_OPTIONS, docs_layout
)

app.clientside_callback(
    """function(workspace) {
        return ['meanline', 'geometry', 'mesh'].map(name => ({display: name === workspace ? 'block' : 'none'}));
    }""",
    Output("meanline_pane", "style"),
    Output("geometry_pane", "style"),
    Output("mesh_pane", "style"),
    Input("td-workspace", "data"),
)


@app.callback(
    Output({"scope": "overall", "key": "fluid_name"}, "options"),
    Input({"scope": "overall", "key": "fluid_name"}, "value"),
)
def working_fluid_options(current):
    return fluid_options(current)


app.clientside_callback(
    """function(stage, row) {return [String(stage || 0), row || 'stator'];}""",
    Output("td-workbench", "data-stage-index"),
    Output("td-workbench", "data-blade-row"),
    Input("td-stage-selector", "value"),
    Input("td-row-selector", "value"),
)


@app.callback(
    Output("td-inlet-label-1", "children"),
    Output("td-inlet-label-2", "children"),
    Input({"scope": "overall", "key": "inlet_property_pair"}, "value"),
)
def inlet_labels(pair):
    return [f"{field['label']} [{field['unit']}]" for field in pair_fields(pair)]


@app.callback(
    Output("td-speed-label", "children"),
    Input({"scope": "overall", "key": "rotational_speed_type"}, "value"),
)
def speed_label(mode):
    return "Specific speed [−]" if mode == "specific" else "Shaft speed [rpm]"


@app.callback(Output("td-metrics", "children"), Input("result_store", "data"))
def update_metrics(results):
    return metric_cards(results)


@app.callback(
    Output("td-results-label", "children"), Input("td-calculation-state", "data")
)
def calculation_label(state):
    return (
        "Live results"
        if (state or {}).get("valid")
        else (
            "Last valid results"
            if (state or {}).get("message")
            else "Waiting for results"
        )
    )


app.clientside_callback(
    """function(job, status) {return job ? status : (status && !status.startsWith('Ready') ? status : '');}""",
    Output("td-job-status", "children"),
    Input("opt-job", "data"),
    Input("opt-status", "children"),
)


# =========================================
# Callback: maintain stage_data_store. Fires on +/-, load, or a turbine-type
# switch, never on an ordinary value edit, so editing a parameter does not
# rebuild/collapse the accordion.
# =========================================
@app.callback(
    Output("stage_data_store", "data"),
    Output({"scope": "overall", "key": ALL}, "value"),
    Output("td-inlet-pair", "value"),
    Output("td-fluid-error", "children"),
    Input("stage_plus", "n_clicks"),
    Input("stage_minus", "n_clicks"),
    Input("loaded_cfg_store", "data"),
    Input({"scope": "overall", "key": "turbine_type"}, "value"),
    Input("td-inlet-pair", "value"),
    Input({"scope": "overall", "key": "rotational_speed_type"}, "value"),
    State("stage_data_store", "data"),
    State({"scope": "overall", "key": ALL}, "id"),
    State({"scope": "stage", "stage": ALL, "key": ALL, "elem": "input"}, "value"),
    State({"scope": "stage", "stage": ALL, "key": ALL, "elem": "input"}, "id"),
    State({"scope": "overall", "key": ALL}, "value"),
    State("result_store", "data"),
    prevent_initial_call=True,
)
def update_stage_store(
    n_plus,
    n_minus,
    loaded_cfg,
    turbine_type,
    pair,
    speed_type,
    stages,
    overall_ids,
    live_values,
    live_ids,
    overall_values,
    results,
):
    trigger = ctx.triggered_id
    if "loaded_cfg_store" in ctx.triggered_prop_ids.values():
        trigger = "loaded_cfg_store"
    stages = [
        dict(stage) for stage in (stages or [dict(STAGE_DEFAULTS, name="stage_1")])
    ]
    overall = {
        identity["key"]: value for identity, value in zip(overall_ids, overall_values)
    }

    if trigger == "td-inlet-pair":
        if pair == overall["inlet_property_pair"]:
            raise PreventUpdate
        # Read the last accepted inlet, not a partially typed/invalid draft.
        source = (results or {}).get("inputs", overall)
        message = ""
        try:
            if source.get("fluid_name") != overall.get("fluid_name"):
                raise ValueError(
                    "The working fluid has changed; enter both inlet values."
                )
            first, second = convert_pair(
                overall["fluid_name"],
                source["inlet_property_pair"],
                (source["inlet_property_1"], source["inlet_property_2"]),
                pair,
            )
        except (ValueError, KeyError) as error:
            first = second = None
            message = f"Enter both inlet values for the new pair. {error}"
        changed = {
            "inlet_property_pair": pair,
            "inlet_property_1": first,
            "inlet_property_2": second,
        }
        return (
            no_update,
            [changed.get(identity["key"], no_update) for identity in overall_ids],
            pair,
            message,
        )

    if trigger == {"scope": "overall", "key": "rotational_speed_type"}:
        performance = (results or {}).get("overall_performance", {})
        value = performance.get(
            "specific_speed" if speed_type == "specific" else "rotational_speed"
        )
        if value is None:
            value = (
                OVERALL_DEFAULTS["rotational_speed_value"]
                if speed_type == "specific"
                else None
            )
        return (
            no_update,
            [
                value if identity["key"] == "rotational_speed_value" else no_update
                for identity in overall_ids
            ],
            no_update,
            no_update,
        )

    # Reconcile the store with the user's current live edits before doing
    # anything else, so that add/remove operate on up-to-date values. (Loading
    # and a type switch intentionally discard live edits, so skip there.)
    if trigger in ("stage_plus", "stage_minus"):
        for d, v in zip(live_ids or [], live_values or []):
            si = d["stage"]
            if 0 <= si < len(stages):
                stages[si][d["key"]] = v

    # Switching turbine type: reset ALL stages to the defaults for that type,
    # keeping the current number of stages.
    if trigger == {"scope": "overall", "key": "turbine_type"}:
        if not turbine_type:
            raise PreventUpdate
        defaults = stage_defaults_for(turbine_type)
        return (
            [
                dict(defaults, name=stage.get("name", f"stage_{index + 1}"))
                for index, stage in enumerate(stages)
            ],
            no_update,
            no_update,
            "",
        )

    # Loading a config: replace the whole stage list from the YAML.
    if trigger == "loaded_cfg_store" and loaded_cfg:
        cfg_stages = loaded_cfg.get("inputs", {}).get("stages", [])
        if cfg_stages:
            new_stages = loaded_stages(loaded_cfg)
            for section in ("setup", "operating", "speed", "stage"):
                set_props(f"td-section-{section}", {"open": False})
            set_props("opt-panel", {"open": False})
            set_props("td-stage-selector", {"value": 0})
            set_props(
                "td-workbench", {"data-design-revision": datetime.now().isoformat()}
            )
            return (
                new_stages,
                loaded_overall_values(loaded_cfg, overall_ids),
                loaded_cfg["inputs"].get("inlet_property_pair", "PT_INPUTS"),
                "",
            )
        raise PreventUpdate

    # Add a stage: copy the previous (last) stage's current values as the seed.
    if trigger == "stage_plus":
        seed = dict(stages[-1]) if stages else dict(STAGE_DEFAULTS)
        existing = {stage.get("name") for stage in stages}
        index = len(stages) + 1
        while f"stage_{index}" in existing:
            index += 1
        seed["name"] = f"stage_{index}"
        return stages + [seed], no_update, no_update, no_update

    # Remove a stage: drop the last one, leaving the others unchanged. Never go
    # below one stage.
    if trigger == "stage_minus":
        return (
            (stages[:-1] if len(stages) > 1 else stages),
            no_update,
            no_update,
            no_update,
        )

    raise PreventUpdate


def loaded_stages(config):
    """Flatten the YAML stage mappings without discarding names or precision."""
    stages = []
    for index, stage in enumerate(config["inputs"]["stages"]):
        values = dict(
            stage_defaults_for(config["inputs"].get("turbine_type")),
            name=stage.get("name", f"stage_{index + 1}"),
        )
        values.update({key: stage[key] for key in STAGE_DEFAULTS if key in stage})
        for row in ("stator", "rotor"):
            geometry = stage.get("blade_geometry", {}).get(row, {})
            for key, *_ in BLADE_GEOMETRY_FIELDS:
                values[f"{row}.{key}"] = geometry.get(key, values[f"{row}.{key}"])
        stages.append(values)
    return stages


@app.callback(
    Output("stages_accordion_container", "children"),
    Output("stage_count_label", "children"),
    Output("geometry_rows_container", "children"),
    Input("stage_data_store", "data"),
    State({"scope": "overall", "key": "turbine_type"}, "value"),
)
def render_stage_accordion(stages, turbine_type):
    stages = stages or [dict(STAGE_DEFAULTS, name="stage_1")]
    return (
        stage_editor(
            stages, turbine_type, STAGE_FIELDS, BLADE_GEOMETRY_FIELDS, STAGE_DEFAULTS
        ),
        str(len(stages)),
        [],
    )


@app.callback(
    Output("td-stage-selector", "options"),
    Output("td-stage-selector", "value"),
    Input("stage_data_store", "data"),
    State("td-stage-selector", "value"),
)
def stage_choices(stages, active):
    stages = stages or [dict(name="stage_1")]
    return [
        {"label": stage.get("name", f"Stage {index + 1}"), "value": index}
        for index, stage in enumerate(stages)
    ], min(active or 0, len(stages) - 1)


# =========================================
# Callback: compute + plot. Reads LIVE values from the inputs (overall + each
# stage's number box), so it updates on every edit without touching the store.
# =========================================
@app.callback(
    Output("meridional_plot", "figure"),
    Output("blades_plot", "figure"),
    Output("triangles_container", "children"),
    Output("efficiency_trends_plot", "figure"),
    Output("loss_plot", "figure"),
    Output("perf_table", "children"),
    Output("geom_table", "children"),
    Output("stations_table", "children"),
    Output("result_store", "data"),
    Output("result_input_signature", "data"),
    Input({"scope": "overall", "key": ALL}, "value"),
    Input({"scope": "stage", "stage": ALL, "key": ALL, "elem": "input"}, "value"),
    Input("opt-applied-result", "data"),
    State({"scope": "overall", "key": ALL}, "id"),
    State({"scope": "stage", "stage": ALL, "key": ALL, "elem": "input"}, "id"),
    State("stage_data_store", "data"),
)
def update_turbine(
    overall_values, stage_values, applied, overall_ids, stage_ids, store_stages
):
    if any(
        isinstance(identity, dict)
        and identity.get("key") in ("turbine_type", "rotational_speed_type")
        for identity in ctx.triggered_prop_ids.values()
    ):
        # Wait for the transaction that replaces stage defaults or converts
        # speed units; do not solve an intermediate interpretation of values.
        raise PreventUpdate
    try:
        cfg = live_config(
            overall_values, stage_values, overall_ids, stage_ids, store_stages
        )
        signature = input_signature(
            overall_values, stage_values, overall_ids, stage_ids, store_stages
        )
        if applied and optimization_jobs.fingerprint(
            cfg
        ) == optimization_jobs.fingerprint(applied["config"]):
            out = dict(applied["turbine_results"])
            out["optimization"] = applied["config"]["optimization"]
        elif ctx.triggered_id == "opt-applied-result":
            raise PreventUpdate
        else:
            out = core.compute_turbine_performance(cfg)
        rendered = render_results(out, signature)
    except PreventUpdate:
        raise
    except Exception as error:
        set_props(
            "td-calculation-status",
            {"children": f"Results retained from the last valid design. {error}"},
        )
        set_props(
            "td-calculation-state", {"data": {"valid": False, "message": str(error)}}
        )
        return (no_update,) * 10
    set_props("td-calculation-status", {"children": ""})
    set_props("td-fluid-error", {"children": ""})
    set_props("td-calculation-state", {"data": {"valid": True, "signature": signature}})
    return rendered


def live_config(overall_values, stage_values, overall_ids, stage_ids, store_stages):
    # Overall inputs.
    overall = {d["key"]: v for d, v in zip(overall_ids or [], overall_values or [])}
    validate_inlet_inputs(overall)
    for key, value in overall.items():
        if isinstance(OVERALL_DEFAULTS.get(key), (float, int)):
            if value is None or isinstance(value, bool) or not np.isfinite(value):
                raise ValueError(f"Enter a finite value for {key.replace('_', ' ')}.")
    for key in (
        "exit_pressure",
        "mass_flow_rate",
        "blade_velocity_ratio",
        "rotational_speed_value",
    ):
        if overall[key] <= 0:
            raise ValueError(
                f"{key.replace('_', ' ').capitalize()} must be greater than zero."
            )

    # Stage list rebuilt from the live number boxes; count comes from the store.
    # The baseline defaults follow the selected turbine type so any key not
    # carried by a live box falls back to the right per-type default.
    defaults = stage_defaults_for(overall.get("turbine_type"))
    n = len(store_stages or [dict(defaults)])
    stages = [
        dict(defaults, **stage)
        for stage in (store_stages or [dict(defaults, name="stage_1")])
    ]
    for d, v in zip(stage_ids or [], stage_values or []):
        si = d["stage"]
        if 0 <= si < n:
            stages[si][d["key"]] = v

    for index, stage in enumerate(stages):
        for key in defaults:
            value = stage.get(key)
            if value is None or isinstance(value, bool) or not np.isfinite(value):
                raise ValueError(
                    f"{stage.get('name', f'Stage {index + 1}')}: enter a finite {key.replace('_', ' ')}."
                )

    return _assemble_cfg(overall, stages)


def render_results(out, signature):
    fig_meridional = _plotting.plot_turbine_meridional_channel(out)
    try:
        fig_blades = _plotting.plot_turbine_blades(
            out, N_points=500, N_blades_plot=8
        )
    except ValueError as error:
        # Physical profile construction must not discard a meanline solve.
        fig_blades = {"data": [], "layout": {"annotations": [{
            "text": f"Blade geometry unavailable: {error}", "showarrow": False,
            "xref": "paper", "yref": "paper", "x": 0.5, "y": 0.5,
        }]}}
    # Now returns a LIST of figures, one per stage.
    triangle_figs = _plotting.plot_velocity_triangles_turbine(
        out, mode="mach"
    )
    fig_loss = _plotting.plot_turbine_loss_distribution(out)

    # Overall efficiency trends vs blade velocity ratio (kinematic sweep).
    nu_sweep = np.linspace(1e-3, 1.0, 100)
    trends = core.compute_turbine_efficiency_trends(out, nu_sweep)
    fig_efficiency_trends = (
        _plotting.plot_turbine_efficiency_trends(trends, out)
    )

    # Wrap each per-stage figure in its own card row.
    triangles_children = [
        triangle_card(f"triangles_plot_{i}", fig)
        for i, fig in enumerate(triangle_figs)
    ]

    # Build the three result tables from the same `out` structure.
    perf_cols, perf_data = reporting.overall_performance_table(out)
    geom_cols, geom_data = reporting.geometry_table(out)
    stn_cols, stn_data = reporting.flow_stations_table(out)
    perf_tbl = make_data_table(perf_cols, perf_data)
    geom_tbl = make_data_table(geom_cols, geom_data)
    stn_tbl = make_data_table(stn_cols, stn_data)

    return (
        fig_meridional,
        fig_blades,
        triangles_children,
        fig_efficiency_trends,
        fig_loss,
        perf_tbl,
        geom_tbl,
        stn_tbl,
        out,
        signature,
    )


def _assemble_cfg(overall, stages):
    """Translate flat overall dict + stage list into the YAML-style cfg."""
    stages = stages or [dict(STAGE_DEFAULTS)]
    inputs = {
        "loss_model": overall.get("loss_model", OVERALL_DEFAULTS["loss_model"]),
        "fluid_name": overall.get("fluid_name", OVERALL_DEFAULTS["fluid_name"]),
        "turbine_type": overall.get("turbine_type", OVERALL_DEFAULTS["turbine_type"]),
        "inlet_property_pair": overall.get(
            "inlet_property_pair", OVERALL_DEFAULTS["inlet_property_pair"]
        ),
        "inlet_property_1": overall.get("inlet_property_1"),
        "inlet_property_2": overall.get("inlet_property_2"),
        "exit_pressure": overall.get("exit_pressure"),
        "mass_flow_rate": overall.get("mass_flow_rate"),
        "inlet_flow_angle": overall.get("inlet_flow_angle"),
        "blade_velocity_ratio": overall.get("blade_velocity_ratio"),
        "rotational_speed": {
            "type": overall.get(
                "rotational_speed_type", OVERALL_DEFAULTS["rotational_speed_type"]
            ),
            "value": overall.get(
                "rotational_speed_value", OVERALL_DEFAULTS["rotational_speed_value"]
            ),
        },
        "stages": [
            {
                "name": sv.get("name", f"stage_{i + 1}"),
                **{
                    k: sv.get(k, STAGE_DEFAULTS[k])
                    for k in STAGE_DEFAULTS
                    if k not in BLADE_GEOMETRY_DEFAULTS
                },
                "blade_geometry": {
                    row: {key: sv[f"{row}.{key}"] for key, *_ in BLADE_GEOMETRY_FIELDS}
                    for row in ("stator", "rotor")
                },
            }
            for i, sv in enumerate(stages)
        ],
    }
    return {"inputs": inputs}


# =========================================
# YAML save / load
# =========================================
@app.callback(
    Output("download_yaml", "data"),
    Input("save_button", "n_clicks"),
    State("result_store", "data"),
    State("result_input_signature", "data"),
    State({"scope": "overall", "key": ALL}, "value"),
    State({"scope": "stage", "stage": ALL, "key": ALL, "elem": "input"}, "value"),
    State({"scope": "overall", "key": ALL}, "id"),
    State({"scope": "stage", "stage": ALL, "key": ALL, "elem": "input"}, "id"),
    State("stage_data_store", "data"),
    State({"type": "opt-variable-path", "row": ALL}, "value"),
    State({"type": "opt-variable-path", "row": ALL}, "id"),
    State({"type": "opt-variable-number", "row": ALL, "field": ALL}, "value"),
    State({"type": "opt-variable-number", "row": ALL, "field": ALL}, "id"),
    State({"type": "opt-constraint-path", "row": ALL}, "value"),
    State({"type": "opt-constraint-path", "row": ALL}, "id"),
    State({"type": "opt-constraint-number", "row": ALL, "field": ALL}, "value"),
    State({"type": "opt-constraint-number", "row": ALL, "field": ALL}, "id"),
    State({"type": "opt-comparison", "row": ALL}, "value"),
    State({"type": "opt-comparison", "row": ALL}, "id"),
    State("opt-objective", "value"),
    State("opt-direction", "value"),
    State("opt-tolerance", "value"),
    State("opt-iterations", "value"),
    State("opt-feasibility", "value"),
    prevent_initial_call=True,
)
def download_yaml(n_clicks, cfg, signature, ov, sv, oi, si, stages, *editor):
    if cfg is None:
        raise PreventUpdate
    try:
        current = live_config(ov, sv, oi, si, stages)
        if input_signature(ov, sv, oi, si, stages) != signature:
            raise ValueError(
                "Wait for a successful calculation of the current inputs before saving."
            )
        variables, constraints = collect_rows(*editor[:4]), collect_rows(*editor[4:10])
        settings = build_settings(
            current,
            variables,
            constraints,
            *editor[10:15],
            results=cfg,
            require_variables=False,
        )
    except (ValueError, KeyError) as error:
        set_props("td-load-error", {"children": f"Design was not saved. {error}"})
        return no_update
    cfg = dict(cfg, inputs=current["inputs"], optimization=settings)
    set_props("td-load-error", {"children": ""})
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    yaml_str = yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False)
    return dict(
        content=yaml_str, filename=f"turbine_{timestamp}.yaml", type="text/yaml"
    )


@app.callback(
    Output("loaded_cfg_store", "data"),
    Input("load_button", "contents"),
    prevent_initial_call=True,
)
def load_design(contents):
    if contents is None:
        raise PreventUpdate
    try:
        _, content_string = contents.split(",", 1)
        decoded = base64.b64decode(content_string, validate=True).decode("utf-8")
        cfg = yaml.safe_load(decoded)
        if not isinstance(cfg, dict) or not isinstance(cfg.get("inputs"), dict):
            raise ValueError("A design must contain an inputs mapping.")
        if (
            not isinstance(cfg["inputs"].get("stages"), list)
            or not cfg["inputs"]["stages"]
        ):
            raise ValueError("A design must contain at least one stage.")
        cfg = nested_config(core.convert_turbine_config_to_solidity(cfg))
        validate_inlet_inputs(cfg["inputs"])
        for stage in cfg["inputs"]["stages"]:
            if not isinstance(stage, dict):
                raise ValueError("Every stage must contain a mapping of design inputs.")
        ids = [{"scope": "overall", "key": key} for key in OVERALL_DEFAULTS]
        live_config(loaded_overall_values(cfg, ids), [], ids, [], loaded_stages(cfg))
    except Exception as error:
        set_props("td-load-error", {"children": f"Could not load YAML. {error}"})
        return no_update
    set_props("td-load-error", {"children": ""})
    return cfg


# =========================================
# Callback: push loaded overall values into the overall inputs
# =========================================
def loaded_overall_values(cfg, overall_ids):
    if not cfg:
        raise PreventUpdate
    inp = nested_config(cfg).get("inputs", {})
    out_values = []
    for id_dict in overall_ids:
        key = id_dict["key"]
        if key == "rotational_speed_type":
            out_values.append(
                inp.get("rotational_speed", {}).get(
                    "type", OVERALL_DEFAULTS["rotational_speed_type"]
                )
            )
        elif key == "rotational_speed_value":
            out_values.append(
                inp.get("rotational_speed", {}).get(
                    "value", OVERALL_DEFAULTS["rotational_speed_value"]
                )
            )
        else:
            out_values.append(inp.get(key, OVERALL_DEFAULTS.get(key)))
    return out_values


register_optimization_callbacks(app, live_config, render_results)


if __name__ == "__main__":
    cli()
