import os
import base64
import yaml
import numpy as np
import jaxprop as jxp
import turbodash as td
import CoolProp.CoolProp as CP
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
    MATCH,
    dash_table,
)
from dash.exceptions import PreventUpdate


# =========================
# Global / setup
# =========================
app = Dash(
    __name__,
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
)
server = app.server


def main():
    app.run(debug=False)


# =========================
# Defaults
# =========================
# Overall (turbine-level) defaults. Mirror config_radial.yaml inputs.
OVERALL_DEFAULTS = dict(
    loss_model="benner",
    fluid_name="Air",
    stage_type="radial",
    inlet_property_pair="PT_INPUTS",
    inlet_property_1=10e5,
    inlet_property_2=600.0,
    exit_pressure=2e5,
    mass_flow_rate=10.0,
    inlet_flow_angle=0.0,
    blade_velocity_ratio=0.2,
    rotational_speed_type="specific",
    rotational_speed_value=0.3,
)

# Per-stage defaults (one stage's worth). New stages are seeded from this.
STAGE_DEFAULTS = dict(
    work_fraction_split=1.0,
    stator_exit_angle=70.0,
    degree_reaction=0.5,
    meridional_velocity_ratio_12=1.00,
    meridional_velocity_ratio_23=1.00,
    meridional_velocity_ratio_34=1.00,
    radius_ratio_01=1.00,
    radius_ratio_12=0.90,
    radius_ratio_23=0.98,
    radius_ratio_34=0.90,
    zweiffel_stator=0.7,
    zweiffel_rotor=0.7,
)

# Per-stage fields: (cfg_key, label, min, max, step). Labels carry the full
# variable name with symbol and unit. Order = display order. Each renders as a
# slider + linked number box.
STAGE_FIELDS = [
    ("work_fraction_split", ["Work fraction split"], 0.1, 10.0, 0.1),
    ("stator_exit_angle", ["Stator exit angle, α", html.Sub("2"), " [deg]"], 0.0, 89.0, 1.0),
    ("degree_reaction", ["Degree of reaction, R"], -0.5, 1.0 - 1e-6, 0.01),
    ("meridional_velocity_ratio_12", ["Meridional velocity ratio, vₘ₁/vₘ₂"], 0.1, 2.0, 0.01),
    ("meridional_velocity_ratio_23", ["Meridional velocity ratio, vₘ₂/vₘ₃"], 0.1, 2.0, 0.01),
    ("meridional_velocity_ratio_34", ["Meridional velocity ratio, vₘ₃/vₘ₄"], 0.1, 2.0, 0.01),
    ("radius_ratio_01", ["Radius ratio, r₀/r₁"], 0.10, 1.20, 0.01),
    ("radius_ratio_12", ["Radius ratio, r₁/r₂"], 0.10, 1.20, 0.01),
    ("radius_ratio_23", ["Radius ratio, r₂/r₃"], 0.10, 1.20, 0.01),
    ("radius_ratio_34", ["Radius ratio, r₃/r₄"], 0.10, 1.20, 0.01),
    ("zweiffel_stator", ["Zweifel coefficient (stator)"], 0.1, 2.0, 0.01),
    ("zweiffel_rotor", ["Zweifel coefficient (rotor)"], 0.1, 2.0, 0.01),
]

# Loss-model dropdown: internal value -> pretty label (Title Case, spaces).
LOSS_MODEL_OPTIONS = ["benner", "kacker_okapuu", "moustapha", "isentropic"]


def _prettify(name):
    """internal_name -> 'Internal Name' for display in dropdowns."""
    return name.replace("_", " ").title()


# =========================
# Documentation layout
# =========================
docs_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "docs",
    "documentation.md",
)
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
                "color": "#24292e",
                "backgroundColor": "#ffffff",
            },
        ),
    ],
)


# =========================
# Styles
# =========================
LABEL_STYLE = dict(
    fontWeight="bold", display="block", marginBottom="6px", marginTop="8px"
)
CONTROLS_STYLE = dict(
    width="600px",
    padding="24px",
    overflowY="auto",
    borderRight="1px solid #ddd",
    backgroundColor="#fdfdfd",
)
INPUT_STYLE = {
    "width": "60px",
    "padding": "6px 8px",
    "fontSize": "14px",
    "borderRadius": "4px",
    "border": "1px solid #ccc",
    "backgroundColor": "#ffffff",
}
WIDE_INPUT_STYLE = {**INPUT_STYLE, "width": "95%"}


def html_section(title):
    return html.H4(
        title,
        style={"marginTop": "20px", "paddingBottom": "6px", "borderBottom": "1px solid #ddd"},
    )


def labeled(label, component):
    return html.Div([html.Label(label, style=LABEL_STYLE), component])


# =========================
# UI helpers
# =========================
def overall_number(label, key, default):
    """A plain number input bound to an overall-parameter dict id."""
    return html.Div(
        style={"marginBottom": "12px"},
        children=[
            html.Label(label, style=LABEL_STYLE),
            dcc.Input(
                id={"scope": "overall", "key": key},
                type="number",
                value=default,
                debounce=True,
                style=WIDE_INPUT_STYLE,
            ),
        ],
    )


def stage_linked(stage_idx, key, label, lo, hi, step, value):
    """
    A slider + linked number box bound to per-stage dict ids. The two share
    the same (stage, key) and differ by `elem` ('slider' / 'input'); a
    pattern-matching callback keeps them in sync.
    """
    return html.Div(
        style={"marginBottom": "16px", "marginTop": "8px"},
        children=[
            html.Label(label, style=LABEL_STYLE),
            html.Div(
                style={"display": "flex", "alignItems": "center", "gap": "10px"},
                children=[
                    html.Div(
                        dcc.Slider(
                            id={"scope": "stage", "stage": stage_idx, "key": key, "elem": "slider"},
                            min=lo, max=hi, step=step, value=value,
                            marks=None,
                            tooltip={"placement": "bottom", "always_visible": False},
                            updatemode="drag",
                        ),
                        style={"flexGrow": "1"},
                    ),
                    dcc.Input(
                        id={"scope": "stage", "stage": stage_idx, "key": key, "elem": "input"},
                        type="number",
                        value=value, min=lo, max=hi, step=step,
                        debounce=True,
                        style=INPUT_STYLE,
                    ),
                ],
            ),
        ],
    )


def accordion_title(text):
    return html.Span(text, style={"fontWeight": "bold", "fontSize": "15px"})


def make_stage_accordion_item(stage_idx, stage_values):
    """Build one AccordionItem for a single stage from a values dict."""
    fields = [
        stage_linked(
            stage_idx, key, label, lo, hi, step,
            stage_values.get(key, STAGE_DEFAULTS[key]),
        )
        for (key, label, lo, hi, step) in STAGE_FIELDS
    ]
    return dbc.AccordionItem(
        title=accordion_title(f"Stage {stage_idx + 1}"),
        item_id=f"stage-{stage_idx}",
        children=html.Div(fields, style={"paddingTop": "6px"}),
    )


# =========================
# Controls: overall accordion + dynamic stage accordion
# =========================
def stage_count_row():
    return html.Div(
        style={"display": "flex", "alignItems": "center", "gap": "10px", "margin": "14px 0"},
        children=[
            html.Label("Number of stages", style={"fontWeight": "bold"}),
            html.Button("-", id="stage_minus", n_clicks=0, style={"width": "34px", "fontWeight": "bold"}),
            html.Div(id="stage_count_label", style={"minWidth": "24px", "textAlign": "center"}),
            html.Button("+", id="stage_plus", n_clicks=0, style={"width": "34px", "fontWeight": "bold"}),
        ],
    )


def overall_accordion():
    body = [
        # 1. Turbine type (formerly "stage type")
        labeled(
            "Turbine type",
            dcc.RadioItems(
                id={"scope": "overall", "key": "stage_type"},
                options=[
                    {"label": "Axial", "value": "axial"},
                    {"label": "Radial", "value": "radial"},
                ],
                value=OVERALL_DEFAULTS["stage_type"],
                style=dict(display="flex", gap="14px"),
            ),
        ),
        # 2. Number of stages
        stage_count_row(),
        # 3. Loss model (pretty labels, internal values)
        labeled(
            "Loss model",
            dcc.Dropdown(
                id={"scope": "overall", "key": "loss_model"},
                options=[{"label": _prettify(m), "value": m} for m in LOSS_MODEL_OPTIONS],
                value=OVERALL_DEFAULTS["loss_model"],
                clearable=False,
            ),
        ),
        # 4. Working fluid and the rest
        labeled(
            "Working fluid",
            dcc.Dropdown(
                id={"scope": "overall", "key": "fluid_name"},
                options=[
                    {"label": f, "value": f}
                    for f in sorted(CP.get_global_param_string("FluidsList").split(","))
                ],
                value=OVERALL_DEFAULTS["fluid_name"],
                clearable=False,
            ),
        ),
        labeled(
            "Inlet property pair",
            dcc.Dropdown(
                id={"scope": "overall", "key": "inlet_property_pair"},
                options=list(jxp.INPUT_PAIRS.keys()),
                value=OVERALL_DEFAULTS["inlet_property_pair"],
                clearable=False,
            ),
        ),
        overall_number("Inlet property 1", "inlet_property_1", OVERALL_DEFAULTS["inlet_property_1"]),
        overall_number("Inlet property 2", "inlet_property_2", OVERALL_DEFAULTS["inlet_property_2"]),
        overall_number(["Exit pressure, p", html.Sub("out"), " [Pa]"], "exit_pressure", OVERALL_DEFAULTS["exit_pressure"]),
        overall_number("Mass flow rate, ṁ [kg/s]", "mass_flow_rate", OVERALL_DEFAULTS["mass_flow_rate"]),
        overall_number(["Inlet flow angle, α", html.Sub("0"), " [deg]"], "inlet_flow_angle", OVERALL_DEFAULTS["inlet_flow_angle"]),
        overall_number("Blade velocity ratio, ν", "blade_velocity_ratio", OVERALL_DEFAULTS["blade_velocity_ratio"]),
        labeled(
            "Rotational speed type",
            dcc.RadioItems(
                id={"scope": "overall", "key": "rotational_speed_type"},
                options=[
                    {"label": "Specific", "value": "specific"},
                    {"label": "Actual", "value": "actual"},
                ],
                value=OVERALL_DEFAULTS["rotational_speed_type"],
                style=dict(display="flex", gap="14px"),
            ),
        ),
        overall_number("Rotational speed value", "rotational_speed_value", OVERALL_DEFAULTS["rotational_speed_value"]),
    ]
    return dbc.Accordion(
        id="overall_accordion",
        start_collapsed=False,
        always_open=True,
        children=[
            dbc.AccordionItem(
                title=accordion_title("Overall parameters"),
                item_id="overall",
                children=html.Div(body),
            )
        ],
    )


def save_load_row():
    return html.Div(
        [
            html.Button("Save design", id="save_button", n_clicks=0,
                        style={"padding": "8px 16px", "fontWeight": "bold"}),
            dcc.Upload(
                id="load_button",
                accept=".yaml,.yml",
                children=html.Button("Load design",
                                     style={"padding": "8px 16px", "fontWeight": "bold"}),
            ),
        ],
        style=dict(display="flex", gap="12px", marginBottom="10px"),
    )


controls = html.Div(
    [
        save_load_row(),
        dcc.Download(id="download_yaml"),
        dcc.Store(id="result_store"),
        dcc.Store(id="loaded_cfg_store"),
        # stage_data_store holds the list of per-stage value dicts. It only
        # changes on +/- or load, so editing a value never rebuilds (and never
        # collapses) the per-stage accordion.
        dcc.Store(id="stage_data_store", data=[dict(STAGE_DEFAULTS)]),
        overall_accordion(),
        # Rebuilt only when the stage count or a loaded config changes.
        html.Div(id="stages_accordion_container", style={"marginTop": "12px"}),
    ],
    style=CONTROLS_STYLE,
)


# =========================
# Results layout (plots; tables left as placeholders)
# =========================
def plot_card(title, graph_id):
    return html.Div(
        style={
            "display": "flex", "flexDirection": "column",
            "border": "1px solid #ddd", "borderRadius": "6px",
            "padding": "10px", "backgroundColor": "white",
        },
        children=[
            html.Div(title, style={"fontWeight": "bold", "marginBottom": "6px"}),
            dcc.Graph(id=graph_id, style={"flex": "1 1 auto"}, config={"responsive": True}),
        ],
    )


plots = html.Div(
    style={
        "display": "grid",
        "gridTemplateColumns": "repeat(2, 1fr)",
        "gridAutoRows": "minmax(420px, 1fr)",
        "gap": "24px",
    },
    children=[
        plot_card("Meridional channel", "meridional_plot"),
        plot_card("Blade-to-blade view", "blades_plot"),
        plot_card("Velocity triangles", "triangles_plot"),
        plot_card("Loss distribution", "loss_plot"),
    ],
)

# Placeholder for tables (intentionally empty for now).
tables = html.Div(
    id="tables_container",
    style={"marginTop": "24px"},
    children=[
        # TODO: stage performance / geometry / flow-stations tables go here.
        html.Div(id="perf_table"),
        html.Div(id="geom_table"),
        html.Div(id="stations_table"),
    ],
)


# =========================
# App layout
# =========================
def styled_tab(label, value):
    base = {"fontWeight": "bold", "padding": "10px 18px",
            "backgroundColor": "#f6f8fa", "border": "1px solid #d0d7de", "borderBottom": "none"}
    sel = {**base, "backgroundColor": "#ffffff", "borderBottom": "3px solid #007acc"}
    return dict(style=base, selected_style=sel, label=label, value=value)


app.layout = html.Div(
    children=[
        dcc.Tabs(
            id="tabs",
            value="calculator",
            style={"fontFamily": "Segoe UI, Arial, sans-serif", "fontSize": "16px",
                   "borderBottom": "1px solid #d0d7de"},
            children=[
                dcc.Tab(
                    **styled_tab("Turbodash", "calculator"),
                    children=[
                        html.Div(
                            style={"display": "flex", "width": "100vw", "height": "100vh",
                                   "overflow": "hidden", "fontFamily": "Arial"},
                            children=[
                                controls,
                                html.Div(
                                    style={"flex": "1 1 auto", "overflowY": "auto", "padding": "20px"},
                                    children=[plots, tables],
                                ),
                            ],
                        )
                    ],
                ),
                dcc.Tab(**styled_tab("Documentation", "docs"), children=[docs_layout]),
            ],
        )
    ]
)


# =========================================
# Callback: maintain stage_data_store. Fires ONLY on +/- or load, never on a
# value edit, so editing a parameter does not rebuild/collapse the accordion.
# =========================================
@app.callback(
    Output("stage_data_store", "data"),
    Input("stage_plus", "n_clicks"),
    Input("stage_minus", "n_clicks"),
    Input("loaded_cfg_store", "data"),
    State("stage_data_store", "data"),
    prevent_initial_call=True,
)
def update_stage_store(n_plus, n_minus, loaded_cfg, stages):
    trigger = ctx.triggered_id
    stages = list(stages or [dict(STAGE_DEFAULTS)])

    # Loading a config: replace the whole stage list from the YAML.
    if trigger == "loaded_cfg_store" and loaded_cfg:
        cfg_stages = loaded_cfg.get("inputs", {}).get("stages", [])
        if cfg_stages:
            new_stages = []
            for sc in cfg_stages:
                merged = dict(STAGE_DEFAULTS)
                merged.update({k: sc[k] for k in STAGE_DEFAULTS if k in sc})
                new_stages.append(merged)
            return new_stages
        raise PreventUpdate

    # Add a stage: copy the last stage's values as the seed.
    if trigger == "stage_plus":
        seed = dict(stages[-1]) if stages else dict(STAGE_DEFAULTS)
        return stages + [seed]

    # Remove a stage: never go below one.
    if trigger == "stage_minus":
        return stages[:-1] if len(stages) > 1 else stages

    raise PreventUpdate


# =========================================
# Callback: keep each stage slider and its number box in sync
# =========================================
@app.callback(
    Output({"scope": "stage", "stage": MATCH, "key": MATCH, "elem": "slider"}, "value"),
    Output({"scope": "stage", "stage": MATCH, "key": MATCH, "elem": "input"}, "value"),
    Input({"scope": "stage", "stage": MATCH, "key": MATCH, "elem": "slider"}, "value"),
    Input({"scope": "stage", "stage": MATCH, "key": MATCH, "elem": "input"}, "value"),
    prevent_initial_call=True,
)
def sync_stage_slider_input(slider_val, input_val):
    trigger = ctx.triggered_id
    if trigger is None:
        raise PreventUpdate
    val = slider_val if trigger.get("elem") == "slider" else input_val
    if val is None:
        raise PreventUpdate
    return val, val


# =========================================
# Callback: rebuild the per-stage accordion. Fires only when stage_data_store
# changes (i.e. +/- or load), so it does not collapse on value edits.
# =========================================
@app.callback(
    Output("stages_accordion_container", "children"),
    Output("stage_count_label", "children"),
    Input("stage_data_store", "data"),
    prevent_initial_call=False,
)
def render_stage_accordion(stages):
    stages = stages or [dict(STAGE_DEFAULTS)]
    items = [make_stage_accordion_item(i, sv) for i, sv in enumerate(stages)]
    accordion = dbc.Accordion(
        id="stages_accordion",
        always_open=True,
        children=items,
    )
    return accordion, str(len(stages))


# =========================================
# Callback: compute + plot. Reads LIVE values from the inputs (overall + each
# stage's number box), so it updates on every edit without touching the store.
# =========================================
@app.callback(
    Output("meridional_plot", "figure"),
    Output("blades_plot", "figure"),
    Output("triangles_plot", "figure"),
    Output("loss_plot", "figure"),
    Output("result_store", "data"),
    Input({"scope": "overall", "key": ALL}, "value"),
    Input({"scope": "stage", "stage": ALL, "key": ALL, "elem": "input"}, "value"),
    State({"scope": "overall", "key": ALL}, "id"),
    State({"scope": "stage", "stage": ALL, "key": ALL, "elem": "input"}, "id"),
    State("stage_data_store", "data"),
)
def update_turbine(overall_values, stage_values, overall_ids, stage_ids, store_stages):
    # Overall inputs.
    overall = {d["key"]: v for d, v in zip(overall_ids or [], overall_values or [])}

    # Stage list rebuilt from the live number boxes; count comes from the store.
    n = len(store_stages or [dict(STAGE_DEFAULTS)])
    stages = [dict(STAGE_DEFAULTS) for _ in range(n)]
    for d, v in zip(stage_ids or [], stage_values or []):
        si = d["stage"]
        if 0 <= si < n and v is not None:
            stages[si][d["key"]] = v

    cfg = _assemble_cfg(overall, stages)

    try:
        out = td.core_turbine.compute_turbine_performance(cfg)

        fig_meridional = td.plotting_plotly_turbine.plot_turbine_meridional_channel(out)
        fig_blades = td.plotting_plotly_turbine.plot_turbine_blades(out, N_points=500, N_blades_plot=8)
        fig_triangles = td.plotting_plotly_turbine.plot_velocity_triangles_turbine(out, mode="mach")
        fig_loss = td.plotting_plotly_turbine.plot_turbine_loss_distribution(out)

        # TODO: build tables here once table builders are ready.
        return fig_meridional, fig_blades, fig_triangles, fig_loss, cfg

    except Exception as e:
        print("\n=== ERROR IN update_turbine CALLBACK ===")
        print(repr(e))
        print("cfg was:", cfg)
        print("========================================\n")
        raise


def _assemble_cfg(overall, stages):
    """Translate flat overall dict + stage list into the YAML-style cfg."""
    stages = stages or [dict(STAGE_DEFAULTS)]
    inputs = {
        "loss_model": overall.get("loss_model", OVERALL_DEFAULTS["loss_model"]),
        "fluid_name": overall.get("fluid_name", OVERALL_DEFAULTS["fluid_name"]),
        "stage_type": overall.get("stage_type", OVERALL_DEFAULTS["stage_type"]),
        "inlet_property_pair": overall.get("inlet_property_pair", OVERALL_DEFAULTS["inlet_property_pair"]),
        "inlet_property_1": overall.get("inlet_property_1"),
        "inlet_property_2": overall.get("inlet_property_2"),
        "exit_pressure": overall.get("exit_pressure"),
        "mass_flow_rate": overall.get("mass_flow_rate"),
        "inlet_flow_angle": overall.get("inlet_flow_angle"),
        "blade_velocity_ratio": overall.get("blade_velocity_ratio"),
        "rotational_speed": {
            "type": overall.get("rotational_speed_type", OVERALL_DEFAULTS["rotational_speed_type"]),
            "value": overall.get("rotational_speed_value", OVERALL_DEFAULTS["rotational_speed_value"]),
        },
        "stages": [
            {"name": f"stage_{i + 1}", **{k: sv.get(k, STAGE_DEFAULTS[k]) for k in STAGE_DEFAULTS}}
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
    prevent_initial_call=True,
)
def download_yaml(n_clicks, cfg):
    if cfg is None:
        raise PreventUpdate
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    yaml_str = yaml.safe_dump(cfg, sort_keys=False, default_flow_style=False)
    return dict(content=yaml_str, filename=f"turbine_{timestamp}.yaml", type="text/yaml")


@app.callback(
    Output("loaded_cfg_store", "data"),
    Input("load_button", "contents"),
    prevent_initial_call=True,
)
def load_design(contents):
    if contents is None:
        raise PreventUpdate
    _, content_string = contents.split(",")
    decoded = base64.b64decode(content_string).decode("utf-8")
    cfg = yaml.safe_load(decoded) or {}
    return cfg


# =========================================
# Callback: push loaded overall values into the overall inputs
# =========================================
@app.callback(
    Output({"scope": "overall", "key": ALL}, "value"),
    Input("loaded_cfg_store", "data"),
    State({"scope": "overall", "key": ALL}, "id"),
    prevent_initial_call=True,
)
def apply_loaded_overall(cfg, overall_ids):
    if not cfg:
        raise PreventUpdate
    inp = cfg.get("inputs", {})
    out_values = []
    for id_dict in overall_ids:
        key = id_dict["key"]
        if key == "rotational_speed_type":
            out_values.append(inp.get("rotational_speed", {}).get("type",
                              OVERALL_DEFAULTS["rotational_speed_type"]))
        elif key == "rotational_speed_value":
            out_values.append(inp.get("rotational_speed", {}).get("value",
                              OVERALL_DEFAULTS["rotational_speed_value"]))
        else:
            out_values.append(inp.get(key, OVERALL_DEFAULTS.get(key)))
    return out_values


if __name__ == "__main__":
    main()