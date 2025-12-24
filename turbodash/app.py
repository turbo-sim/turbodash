import os
import yaml
import numpy as np
import jaxprop as jxp
import turbodash as td
import CoolProp.CoolProp as CP

from datetime import datetime

from dash import (
    Dash,
    dcc,
    html,
    Input,
    Output,
    ctx,
    dash_table,
    State,
)
from dash.dash_table.Format import Format, Scheme


# =========================
# Global / setup
# =========================
app = Dash(__name__)
server = app.server


def main():
    app.run(debug=False)


# =========================
# Default inputs
# =========================
DEFAULTS = dict(
    # Operating conditions
    fluid_name="Air",
    inlet_property_pair="PT_INPUTS",
    inlet_property_1=1e5,  # inlet static pressure [Pa]
    inlet_property_2=300,  # inlet temperature [K]
    p_out=0.5e5,  # exit static pressure [Pa]
    mdot=5.0,  # mass flow rate [kg/s]
    alpha1=0.0,  # absolute flow angle at stator inlet [deg]
    alpha2=70.0, # absolute flow angle at stator exit [deg]
    nu=0.7,  # blade velocity ratio
    R=0.5,  # degree of reaction
    m_12=1.0,  # meridional velocity ratio 1-2
    m_23=1.0,  # meridional velocity ratio 2-3
    m_34=1.0,  # meridional velocity ratio 3-4
    rr_12=0.75,  # radius ratio 1-2
    rr_23=0.95,  # radius ratio 2-3
    rr_34=0.80,  # radius ratio 3-4
    HR_inlet=0.25,  # height-to-radius ratio at inlet
    Z_stator=0.7,  # Zweifel coefficient (stator)
    Z_rotor=0.7,  # Zweifel coefficient (rotor)
    xi_stator=0.05,  # loss coefficient (stator)
    xi_rotor=0.06,  # loss coefficient (rotor
)

# =========================
# Documentation layout
# =========================
# Go one level up from the turbodash/ folder to root/, then into docs/
docs_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),  # root folder
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
# UI helpers
# =========================
def input_only(label_children, id_prefix, default):
    return html.Div(
        style={"marginBottom": "16px", "marginTop": "8px"},
        children=[
            html.Label(
                label_children,
                style={
                    "fontWeight": "bold",
                    "display": "block",
                    "marginBottom": "6px",
                    "marginTop": "8px",
                },
            ),
            dcc.Input(
                id=f"{id_prefix}_input",
                type="number",
                value=default,
                debounce=True,
                style={
                    "width": "95%",
                    "padding": "6px 8px",
                    "fontSize": "14px",
                    "borderRadius": "4px",
                    "border": "1px solid #ccc",
                    "backgroundColor": "#ffffff",
                    "boxShadow": "inset 0 1px 2px rgba(0,0,0,0.08)",
                },
            ),
        ],
    )


def linked_input(label_children, id_prefix, min_val, max_val, step, default):
    return html.Div(
        style={"marginBottom": "16px", "marginTop": "8px"},
        children=[
            html.Label(
                label_children,
                style={
                    "fontWeight": "bold",
                    "display": "block",
                    "marginBottom": "6px",
                    "marginTop": "8px",
                },
            ),
            html.Div(
                style={"display": "flex", "alignItems": "center", "gap": "10px"},
                children=[
                    html.Div(
                        dcc.Slider(
                            id=f"{id_prefix}_slider",
                            min=min_val,
                            max=max_val,
                            step=step,
                            value=default,
                            marks=None,
                            tooltip={"placement": "bottom", "always_visible": False},
                            updatemode="drag",
                        ),
                        style={"flexGrow": "1"},
                    ),
                    dcc.Input(
                        id=f"{id_prefix}_input",
                        type="number",
                        value=default,
                        min=min_val,
                        max=max_val,
                        step=step,
                        debounce=True,  # Wait until number is types before callback
                        style={
                            "width": "60px",
                            "padding": "6px 8px",
                            "fontSize": "14px",
                            "borderRadius": "4px",
                            "border": "1px solid #ccc",
                            "backgroundColor": "#ffffff",
                            "boxShadow": "inset 0 1px 2px rgba(0,0,0,0.08)",
                        },
                    ),
                ],
            ),
        ],
    )


# =========================
# Controls layout
# =========================

LABEL_STYLE = dict(
    fontWeight="bold",
    display="block",
    marginBottom="6px",
    marginTop="8px",
)

CONTROLS_STYLE = dict(
    width="400px",
    padding="30px",
    overflowY="auto",
    borderRight="1px solid #ddd",
    backgroundColor="#fdfdfd",
)


def html_section(title):
    return html.H4(
        title,
        style={
            "marginTop": "24px",
            "paddingBottom": "6px",
            "borderBottom": "1px solid #ddd",
        },
    )


def labeled(label, component):
    return html.Div([html.Label(label, style=LABEL_STYLE), component])


def save_load_row():
    return html.Div(
        [
            html.Button(
                "Save current design",
                id="save_button",
                n_clicks=0,
                style={"padding": "8px 16px", "fontWeight": "bold"},
            ),
            dcc.Upload(
                id="load_button",
                accept=".yaml,.yml",
                children=html.Button(
                    "Load previous design",
                    style={"padding": "8px 16px", "fontWeight": "bold"},
                ),
            ),
        ],
        style=dict(display="flex", gap="12px", marginBottom="10px"),
    )


def stage_type_selector():
    return html.Div(
        [
            html.Label("Stage type", style=LABEL_STYLE),
            dcc.RadioItems(
                id="stage_type",
                options=[
                    {"label": "Axial", "value": "axial"},
                    {"label": "Radial", "value": "radial"},
                ],
                value="radial",
                style=dict(display="flex", flexDirection="column", gap="6px"),
                labelStyle=dict(
                    display="flex",
                    alignItems="center",
                    width="100%",
                    padding="8px 10px",
                    border="1px solid #ddd",
                    borderRadius="4px",
                    backgroundColor="#fafafa",
                ),
                inputStyle={"marginRight": "10px"},
            ),
        ],
        style={"marginBottom": "20px"},
    )


OPERATING_DROPDOWNS = [
    (
        "Working fluid",
        dcc.Dropdown(
            id="fluid_name",
            options=[
                {"label": f, "value": f}
                for f in sorted(CP.get_global_param_string("FluidsList").split(","))
            ],
            value=DEFAULTS["fluid_name"],
            clearable=False,
        ),
    ),
    (
        "Inlet property pair",
        dcc.Dropdown(
            id="inlet_property_pair",
            options=list(jxp.INPUT_PAIRS.keys()),
            value=DEFAULTS["inlet_property_pair"],
            clearable=False,
        ),
    ),
]


DESIGN_VARIABLES = [
    (["Stator inlet angle, α", html.Sub("1"), " [deg]"], "alpha1", -75.0, 75.0, 1.0),
    (["Stator exit angle, α", html.Sub("2"), " [deg]"], "alpha2", 0.0, 90.0, 1.0),
    (["Blade velocity ratio, ν"], "nu", 0.05, 2.0, 0.01),
    (["Degree of reaction, R"], "R", -0.5, 1.0 - 1e-6, 0.01),
    (["Meridional velocity ratio, vₘ₁/vₘ₂"], "m_12", 0.1, 2.0, 0.01),
    (["Meridional velocity ratio, vₘ₂/vₘ₃"], "m_23", 0.1, 2.0, 0.01),
    (["Meridional velocity ratio, vₘ₃/vₘ₄"], "m_34", 0.1, 2.0, 0.01),
    (["Radius ratio, r₁ / r₂"], "rr_12", 0.10, 1.00, 0.001),
    (["Radius ratio, r₂ / r₃"], "rr_23", 0.10, 1.00, 0.001),
    (["Radius ratio, r₃ / r₄"], "rr_34", 0.10, 1.00, 0.001),
    (["Inlet height-to-radius ratio, H₁ / r₁"], "HR_inlet", 0.01, 2.0, 0.005),
    (["Zweifel (stator)"], "Z_stator", 0.1, 2.0, 0.01),
    (["Zweifel (rotor)"], "Z_rotor", 0.1, 2.0, 0.01),
]


controls = html.Div(
    [
        save_load_row(),
        dcc.Download(id="download_meanline_yaml"),
        dcc.Store(id="stage_result_store"),
        dcc.Store(id="loaded_cfg_store"),
        stage_type_selector(),
        html_section("Operating conditions"),
        *[labeled(lbl, comp) for lbl, comp in OPERATING_DROPDOWNS],
        html.Div(
            style={"marginLeft": "16px"},
            children=[
                input_only(["Inlet property 1"], "inlet_property_1", DEFAULTS["inlet_property_1"]),
                input_only(["Inlet property 2"], "inlet_property_2", DEFAULTS["inlet_property_2"]),
            ],
        ),


        input_only(
            ["Exit static pressure, p", html.Sub("out"), " [Pa]"],
            "p_out",
            DEFAULTS["p_out"],
        ),
        input_only(["Mass flow rate, ṁ [kg/s]"], "mdot", DEFAULTS["mdot"]),
        html_section("Design variables"),
        *[
            linked_input(label, key, lo, hi, step, DEFAULTS[key])
            for label, key, lo, hi, step in DESIGN_VARIABLES
        ],
        html_section("Loss coefficients"),
        linked_input(
            ["Loss coefficient, ξ", html.Sub("stator")],
            "xi_stator",
            0.0,
            0.5,
            0.005,
            DEFAULTS["xi_stator"],
        ),
        linked_input(
            ["Loss coefficient, ξ", html.Sub("rotor")],
            "xi_rotor",
            0.0,
            0.5,
            0.005,
            DEFAULTS["xi_rotor"],
        ),
    ],
    style=CONTROLS_STYLE,
)

# =========================
# Results layout
# =========================

TABLE_COLUMNS = {
    "Value",
    "Stator",
    "Rotor",
    "p [bar]",
    "T [K]",
    "ρ [kg/m³]",
    "q [-]",
    "v [m/s]",
    "w [m/s]",
    "u [m/s]",
    "Ma [-]",
    "α [deg]",
    "β [deg]",
    "r [mm]",
    "H [mm]",
}


def make_table(data, columns):
    formatted_columns = []

    for c in columns:
        if c in TABLE_COLUMNS:
            formatted_columns.append(
                {
                    "name": c,
                    "id": c,
                    "type": "numeric",
                    "format": Format(precision=3, scheme=Scheme.fixed),
                }
            )
        else:
            formatted_columns.append({"name": c, "id": c})

    return dash_table.DataTable(
        data=data,
        columns=formatted_columns,
        style_table={
            "overflowX": "auto",
        },
        style_cell={
            "fontFamily": "monospace",
            "fontSize": "13px",
            "padding": "6px",
            "textAlign": "right",
        },
        style_header={
            "fontWeight": "bold",
            "backgroundColor": "#f0f0f0",
            "borderBottom": "2px solid black",
        },
    )


def plot_card(title, graph_id):
    return html.Div(
        style={
            "display": "flex",
            "flexDirection": "column",
            "border": "1px solid #ddd",
            "borderRadius": "6px",
            "padding": "10px",
            "backgroundColor": "white",
        },
        children=[
            html.Div(
                title,
                style={"fontWeight": "bold", "marginBottom": "6px"},
            ),
            dcc.Graph(
                id=graph_id,
                style={"flex": "1 1 auto"},
                config={"responsive": True},
            ),
        ],
    )


def table_card(title, table_id, max_height):
    return html.Div(
        style={
            "display": "flex",
            "flexDirection": "column",
            "border": "1px solid #ddd",
            "borderRadius": "6px",
            "padding": "10px",
            "backgroundColor": "white",
        },
        children=[
            html.Div(
                title,
                style={"fontWeight": "bold", "marginBottom": "8px"},
            ),
            html.Div(
                id=table_id,
                style={
                    "maxHeight": f"{max_height}px",
                    "overflowY": "auto",
                },
            ),
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
        plot_card("Total-to-static efficiency", "efficiency_ts_plot"),
        plot_card("Total-to-total efficiency", "efficiency_tt_plot"),
    ],
)


tables = html.Div(
    style={
        "display": "flex",
        "flexDirection": "column",
        "gap": "24px",
        "marginTop": "24px",
    },
    children=[
        # Row 1: two tables (50/50)
        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": "1fr 1fr",
                "gap": "20px",
            },
            children=[
                table_card("Stage performance", "perf_table", 300),
                table_card("Geometry summary", "geom_table", 300),
            ],
        ),
        # Row 2: full-width table
        html.Div(
            children=[
                table_card("Flow stations", "stations_table", 450),
            ],
        ),
    ],
)





# =========================
# Complete app layout
# =========================
app.layout = html.Div(
    children=[
        dcc.Tabs(
            id="tabs",
            value="calculator",
            style={
                "fontFamily": "Segoe UI, Arial, sans-serif",
                "fontSize": "16px",
                "borderBottom": "1px solid #d0d7de",
            },
            children=[
                dcc.Tab(
                    label="Turbodash",
                    value="calculator",
                    style={
                        "fontWeight": "bold",
                        "padding": "10px 18px",
                        "backgroundColor": "#f6f8fa",
                        "border": "1px solid #d0d7de",
                        "borderBottom": "none",
                    },
                    selected_style={
                        "fontWeight": "bold",
                        "padding": "10px 18px",
                        "backgroundColor": "#ffffff",
                        "border": "1px solid #d0d7de",
                        "borderBottom": "3px solid #007acc",
                    },
                    children=[
                        html.Div(
                            style={
                                "display": "flex",
                                "width": "100vw",
                                "height": "100vh",
                                "overflow": "hidden",
                                "fontFamily": "Arial",
                            },
                            children=[
                                controls,
                                html.Div(
                                    style={
                                        "flex": "1 1 auto",
                                        "overflowY": "auto",
                                        "padding": "20px",
                                    },
                                    children=[plots, tables],
                                ),
                            ],
                        )
                    ],
                ),

                dcc.Tab(
                    label="Documentation",
                    value="docs",
                    style={
                        "fontWeight": "bold",
                        "padding": "10px 18px",
                        "backgroundColor": "#f6f8fa",
                        "border": "1px solid #d0d7de",
                        "borderBottom": "none",
                    },
                    selected_style={
                        "fontWeight": "bold",
                        "padding": "10px 18px",
                        "backgroundColor": "#ffffff",
                        "border": "1px solid #d0d7de",
                        "borderBottom": "3px solid #007acc",
                    },
                    children=[docs_layout],
                ),
            ],
        )
    ]
)




# app.layout = html.Div(
#     style={
#         "display": "flex",
#         "width": "100vw",
#         "height": "100vh",
#         "overflow": "hidden",
#         "fontFamily": "Arial",
#     },
#     children=[
#         # Left: controls (own scrollbar)
#         controls,
#         # Right: plots + tables (own scrollbar)
#         html.Div(
#             style={
#                 "flex": "1 1 auto",
#                 "overflowY": "auto",
#                 "padding": "20px",
#             },
#             children=[
#                 plots,
#                 tables,
#             ],
#         ),
#     ],
# )


# =========================
# Callback: compute + plot
# =========================
@app.callback(
    Output("meridional_plot", "figure"),
    Output("blades_plot", "figure"),
    Output("efficiency_ts_plot", "figure"),
    Output("efficiency_tt_plot", "figure"),
    Output("perf_table", "children"),
    Output("geom_table", "children"),
    Output("stations_table", "children"),
    Output("stage_result_store", "data"),
    Input("fluid_name", "value"),
    Input("stage_type", "value"),
    Input("inlet_property_pair", "value"),
    Input("inlet_property_1_input", "value"),
    Input("inlet_property_2_input", "value"),
    Input("p_out_input", "value"),
    Input("mdot_input", "value"),
    Input("alpha1_slider", "value"),
    Input("alpha2_slider", "value"),
    Input("nu_slider", "value"),
    Input("R_slider", "value"),
    Input("m_12_slider", "value"),
    Input("m_23_slider", "value"),
    Input("m_34_slider", "value"),
    Input("rr_12_slider", "value"),
    Input("rr_23_slider", "value"),
    Input("rr_34_slider", "value"),
    Input("HR_inlet_slider", "value"),
    Input("Z_stator_slider", "value"),
    Input("Z_rotor_slider", "value"),
    Input("xi_stator_slider", "value"),
    Input("xi_rotor_slider", "value"),
)
def update_turbine(
    fluid_name,
    stage_type,
    inlet_property_pair,
    inlet_property_1,
    inlet_property_2,
    p_out,
    mdot,
    alpha1,
    alpha2,
    nu,
    R,
    m_12,
    m_23,
    m_34,
    rr_12,
    rr_23,
    rr_34,
    HR_inlet,
    Z_stator,
    Z_rotor,
    xi_stator,
    xi_rotor,
):

    try:
        # Calculate performance
        out = td.compute_stage_meanline(
            fluid_name=fluid_name,
            inlet_property_pair_string=inlet_property_pair,
            inlet_property_1=inlet_property_1,
            inlet_property_2=inlet_property_2,
            exit_pressure=p_out,
            mass_flow_rate=mdot,
            stator_inlet_angle=alpha1,
            stator_exit_angle=alpha2,
            blade_velocity_ratio=nu,
            degree_reaction=R,
            meridional_velocity_ratio_12=m_12,
            meridional_velocity_ratio_23=m_23,
            meridional_velocity_ratio_34=m_34,
            radius_ratio_12=rr_12,
            radius_ratio_23=rr_23,
            radius_ratio_34=rr_34,
            height_radius_ratio=HR_inlet,
            zweiffel_stator=Z_stator,
            zweiffel_rotor=Z_rotor,
            loss_coeff_stator=xi_stator,
            loss_coeff_rotor=xi_rotor,
            stage_type=stage_type,
        )

        # Convert results to Python types
        out_python = {k: getattr(v, "item", lambda: v)() for k, v in out.items()}

        # Create plots
        fig_meridional = td.plotly.plot_meridional_channel(out)
        fig_blades = td.plotly.plot_blades(out, N_points=500, N_blades_plot=10)
        fig_ts = td.plotly.plot_eta_ts(out)
        fig_tt = td.plotly.plot_eta_tt(out)

        # Create tables
        perf_data = td.stage_performance_table(out)
        geom_data = td.geometry_table(out)
        stations_data = td.flow_stations_table(out)
        perf_table = make_table(perf_data, ["Quantity", "Value", "Unit"])
        geom_table = make_table(geom_data, ["Variable", "Stator", "Rotor", "Unit"])
        stations_table = make_table(stations_data, list(stations_data[0].keys()))

        return (
            fig_meridional,
            fig_blades,
            fig_ts,
            fig_tt,
            perf_table,
            geom_table,
            stations_table,
            out_python,
        )
        # return fig_meridional, fig_blades, fig_ts, fig_tt, None, None, None, out_python

    except Exception as e:
        print("\n=== ERROR IN update_turbine CALLBACK ===")
        print(e)
        print("========================================\n")
        raise


# =========================================
# YAML saving
# =========================================
@app.callback(
    Output("download_meanline_yaml", "data"),
    Input("save_button", "n_clicks"),
    State("stage_result_store", "data"),
    prevent_initial_call=True,
)
def download_meanline_yaml(n_clicks, out):
    if out is None:
        return None

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"meanline_stage_{timestamp}.yaml"

    yaml_str = yaml.safe_dump(
        out,
        sort_keys=False,
        default_flow_style=False,
    )

    return dict(
        content=yaml_str,
        filename=filename,
        type="text/yaml",
    )


# =========================================
# Register slider–input sync + YAML load
# =========================================

@app.callback(
    Output("loaded_cfg_store", "data"),
    Input("load_button", "contents"),
    prevent_initial_call=True,
)
def load_design(contents):
    import base64, yaml
    from dash.exceptions import PreventUpdate

    if contents is None:
        raise PreventUpdate

    _, content_string = contents.split(",")
    decoded = base64.b64decode(content_string).decode("utf-8")
    cfg = yaml.safe_load(decoded) or {}
    return cfg


def register_link_value(prefix, min_val, max_val, cfg_key=None):
    slider_id = f"{prefix}_slider"
    input_id = f"{prefix}_input"

    @app.callback(
        Output(slider_id, "value"),
        Output(input_id, "value"),
        Input(slider_id, "value"),
        Input(input_id, "value"),
        Input("loaded_cfg_store", "data"),
        prevent_initial_call=True,
    )
    def _sync(val_slider, val_input, loaded_cfg):
        trigger = ctx.triggered_id

        # 1) Loading: apply cfg["inputs"][cfg_key] if present
        if trigger == "loaded_cfg_store" and loaded_cfg and cfg_key:
            inp = loaded_cfg.get("inputs", {})
            if cfg_key in inp and inp[cfg_key] is not None:
                v = float(np.clip(inp[cfg_key], min_val, max_val))
                return v, v

        # 2) Normal slider <-> input sync
        v_slider = np.clip(
            val_slider if val_slider is not None else min_val, min_val, max_val
        )
        v_input = np.clip(
            val_input if val_input is not None else min_val, min_val, max_val
        )

        if trigger == slider_id:
            return float(v_slider), float(v_slider)
        if trigger == input_id:
            return float(v_input), float(v_input)

        return float(v_slider), float(v_input)


def register_input_only_load(id_prefix, cfg_key):
    input_id = f"{id_prefix}_input"

    @app.callback(
        Output(input_id, "value"),
        Input("loaded_cfg_store", "data"),
        State(input_id, "value"),
        prevent_initial_call=True,
    )
    def _load_input_only(loaded_cfg, cur):
        if not loaded_cfg:
            return cur
        inp = loaded_cfg.get("inputs", {})
        if cfg_key in inp and inp[cfg_key] is not None:
            return inp[cfg_key]
        return cur


register_input_only_load("inlet_property_1", "inlet_property_1")
register_input_only_load("inlet_property_2", "inlet_property_2")
register_input_only_load("p_out", "exit_pressure")
register_input_only_load("mdot", "mass_flow_rate")
register_link_value("alpha1", -75.0, 75.0, cfg_key="stator_inlet_angle")
register_link_value("alpha2", 0.0, 90.0, cfg_key="stator_exit_angle")
register_link_value("nu", 0.05, 2.0, cfg_key="blade_velocity_ratio")
register_link_value("R", -0.5, 1.0 - 1e-9, cfg_key="degree_reaction")
register_link_value("HR_inlet", 0.01, 2.0, cfg_key="height_radius_ratio")
register_link_value("rr_12", 0.10, 1.00, cfg_key="radius_ratio_12")
register_link_value("rr_23", 0.10, 1.00, cfg_key="radius_ratio_23")
register_link_value("rr_34", 0.10, 1.00, cfg_key="radius_ratio_34")
register_link_value("m_12", 0.1, 2.0, cfg_key="meridional_velocity_ratio_12")
register_link_value("m_23", 0.1, 2.0, cfg_key="meridional_velocity_ratio_23")
register_link_value("m_34", 0.1, 2.0, cfg_key="meridional_velocity_ratio_34")
register_link_value("Z_stator", 0.1, 1.5, cfg_key="zweiffel_stator")
register_link_value("Z_rotor", 0.1, 1.5, cfg_key="zweiffel_rotor")
register_link_value("xi_stator", 0.0, 0.5, cfg_key="loss_coeff_stator")
register_link_value("xi_rotor", 0.0, 0.5, cfg_key="loss_coeff_rotor")


@app.callback(
    Output("fluid_name", "value"),
    Output("stage_type", "value"),
    Output("inlet_property_pair", "value"),
    Input("loaded_cfg_store", "data"),
    State("fluid_name", "value"),
    State("stage_type", "value"),
    State("inlet_property_pair", "value"),
    prevent_initial_call=True,
)
def apply_loaded_meta(cfg, fluid_cur, stage_cur, pair_cur):
    if not cfg:
        return fluid_cur, stage_cur, pair_cur

    inp = cfg.get("inputs", {})

    fluid = inp.get("fluid", fluid_cur)
    stage = inp.get("stage_type", stage_cur)
    pair = inp.get("inlet_property_pair", pair_cur)  # now a string in YAML

    return fluid, stage, pair
