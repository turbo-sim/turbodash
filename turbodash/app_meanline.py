import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, ctx, dash_table, State

from dash.dash_table.Format import Format, Scheme
from plotly.colors import sample_colorscale

import jaxprop as jxp
import turbodash as td
import CoolProp.CoolProp as CP

import yaml
from datetime import datetime


# =========================
# Global / setup
# =========================
# === initialize app ===
app = Dash(__name__)
server = app.server

def main():
    app.run(debug=False)



# =========================
# UI helpers
# =========================
def linked_input(label_children, id_prefix, min_val, max_val, step, default):
    return html.Div(
        style={"marginBottom": "16px"},
        children=[
            html.Label(
                label_children,
                style={"fontWeight": "bold", "display": "block", "marginBottom": "6px"},
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
                        style={"width": "110px"},
                    ),
                ],
            ),
        ],
    )


# def register_link_value(prefix, min_val, max_val):
#     @app.callback(
#         Output(f"{prefix}_slider", "value"),
#         Output(f"{prefix}_input", "value"),
#         Input(f"{prefix}_slider", "value"),
#         Input(f"{prefix}_input", "value"),
#     )
#     def _sync(val_slider, val_input):
#         trigger = ctx.triggered_id
#         v_slider = np.clip(val_slider if val_slider is not None else min_val, min_val, max_val)
#         v_input = np.clip(val_input if val_input is not None else min_val, min_val, max_val)

#         if trigger == f"{prefix}_slider":
#             return float(v_slider), float(v_slider)
#         if trigger == f"{prefix}_input":
#             return float(v_input), float(v_input)

#         return float(v_slider), float(v_input)


def register_link_value(prefix, min_val, max_val, yaml_key=None):
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

        # 1) If a config was loaded, and it contains this key, apply it
        if trigger == "loaded_cfg_store" and loaded_cfg is not None and yaml_key is not None:
            raw = loaded_cfg.get(yaml_key, None)
            if raw is not None:
                v = float(np.clip(raw, min_val, max_val))
                return v, v  # set both to loaded value

        # 2) Normal slider<->input sync
        v_slider = np.clip(val_slider if val_slider is not None else min_val, min_val, max_val)
        v_input  = np.clip(val_input  if val_input  is not None else min_val, min_val, max_val)

        if trigger == slider_id:
            return float(v_slider), float(v_slider)
        if trigger == input_id:
            return float(v_input), float(v_input)

        # Fallback
        return float(v_slider), float(v_input)



def to_python(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


# =========================
# Defaults
# =========================
defaults = dict(
    # Operating conditions
    d_in=1.0,            # kg/m3
    p_in=1e5,            # Pa
    p_out=0.5e5,         # Pa
    mdot=5.0,            # kg/s

    # Angles / performance inputs
    alpha1=0.0,          # deg
    alpha2=70.0,         # deg
    nu=0.7,              # blade velocity ratio
    R=0.5,               # degree of reaction

    # Radii ratios (meanline sizing inputs)
    rr_12=0.75,
    rr_23=0.95,
    rr_34=0.80,

    # Geometry closures
    HR_inlet=0.25,
    Z_stator=0.7,
    Z_rotor=0.7,

    # Losses
    xi_stator=0.05,
    xi_rotor=0.06,
)

# INPUTS_TO_DASH = {
#     "inputs.fluid": "fluid_name",
#     "inputs.stage_type": "stage_type",

#     "inputs.inlet_pressure": "p_in_slider",
#     "inputs.exit_pressure": "p_out_slider",
#     "inputs.mass_flow_rate": "mdot_slider",

#     "inputs.stator_inlet_angle": "alpha1_slider",
#     "inputs.stator_exit_angle": "alpha2_slider",
#     "inputs.blade_velocity_ratio": "nu_slider",
#     "inputs.degree_reaction": "R_slider",

#     "inputs.radius_ratio_12": "rr_12_slider",
#     "inputs.radius_ratio_23": "rr_23_slider",
#     "inputs.radius_ratio_34": "rr_34_slider",

#     "inputs.height_radius_ratio": "HR_inlet_slider",

#     "inputs.zweiffel_stator": "Z_stator_slider",
#     "inputs.zweiffel_rotor": "Z_rotor_slider",

#     "inputs.loss_coeff_stator": "xi_stator_slider",
#     "inputs.loss_coeff_rotor": "xi_rotor_slider",
# }

INPUTS_TO_DASH = {
    "inputs.fluid": "fluid_name",
    "inputs.stage_type": "stage_type",

    "inputs.inlet_pressure": "p_in_slider",
    "inputs.exit_pressure": "p_out_slider",
    "inputs.mass_flow_rate": "mdot_slider",

    "inputs.stator_inlet_angle": "alpha1_slider",
    "inputs.stator_exit_angle": "alpha2_slider",
    "inputs.blade_velocity_ratio": "nu_slider",
    "inputs.degree_reaction": "R_slider",

    "inputs.radius_ratio_12": "rr_12_slider",
    "inputs.radius_ratio_23": "rr_23_slider",
    "inputs.radius_ratio_34": "rr_34_slider",

    "inputs.height_radius_ratio": "HR_inlet_slider",

    "inputs.zweiffel_stator": "Z_stator_slider",
    "inputs.zweiffel_rotor": "Z_rotor_slider",

    "inputs.loss_coeff_stator": "xi_stator_slider",
    "inputs.loss_coeff_rotor": "xi_rotor_slider",
}


# =========================
# Tabular reports
# =========================
def stage_performance_table(out):
    rows = [

        ("Total-to-total efficiency", out["eta_tt"], "-"),
        ("Total-to-static efficiency", out["eta_ts"], "-"),
        ("Flow coefficient", out["phi"], "-"),
        ("Loading coefficient", out["psi"], "-"),
        ("Blade velocity ratio", out["inputs.blade_velocity_ratio"], "-"),
        ("Specific speed", out["specific_speed"], "-"),
        ("Spouting velocity", out["v_0"], "m/s"),
        ("Blade speed at rotor exit", out["U"], "m/s"),
        ("Meridional velocity", out["v_m"], "m/s"),
        ("Rotational speed", out["RPM"], "rpm"),
        ("Isentropic power", out["power_isentropic"]/1e3, "kW"),
        ("Actual power (t-t)", out["power_actual_tt"]/1e3, "kW"),
        ("Actual power (t-s)", out["power_actual_ts"]/1e3, "kW"),
    ]

    return [
        {"Quantity": k, "Value": float(v), "Unit": u}
        for k, v, u in rows
    ]

def geometry_table(out):
    rows = []

    for comp in ["stator", "rotor"]:
        rows.extend([
            (comp.capitalize(), "Chord", 1e3 * out[f"{comp}.chord"], "mm"),
            (comp.capitalize(), "Height", 1e3 * out[f"{comp}.height"], "mm"),
            (comp.capitalize(), "Aspect ratio", out[f"{comp}.aspect_ratio"], "-"),
            (comp.capitalize(), "Spacing", 1e3 * out[f"{comp}.spacing"], "mm"),
            (comp.capitalize(), "Opening", 1e3 * out[f"{comp}.opening"], "mm"),
            (comp.capitalize(), "Number of blades", out[f"{comp}.N_blades"], "-"),
            (comp.capitalize(), "Solidity", out[f"{comp}.solidity"], "-"),
            (comp.capitalize(), "Zweifel coefficient", out[f"{comp}.zweiffel"], "-"),
        ])

    return [
        {
            "Row": section,
            "Quantity": name,
            "Value": float(val),
            "Unit": unit,
        }
        for section, name, val, unit in rows
    ]

def flow_stations_table(out):
    rows = []

    for i in range(0, 5):
        rows.append({
            "Station": i,
            "p [bar]": out[f"station_{i}.p"] / 1e5,
            "T [K]": out[f"station_{i}.T"],
            "ρ [kg/m³]": out[f"station_{i}.d"],
            "q [kg/m³]": out[f"station_{i}.q"],
            "v [m/s]": out[f"station_{i}.v"],
            "w [m/s]": out[f"station_{i}.w"],
            "Ma": out[f"station_{i}.Ma"],
            "α [deg]": out[f"station_{i}.alpha"],
            "β [deg]": out[f"station_{i}.beta"],
            "r [mm]": 1e3 * out[f"station_{i}.r"],
            "H [mm]": 1e3 * out[f"station_{i}.H"],
        })

    return rows


def make_table(data, columns, height=260):
    formatted_columns = []

    for c in columns:
        if c in ["Value", "p [bar]", "T [K]", "ρ [kg/m³]", "v [m/s]",
                 "w [m/s]", "Ma", "α [deg]", "β [deg]", "r [mm]", "H [mm]"]:
            formatted_columns.append({
                "name": c,
                "id": c,
                "type": "numeric",
                "format": Format(precision=3, scheme=Scheme.fixed),
            })
        else:
            formatted_columns.append({"name": c, "id": c})

    return dash_table.DataTable(
        data=data,
        columns=formatted_columns,
        style_table={
            "overflowX": "auto",
            "maxHeight": f"{height}px",
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




# =========================
# Layout
# =========================
controls = html.Div(
    style={"width": "30%", "padding": "30px"},
    children=[

        # # Save buttom
        # html.Button(
        #     "Save current design",
        #     id="save_button",
        #     n_clicks=0,
        #     style={
        #         "marginTop": "10px",
        #         "padding": "8px 16px",
        #         "fontWeight": "bold",
        #     },
        # ),
        # dcc.Download(id="download_meanline_yaml"),
        # dcc.Store(id="stage_result_store"),
        # html.Div(
        #     id="save_status",
        #     style={"marginTop": "8px", "fontSize": "13px"},
        # ),

        # # Load buttom
        # dcc.Upload(
        #     id="load_button",
        #     children=html.Button(
        #         "Load design",
        #         style={
        #             "marginTop": "10px",
        #             "padding": "8px 16px",
        #             "fontWeight": "bold",
        #         },
        #     ),
        #     accept=".yaml,.yml",
        # ),
        # dcc.Store(id="loaded_cfg_store"),


        # Save / Load buttons row
        html.Div(
            style={
                "display": "flex",
                "flexDirection": "row",
                "gap": "12px",
                "alignItems": "center",
                "marginBottom": "10px",
            },
            children=[

                html.Button(
                    "Save current design",
                    id="save_button",
                    n_clicks=0,
                    style={
                        "padding": "8px 16px",
                        "fontWeight": "bold",
                    },
                ),

                dcc.Upload(
                    id="load_button",
                    children=html.Button(
                        "Load previous design",
                        style={
                            "padding": "8px 16px",
                            "fontWeight": "bold",
                        },
                    ),
                    accept=".yaml,.yml",
                ),
            ],
        ),

        dcc.Download(id="download_meanline_yaml"),
        dcc.Store(id="stage_result_store"),
        dcc.Store(id="loaded_cfg_store"),

        html.Div(
            id="save_status",
            style={"marginTop": "8px", "fontSize": "13px"},
        ),


        # Input parameters
        html.H4("Stage type"),
        dcc.RadioItems(
            id="stage_type",
            options=[
                {"label": "Axial", "value": "axial"},
                {"label": "Radial", "value": "radial"},
            ],
            value="radial",
            inline=True,
            style={"marginBottom": "20px"},
        ),

        html.H4("Working fluid"),
        dcc.Dropdown(
            id="fluid_name",
            options=[
                {"label": f, "value": f}
                for f in sorted(CP.get_global_param_string("FluidsList").split(","))
            ],
            value="Air",          # sensible default
            clearable=False,
        ),

        html.H4("Operating conditions"),
        linked_input(["Inlet density, d", html.Sub("in"), " [kg/m", html.Sup("3"), "]"], "d_in", 0.001, 1e4, 0.001, defaults["d_in"]),
        linked_input(["Inlet pressure, p", html.Sub("in"), " [Pa]"], "p_in", 1.0, 1e9, 0.1, defaults["p_in"]),
        linked_input(["Exit pressure, p", html.Sub("out"), " [Pa]"], "p_out", 1.0, 1e9, 0.1, defaults["p_out"]),
        linked_input(["Mass flow rate, ṁ [kg/s]"], "mdot", 0.001, 1000.0, 0.001, defaults["mdot"]),

        html.H4("Design variables"),
        linked_input(["Stator inlet angle, α", html.Sub("1"), " [deg]"], "alpha1", -75.0, 75.0, 1.0, defaults["alpha1"]),
        linked_input(["Stator exit angle, α", html.Sub("2"), " [deg]"], "alpha2", 0.0, 90.0, 1.0, defaults["alpha2"]),
        linked_input(["Blade velocity ratio, ν"], "nu", 0.05, 2.0, 0.01, defaults["nu"]),
        linked_input(["Degree of reaction, R"], "R", 0.0, 1.0 - 1e-6, 0.01, defaults["R"]),
        linked_input(["Inlet height-to-radius ratio, H", html.Sub("1"), "/", "r", html.Sub("1")], "HR_inlet", 0.01, 2.0, 0.01, defaults["HR_inlet"]),
        linked_input(["Radius ratio, r", html.Sub("1"), "/", "r", html.Sub("2")], "rr_12", 0.10, 1.00, 0.001, defaults["rr_12"]),
        linked_input(["Radius ratio, r", html.Sub("2"), "/", "r", html.Sub("3")], "rr_23", 0.10, 1.00, 0.001, defaults["rr_23"]),
        linked_input(["Radius ratio, r", html.Sub("3"), "/", "r", html.Sub("4")], "rr_34", 0.10, 1.00, 0.001, defaults["rr_34"]),
        linked_input(["Zweifel (stator)"], "Z_stator", 0.1, 2.0, 0.01, defaults["Z_stator"]),
        linked_input(["Zweifel (rotor)"], "Z_rotor", 0.1, 2.0, 0.01, defaults["Z_rotor"]),

        html.H4("Loss coefficients"),
        linked_input(["Loss coefficient, ξ", html.Sub("stator")], "xi_stator", 0.0, 0.5, 0.005, defaults["xi_stator"]),
        linked_input(["Loss coefficient, ξ", html.Sub("rotor")], "xi_rotor", 0.0, 0.5, 0.005, defaults["xi_rotor"]),
    ],
)



plots = html.Div(
    style={
        "width": "70%",
        "padding": "20px",
        "display": "flex",
        "flexDirection": "column",
        "gap": "20px",
    },
    children=[
        # =====================
        # Row 1: geometry plots
        # =====================
        html.Div(
            style={
                "display": "flex",
                "flexDirection": "row",
                "gap": "20px",
            },
            children=[
                html.Div(
                    style={"flex": "1 1 50%"},
                    children=[
                        html.H3("Meridional channel"),
                        dcc.Graph(
                            id="meridional_plot",
                            style={"height": "350px"},
                        ),
                    ],
                ),
                html.Div(
                    style={"flex": "1 1 50%"},
                    children=[
                        html.H3("Blade-to-blade view"),
                        dcc.Graph(
                            id="blades_plot",
                            style={"height": "350px"},
                        ),
                    ],
                ),
            ],
        ),

        # ==========================
        # Row 2: performance plots
        # ==========================
        html.Div(
            style={
                "display": "flex",
                "flexDirection": "row",
                "gap": "20px",
            },
            children=[
                html.Div(
                    style={"flex": "1 1 60%"},
                    children=[
                        html.H3("Total-to-static efficiency"),
                        dcc.Graph(
                            id="efficiency_ts_plot",
                            style={"height": "350px"},
                        ),
                    ],
                ),
                html.Div(
                    style={"flex": "1 1 60%"},
                    children=[
                        html.H3("Total-to-total efficiency"),
                        dcc.Graph(
                            id="efficiency_tt_plot",
                            style={"height": "350px"},
                        ),
                    ],
                ),
            ],
        ),
    ],
)


tables = html.Div(
    style={
        "width": "70%",
        "padding": "20px",
        "display": "flex",
        "flexDirection": "column",
        "gap": "20px",
    },
    children=[
        # ==========================
        # Row 1: two tables side by side
        # ==========================
        html.Div(
            style={
                "display": "flex",
                "flexDirection": "row",
                "gap": "20px",
            },
            children=[
                html.Div(
                    style={"flex": "1 1 60%"},
                    children=[
                        html.H3("Stage performance"),
                        html.Div(id="perf_table"),
                    ],
                ),
                html.Div(
                    style={"flex": "1 1 60%"},
                    children=[
                        html.H3("Geometry summary"),
                        html.Div(id="geom_table"),
                    ],
                ),
            ],
        ),

        # ==========================
        # Row 2: full-width table
        # ==========================
        html.Div(
            style={
                "display": "flex",
                "flexDirection": "row",
                "gap": "20px",
            },
            children=[
                html.H3("Flow stations"),
                html.Div(id="stations_table"),
            ]
        ),
    ],
)


app.layout = html.Div(
    style={
        "fontFamily": "Arial",
        "display": "flex",
        "maxWidth": "1500px",
        "margin": "auto",
    },
    children=[
        controls,
        html.Div(
            style={"width": "70%"},
            children=[plots, tables],
        ),
    ],
)




# =========================
# Register slider-input sync
# =========================
# register_link_value("d_in", 0.001, 1e4)
# register_link_value("p_in", 1.0, 1e9)
# register_link_value("p_out", 1.0, 1e9)
# register_link_value("mdot", 0.001, 1000.0)

# register_link_value("alpha1", -75.0, 75.0)
# register_link_value("alpha2", 0.0, 90.0)
# register_link_value("nu", 0.05, 2.0)
# register_link_value("R", 1e-9, 1.0 - 1e-9)
# register_link_value("HR_inlet", 0.01, 2.00)
# register_link_value("rr_12", 0.10, 1.00)
# register_link_value("rr_23", 0.10, 1.00)
# register_link_value("rr_34", 0.10, 1.00)

# register_link_value("Z_stator", 0.1, 1.5)
# register_link_value("Z_rotor", 0.1, 1.5)

# register_link_value("xi_stator", 0.0, 0.5)
# register_link_value("xi_rotor", 0.0, 0.5)
# =========================
# Register slider–input sync + YAML load
# =========================

register_link_value(
    "d_in", 0.001, 1e4,
    yaml_key="inputs.inlet_density",
)

register_link_value(
    "p_in", 1.0, 1e9,
    yaml_key="inputs.inlet_pressure",
)

register_link_value(
    "p_out", 1.0, 1e9,
    yaml_key="inputs.exit_pressure",
)

register_link_value(
    "mdot", 0.001, 1000.0,
    yaml_key="inputs.mass_flow_rate",
)

register_link_value(
    "alpha1", -75.0, 75.0,
    yaml_key="inputs.stator_inlet_angle",
)

register_link_value(
    "alpha2", 0.0, 90.0,
    yaml_key="inputs.stator_exit_angle",
)

register_link_value(
    "nu", 0.05, 2.0,
    yaml_key="inputs.blade_velocity_ratio",
)

register_link_value(
    "R", 1e-9, 1.0 - 1e-9,
    yaml_key="inputs.degree_reaction",
)

register_link_value(
    "HR_inlet", 0.01, 2.0,
    yaml_key="inputs.height_radius_ratio",
)

register_link_value(
    "rr_12", 0.10, 1.00,
    yaml_key="inputs.radius_ratio_12",
)

register_link_value(
    "rr_23", 0.10, 1.00,
    yaml_key="inputs.radius_ratio_23",
)

register_link_value(
    "rr_34", 0.10, 1.00,
    yaml_key="inputs.radius_ratio_34",
)

register_link_value(
    "Z_stator", 0.1, 1.5,
    yaml_key="inputs.zweiffel_stator",
)

register_link_value(
    "Z_rotor", 0.1, 1.5,
    yaml_key="inputs.zweiffel_rotor",
)

register_link_value(
    "xi_stator", 0.0, 0.5,
    yaml_key="inputs.loss_coeff_stator",
)

register_link_value(
    "xi_rotor", 0.0, 0.5,
    yaml_key="inputs.loss_coeff_rotor",
)



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
    Input("d_in_slider", "value"),
    Input("p_in_slider", "value"),
    Input("p_out_slider", "value"),
    Input("mdot_slider", "value"),
    Input("alpha1_slider", "value"),
    Input("alpha2_slider", "value"),
    Input("nu_slider", "value"),
    Input("R_slider", "value"),
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
    d_in, p_in, p_out, mdot,
    alpha1, alpha2, nu, R,
    rr_12, rr_23, rr_34,
    HR_inlet, Z_stator, Z_rotor,
    xi_stator, xi_rotor,
):

    try:
        # -------------------------
        # Thermodynamic state
        # -------------------------
        fluid = jxp.Fluid(fluid_name, backend="HEOS")
        result = td.compute_stage_meanline(
            fluid=fluid,
            inlet_density=d_in,
            inlet_pressure=p_in,
            exit_pressure=p_out,
            mass_flow_rate=mdot,
            stator_inlet_angle=alpha1,
            stator_exit_angle=alpha2,
            blade_velocity_ratio=nu,
            degree_reaction=R,
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
        result_python = {k: to_python(v) for k, v in result.items()}


        # -------------------------
        # Geometry figures (your existing functions)
        # -------------------------
        fig_mer = td.plotly.plot_meridional_channel(result, fig_size=350)
        fig_bld = td.plotly.plot_blades(
            result,
            N_points=500,
            N_blades_plot=10,
            fig_size=350,
        )

        # -------------------------
        # Performance plots
        # -------------------------
        nu_vals = np.linspace(1e-9, 2.0, 200)
        R_list = [1e-9, 0.25, 0.5, 0.75, 1.0 - 1e-9]

        fig_ts = go.Figure()
        fig_tt = go.Figure()

        colors = [
            sample_colorscale("Magma", 0.2 + 0.6 * i / (len(R_list) - 1))[0]
            for i in range(len(R_list))
        ]
        for Ri, color in zip(R_list, colors):
            perf = td.compute_performance_stage(
                stator_inlet_angle=alpha1,
                stator_exit_angle=alpha2,
                degree_reaction=Ri,
                blade_velocity_ratio=nu_vals,
                radius_ratio_34=rr_34,
                loss_coeff_stator=xi_stator,
                loss_coeff_rotor=xi_rotor,
            )

            fig_ts.add_trace(
                go.Scatter(
                    x=nu_vals,
                    y=perf["eta_ts"],
                    mode="lines",
                    line=dict(color=color, width=2),
                    name=f"R = {Ri:.2f}",
                )
            )

            fig_tt.add_trace(
                go.Scatter(
                    x=nu_vals,
                    y=perf["eta_tt"],
                    mode="lines",
                    line=dict(color=color, width=2),
                    name=f"R = {Ri:.2f}",
                )
            )

        # -------------------------
        # Current operating point (marker)
        # -------------------------
        perf_now = td.compute_performance_stage(
            stator_inlet_angle=alpha1,
            stator_exit_angle=alpha2,
            degree_reaction=R,
            blade_velocity_ratio=np.array([nu]),
            radius_ratio_34=rr_34,
            loss_coeff_stator=xi_stator,
            loss_coeff_rotor=xi_rotor,
        )

        fig_ts.add_trace(
            go.Scatter(
                x=[nu],
                y=[perf_now["eta_ts"][0]],
                mode="markers",
                marker=dict(size=10, color="black"),
                name="Current",
            )
        )

        fig_tt.add_trace(
            go.Scatter(
                x=[nu],
                y=[perf_now["eta_tt"][0]],
                mode="markers",
                marker=dict(size=10, color="black"),
                name="Current",
            )
        )

        # -------------------------
        # Layout styling
        # -------------------------
        for fig, ylabel in zip(
            [fig_ts, fig_tt],
            ["Total-to-static efficiency", "Total-to-total efficiency"],
        ):
            fig.update_layout(
                template="simple_white",
                xaxis_title="Blade velocity ratio",
                yaxis_title=ylabel,
                yaxis_range=[0.0, 1.1],
                xaxis_range=[0.0, 2.0],
                margin=dict(l=70, r=20, t=30, b=60),
                legend=dict(
                    orientation="v",
                    x=0.98,
                    y=0.02,
                    xanchor="right",
                    yanchor="bottom",
                    bgcolor="rgba(255,255,255,0.85)",
                    bordercolor="black",
                    borderwidth=1,
                ),
            )

            fig.update_xaxes(
                showline=True,
                linewidth=1,
                linecolor="black",
                mirror=True,
            )

            fig.update_yaxes(
                showline=True,
                linewidth=1,
                linecolor="black",
                mirror=True,
    )


        perf_data = stage_performance_table(result)
        geom_data = geometry_table(result)
        stations_data = flow_stations_table(result)

        perf_table = make_table(perf_data, ["Quantity", "Value", "Unit"], height=300)
        geom_table = make_table(geom_data, ["Row", "Quantity", "Value", "Unit"], height=300)

        stations_table = make_table(
            stations_data,
            list(stations_data[0].keys()),
            height=400,
        )

        return fig_mer, fig_bld, fig_ts, fig_tt, perf_table, geom_table, stations_table, result_python


    except Exception as e:
        print("\n=== ERROR IN update_turbine CALLBACK ===")
        print(e)
        print("========================================\n")
        raise


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


@app.callback(
    Output("fluid_name", "value"),
    Output("stage_type", "value"),
    Input("loaded_cfg_store", "data"),
    State("fluid_name", "value"),
    State("stage_type", "value"),
    prevent_initial_call=True,
)
def apply_loaded_meta(cfg, fluid_cur, stage_cur):
    if not cfg:
        return fluid_cur, stage_cur

    fluid = cfg.get("inputs.fluid", fluid_cur)
    stage = cfg.get("inputs.stage_type", stage_cur)
    return fluid, stage
