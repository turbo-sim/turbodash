import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, ctx
from plotly.colors import sample_colorscale
from turbodash import compute_performance_stage

# === initialize app ===
app = Dash(__name__)
server = app.server


def main():
    app.run(debug=False)


# === default values ===
default_params = dict(
    nu_lower=0.0,
    nu_upper=2.0,
    alpha1=0.0,
    alpha2=60.0,
    radius=0.9,
    xi_stator=0.05,
    xi_rotor=0.05,
)
R_list_default = [0.0, 0.25, 0.5, 0.75, 1 - 1e-6]


def linked_input(label_children, id_prefix, min_val, max_val, step, default):
    return html.Div(
        style={"margin-bottom": "20px"},
        children=[
            html.Label(label_children, style={"font-weight": "bold"}),
            html.Div(
                style={"display": "flex", "align-items": "center", "gap": "10px"},
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
                            updatemode="drag",  # 👈 update continuously while dragging
                        ),
                        style={"flex-grow": "1"},
                    ),
                    dcc.Input(
                        id=f"{id_prefix}_input",
                        type="number",
                        value=default,
                        min=min_val,
                        max=max_val,
                        step=step,
                        style={"width": "80px"},
                    ),
                ],
            ),
        ],
    )


# === define calculator layout (your existing code) ===
calculator_layout = html.Div(
    style={
        "font-family": "Arial",
        "display": "flex",
        "max-width": "1400px",
        "margin": "auto",
    },
    children=[
        # === left control panel ===
        html.Div(
            style={"width": "38%", "padding": "20px"},
            children=[
                html.H3("Stage parameters"),
                linked_input(
                    ["Blade velocity ratio, ν", html.Sub("min")],
                    "nu_lower",
                    0.0,
                    10.0,
                    0.01,
                    default_params["nu_lower"],
                ),
                linked_input(
                    ["Blade velocity ratio, ν", html.Sub("max")],
                    "nu_upper",
                    0.0,
                    10.0,
                    0.01,
                    default_params["nu_upper"],
                ),
                linked_input(
                    ["Stator inlet angle, α", html.Sub("1"), " [deg]"],
                    "alpha1",
                    -60.0,
                    60.0,
                    1.0,
                    default_params["alpha1"],
                ),
                linked_input(
                    ["Stator exit angle, α", html.Sub("2"), " [deg]"],
                    "alpha2",
                    0.0,
                    90.0,
                    1.0,
                    default_params["alpha2"],
                ),
                linked_input(
                    ["Radius ratio, r", html.Sub("2"), "/", "r", html.Sub("3")],
                    "radius",
                    0.0,
                    1.0,
                    0.01,
                    default_params["radius"],
                ),
                linked_input(
                    ["Loss coefficient, ξ", html.Sub("stator")],
                    "xi_stator",
                    0.0,
                    0.5,
                    0.01,
                    default_params["xi_stator"],
                ),
                linked_input(
                    ["Loss coefficient, ξ", html.Sub("rotor")],
                    "xi_rotor",
                    0.0,
                    0.5,
                    0.01,
                    default_params["xi_rotor"],
                ),
                html.Label(
                    ["Degree of reaction, ", html.I("R")], style={"font-weight": "bold"}
                ),
                dcc.Input(
                    id="R_values_input",
                    type="text",
                    value=", ".join([f"{R:.2f}" for R in R_list_default]),
                    style={"width": "100%", "margin-top": "5px"},
                    debounce=True,
                ),
            ],
        ),
        # === right plot ===
        html.Div(
            style={"width": "62%", "padding": "20px"},
            children=[
                html.H2("Turbine stage performance analysis"),
                html.Div(
                    children=[
                        dcc.Graph(
                            id="efficiency_ts_plot",
                            style={"height": "340px", "margin-bottom": "20px"},
                        ),
                        dcc.Graph(id="efficiency_tt_plot", style={"height": "340px"}),
                    ]
                ),
            ],
        ),
    ],
)

# === load your theory markdown (static doc) ===
import os

# Go one level up from the turbodash/ folder to root/, then into docs/
theory_path = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),  # root folder
    "docs",
    "documentation.md",
)

if os.path.exists(theory_path):
    with open(theory_path, "r", encoding="utf-8") as f:
        theory_md = f.read()
else:
    theory_md = "Theory file not found."


docs_markdown = (
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
)

app.layout = html.Div(
    [
        dcc.Tabs(
            id="tabs",
            value="calculator",
            children=[
                dcc.Tab(
                    label="Calculator", value="calculator", children=[calculator_layout]
                ),
                dcc.Tab(
                    label="Documentation",
                    value="docs",
                    children=[
                        html.Div(
                            docs_markdown,
                            style={"max-width": "1000px", "margin": "auto"},
                        )
                    ],
                ),
            ],
            colors={
                "border": "#007acc",
                "primary": "#007acc",
                "background": "#f9f9f9",
            },
            style={"fontFamily": "Arial", "fontSize": "16px", "font-weight": "bold"},
        )
    ]
)


# === sync sliders and inputs (with range enforcement) ===
def link_value(prefix, min_val, max_val):
    @app.callback(
        Output(f"{prefix}_slider", "value"),
        Output(f"{prefix}_input", "value"),
        Input(f"{prefix}_slider", "value"),
        Input(f"{prefix}_input", "value"),
    )
    def sync(val_slider, val_input):
        trigger = ctx.triggered_id
        val_slider = np.clip(
            val_slider if val_slider is not None else min_val, min_val, max_val
        )
        val_input = np.clip(
            val_input if val_input is not None else min_val, min_val, max_val
        )
        if trigger == f"{prefix}_slider":
            return val_slider, val_slider
        elif trigger == f"{prefix}_input":
            return val_input, val_input
        return val_slider, val_input


# register sync callbacks
link_value("nu_lower", 1e-12, 10.0)
link_value("nu_upper", 1e-12, 10.0)
link_value("alpha1", -60.0, 60.0)
link_value("alpha2", 0.0, 90.0)
link_value("radius", 0.0, 1.0)
link_value("xi_stator", 0.0, 0.5)
link_value("xi_rotor", 0.0, 0.5)


# === update plots callback ===
@app.callback(
    Output("efficiency_ts_plot", "figure"),
    Output("efficiency_tt_plot", "figure"),
    Input("alpha1_slider", "value"),
    Input("alpha2_slider", "value"),
    Input("radius_slider", "value"),
    Input("xi_stator_slider", "value"),
    Input("xi_rotor_slider", "value"),
    Input("nu_lower_slider", "value"),
    Input("nu_upper_slider", "value"),
    Input("R_values_input", "value"),
)
def update_plots(
    alpha1, alpha2, radius, xi_stator, xi_rotor, nu_lower, nu_upper, R_values_input
):

    # --- Parse and sanitize R values ---
    if isinstance(R_values_input, str):
        # Split by comma or space and convert to floats
        try:
            R_list = np.array(
                [
                    float(r.strip())
                    for r in R_values_input.replace(";", ",").split(",")
                    if r.strip() != ""
                ]
            )
        except ValueError:
            R_list = np.array([0.0, 0.25, 0.5, 0.75, 1.0 - 1e-9])
    else:
        # fallback (shouldn't happen)
        R_list = np.array([0.0, 0.25, 0.5, 0.75, 1.0 - 1e-9])

    # Clamp all values to [0.0, 1 - 1e-9]
    R_list = np.clip(R_list, 0.0, 1.0 - 1e-9)

    # Ensure unique, sorted order (optional but nice)
    R_list = np.unique(R_list)

    # generate ν array
    n_points = 200
    nu_vals = np.linspace(nu_lower, nu_upper, n_points)

    fig_ts = go.Figure()
    fig_tt = go.Figure()

    colors = [
        sample_colorscale("Magma", 0.2 + 0.6 * i / (len(R_list) - 1))[0]
        for i in range(len(R_list))
    ]

    for R, color in zip(R_list, colors):
        results = compute_performance_stage(
            stator_inlet_angle=alpha1,
            stator_exit_angle=alpha2,
            degree_reaction=R,
            blade_velocity_ratio=nu_vals,
            radius_ratio_34=radius,
            loss_coeff_stator=xi_stator,
            loss_coeff_rotor=xi_rotor,
        )
        # first plot: total-to-static efficiency
        fig_ts.add_trace(
            go.Scatter(
                x=nu_vals,
                y=results["eta_ts"],
                mode="lines",
                line=dict(color=color, width=2),
                name=f"R={R:.2f}",
            )
        )
        # second plot: total-to-total efficiency
        if "eta_tt" in results:
            fig_tt.add_trace(
                go.Scatter(
                    x=nu_vals,
                    y=results["eta_tt"],
                    mode="lines",
                    line=dict(color=color, width=2),
                    name=f"R={R:.2f}",
                )
            )

    # === Layouts ===
    for fig, ylabel in zip(
        [fig_ts, fig_tt],
        ["Total-to-static efficiency", "Total-to-total efficiency"],
    ):
        fig.update_layout(
            xaxis_title="Blade velocity ratio",
            yaxis_title=ylabel,
            template="simple_white",
            showlegend=True,
            legend=dict(
                x=1.02,
                y=1.0,
                xanchor="left",
                yanchor="top",
                bgcolor="rgba(255,255,255,0.8)",
            ),
            margin=dict(l=80, r=20, t=30, b=60),
        )
        fig.update_xaxes(
            range=[0, max(2, nu_upper)],
            showline=True,  # box edges
            linewidth=1,
            linecolor="black",
            mirror=True,
            showgrid=True,
            gridcolor="lightgray",
            gridwidth=0.5,
        )
        fig.update_yaxes(
            range=[0, 1.1],
            showline=True,
            linewidth=1,
            linecolor="black",
            mirror=True,
            showgrid=True,
            gridcolor="lightgray",
            gridwidth=0.5,
        )

    return fig_ts, fig_tt
