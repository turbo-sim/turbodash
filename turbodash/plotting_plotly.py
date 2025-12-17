import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.colors import sample_colorscale

from .core import compute_performance_stage, compute_stage_meanline

from .geom_blade import (
    compute_blade_coordinates_radial,
    compute_blade_coordinates_cartesian,
)

# ---------------------------------------------------------------------
# Styling constants
# ---------------------------------------------------------------------
COLOR_STATOR = "#1f77b4"  # blue
COLOR_ROTOR = "#ff7f0e"  # orange
LINE_WIDTH = 1.6


# =====================================================================
# Meridional channel plots
# =====================================================================
def plot_meridional_channel(
    out,
):
    """
    Automatically plot meridional channel depending on stage_type.

    - stage_type == "axial"  -> axial meridional channel
    - stage_type == "radial" -> radial meridional channel

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """

    stage_type = out["inputs"].get("stage_type", None)

    if stage_type == "axial":
        return plot_meridional_channel_axial(out)

    elif stage_type == "radial":
        return plot_meridional_channel_radial(out)

    else:
        raise ValueError(
            f"Invalid or missing stage_type: {stage_type!r}. "
            "Expected 'axial' or 'radial'."
        )


def plot_meridional_channel_radial(
    out,
):
    """
    Standalone radial meridional channel plot with true 1:1 metric scaling.
    """

    fig = go.Figure()

    AXIS_FONT_SIZE = 18
    TICK_FONT_SIZE = 16
    AXIS_LINE_WIDTH = 2
    TICK_LENGTH = 6

    r1, r2, r3, r4 = (
        out["flow_stations"][1]["r"],
        out["flow_stations"][2]["r"],
        out["flow_stations"][3]["r"],
        out["flow_stations"][4]["r"],
    )

    H1, H2, H3, H4 = (
        out["flow_stations"][1]["H"],
        out["flow_stations"][2]["H"],
        out["flow_stations"][3]["H"],
        out["flow_stations"][4]["H"],
    )

    # -----------------------------
    # Stator (1 -> 2)
    # -----------------------------
    x1 = [-H1 / 2, +H1 / 2]
    x2 = [-H2 / 2, +H2 / 2]

    for x, y in [
        (x1, [r1, r1]),
        (x2, [r2, r2]),
        ([x1[0], x2[0]], [r1, r2]),
        ([x1[1], x2[1]], [r1, r2]),
    ]:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                line=dict(color=COLOR_STATOR, width=LINE_WIDTH),
                showlegend=False,
            )
        )

    # -----------------------------
    # Rotor (3 -> 4)
    # -----------------------------
    x3 = [-H3 / 2, +H3 / 2]
    x4 = [-H4 / 2, +H4 / 2]

    for x, y in [
        (x3, [r3, r3]),
        (x4, [r4, r4]),
        ([x3[0], x4[0]], [r3, r4]),
        ([x3[1], x4[1]], [r3, r4]),
    ]:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                line=dict(color=COLOR_ROTOR, width=LINE_WIDTH),
                showlegend=False,
            )
        )

    # -----------------------------
    # Axis ranges and scaling
    # -----------------------------
    H_max = max(H1, H2, H3, H4)
    r_max = 1.05 * r4

    fig.update_xaxes(
        range=[-1.5 * H_max, 1.5 * H_max],
        constrain="domain",
        ticks="inside",
        ticklen=TICK_LENGTH,
        showline=True,
        mirror=True,
        linewidth=AXIS_LINE_WIDTH,
        linecolor="black",
        title_text="Axial direction",
        title_font=dict(size=AXIS_FONT_SIZE),
        tickfont=dict(size=TICK_FONT_SIZE),
    )

    fig.update_yaxes(
        range=[0.0, r_max],
        scaleanchor="x",
        scaleratio=1.0,
        constrain="domain",
        ticks="inside",
        ticklen=TICK_LENGTH,
        showline=True,
        mirror=True,
        linewidth=AXIS_LINE_WIDTH,
        linecolor="black",
        title_text="Radial direction",
        title_font=dict(size=AXIS_FONT_SIZE),
        tickfont=dict(size=TICK_FONT_SIZE),
    )

    fig.update_xaxes(tickangle=0)
    fig.update_yaxes(tickangle=0)

    fig.update_layout(
        template="simple_white",
        margin=dict(l=30, r=30, t=30, b=30),
    )

    return fig


def plot_meridional_channel_axial(
    out,
):
    """
    Standalone axial meridional channel plot with true 1:1 metric scaling.
    """

    fig = go.Figure()

    AXIS_FONT_SIZE = 18
    TICK_FONT_SIZE = 16
    AXIS_LINE_WIDTH = 2
    TICK_LENGTH = 6

    r1, r2, r3, r4 = (
        out["flow_stations"][1]["r"],
        out["flow_stations"][2]["r"],
        out["flow_stations"][3]["r"],
        out["flow_stations"][4]["r"],
    )

    H1, H2, H3, H4 = (
        out["flow_stations"][1]["H"],
        out["flow_stations"][2]["H"],
        out["flow_stations"][3]["H"],
        out["flow_stations"][4]["H"],
    )

    x1 = 0.0
    x2 = x1 + out["geometry"]["stator"]["chord"]
    x3 = x2 + out["geometry"]["stator"]["opening"]
    x4 = x3 + out["geometry"]["rotor"]["chord"]

    def hub_tip(r, H):
        return r - H / 2, r + H / 2

    r1h, r1t = hub_tip(r1, H1)
    r2h, r2t = hub_tip(r2, H2)
    r3h, r3t = hub_tip(r3, H3)
    r4h, r4t = hub_tip(r4, H4)

    # -----------------------------
    # Stator
    # -----------------------------
    for x, y in [
        ([x1, x2], [r1h, r2h]),
        ([x1, x2], [r1t, r2t]),
        ([x1, x1], [r1h, r1t]),
        ([x2, x2], [r2h, r2t]),
    ]:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                line=dict(color=COLOR_STATOR, width=LINE_WIDTH),
                showlegend=False,
            )
        )

    # -----------------------------
    # Rotor
    # -----------------------------
    for x, y in [
        ([x3, x4], [r3h, r4h]),
        ([x3, x4], [r3t, r4t]),
        ([x3, x3], [r3h, r3t]),
        ([x4, x4], [r4h, r4t]),
    ]:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                line=dict(color=COLOR_ROTOR, width=LINE_WIDTH),
                showlegend=False,
            )
        )

    # -----------------------------
    # Axis ranges and scaling
    # -----------------------------
    y_max = 1.05 * max(r1 + H1, r2 + H2, r3 + H3, r4 + H4)
    x_max = (
        out["geometry"]["stator"]["chord"]
        + out["geometry"]["stator"]["opening"]
        + out["geometry"]["rotor"]["chord"]
    )
    dx = x_max * 0.1

    fig.update_xaxes(
        range=[0.0 - dx, x_max + dx],
        constrain="domain",
        ticks="inside",
        ticklen=TICK_LENGTH,
        showline=True,
        mirror=True,
        linewidth=AXIS_LINE_WIDTH,
        linecolor="black",
        title_text="Axial direction",
        title_font=dict(size=AXIS_FONT_SIZE),
        tickfont=dict(size=TICK_FONT_SIZE),
    )

    fig.update_yaxes(
        range=[0.0, y_max],
        scaleanchor="x",
        scaleratio=1.0,
        constrain="domain",
        ticks="inside",
        ticklen=TICK_LENGTH,
        showline=True,
        mirror=True,
        linewidth=AXIS_LINE_WIDTH,
        linecolor="black",
        title_text="Radius",
        title_font=dict(size=AXIS_FONT_SIZE),
        tickfont=dict(size=TICK_FONT_SIZE),
    )

    fig.update_xaxes(tickangle=0)
    fig.update_yaxes(tickangle=0)

    fig.update_layout(
        template="simple_white",
        margin=dict(l=30, r=30, t=30, b=30),
    )

    return fig


# =====================================================================
# Blade-to-blade plots
# =====================================================================
def plot_blades(
    out,
    *,
    N_points=500,
    N_blades_plot=10,
):
    """
    Automatically plot blade-to-blade view depending on stage_type.

    - stage_type == "axial"  -> axial blade-to-blade
    - stage_type == "radial" -> radial blade-to-blade

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """

    stage_type = out["inputs"].get("stage_type", None)

    if stage_type == "axial":
        return plot_blades_axial(
            out,
            N_points=N_points,
            N_blades_plot=N_blades_plot,
        )

    elif stage_type == "radial":
        return plot_blades_radial(
            out,
            N_points=N_points,
        )

    else:
        raise ValueError(
            f"Invalid or missing stage_type: {stage_type!r}. "
            "Expected 'axial' or 'radial'."
        )


def plot_blades_radial(
    out,
    N_points=500,
):
    """
    Standalone radial blade-to-blade plot with true 1:1 metric scaling.
    """

    fig = go.Figure()

    AXIS_FONT_SIZE = 18
    TICK_FONT_SIZE = 16
    AXIS_LINE_WIDTH = 2
    TICK_LENGTH = 6

    def add_circumference(fig, r, color, n_theta=400):
        theta = np.linspace(0.0, 2.0 * np.pi, n_theta)
        x = r * np.cos(theta)
        y = r * np.sin(theta)

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                line=dict(color=color, width=1.0),
                showlegend=False,
            )
        )

    def plot_row(geom, angle_in, angle_out, color):
        loc_max = geom["maximum_thickness_location"]
        r_in = geom["radius_in"]
        r_out = geom["radius_out"]
        t_max = geom["maximum_thickness"]
        r_le = geom["leading_edge_radius"]
        t_te = geom["trailing_edge_thickness"]
        wedge = np.deg2rad(geom["trailing_edge_wedge_angle"])

        x_b, y_b, *_ = compute_blade_coordinates_radial(
            "linear_angle_change",
            r_in,
            r_out,
            np.deg2rad(angle_in),
            np.deg2rad(angle_out),
            0.0,
            loc_max,
            t_max,
            t_te,
            wedge,
            r_le,
            N_points,
        )

        add_circumference(fig, r_in, color)
        add_circumference(fig, r_out, color)

        N = int(round(geom["blade_count"]))
        dtheta = 2.0 * np.pi / N

        for i in range(N):
            ct, st = np.cos(i * dtheta), np.sin(i * dtheta)
            X = ct * x_b - st * y_b
            Y = st * x_b + ct * y_b
            fig.add_trace(
                go.Scatter(
                    x=X,
                    y=Y,
                    mode="lines",
                    line=dict(color=color, width=LINE_WIDTH),
                    showlegend=False,
                )
            )

    # -------------------------------------------------
    # Plot stator and rotor
    # -------------------------------------------------
    plot_row(
        out["geometry"]["stator"],
        out["flow_stations"][1]["alpha"],
        out["flow_stations"][2]["alpha"],
        COLOR_STATOR,
    )

    plot_row(
        out["geometry"]["rotor"],
        out["flow_stations"][3]["beta"],
        out["flow_stations"][4]["beta"],
        COLOR_ROTOR,
    )

    # -------------------------------------------------
    # Axis ranges and scaling
    # -------------------------------------------------
    r_max = 1.05 * out["flow_stations"][4]["r"]

    fig.update_xaxes(
        range=[0.0, r_max],
        scaleanchor="y",
        constrain="domain",
        ticks="inside",
        ticklen=TICK_LENGTH,
        showline=True,
        mirror=True,
        linewidth=AXIS_LINE_WIDTH,
        linecolor="black",
        title_text="x direction",
        title_font=dict(size=AXIS_FONT_SIZE),
        tickfont=dict(size=TICK_FONT_SIZE),
    )

    fig.update_yaxes(
        range=[0.0, r_max],
        scaleanchor="x",
        scaleratio=1.0,
        constrain="domain",
        ticks="inside",
        ticklen=TICK_LENGTH,
        showline=True,
        mirror=True,
        linewidth=AXIS_LINE_WIDTH,
        linecolor="black",
        title_text="y direction",
        title_font=dict(size=AXIS_FONT_SIZE),
        tickfont=dict(size=TICK_FONT_SIZE),
    )

    fig.update_xaxes(tickangle=0)
    fig.update_yaxes(tickangle=0)

    fig.update_layout(
        template="simple_white",
        margin=dict(l=30, r=30, t=30, b=30),
    )

    return fig


def plot_blades_axial(
    out,
    N_points=500,
    N_blades_plot=10,
):
    """
    Standalone axial blade-to-blade plot with true 1:1 metric scaling.

    A square in data space appears square on screen.
    """

    fig = go.Figure()

    AXIS_FONT_SIZE = 18
    TICK_FONT_SIZE = 16
    AXIS_LINE_WIDTH = 2
    TICK_LENGTH = 6

    def plot_row(x0, geom, beta_in, beta_out, color):
        loc_max = geom["maximum_thickness_location"]
        chord = geom["chord"]
        spacing = geom["spacing"]
        opening = geom["opening"]
        t_max = geom["maximum_thickness"]
        r_le = geom["leading_edge_radius"]
        t_te = geom["trailing_edge_thickness"]
        wedge = np.deg2rad(geom["trailing_edge_wedge_angle"])

        x_b, y_b, *_ = compute_blade_coordinates_cartesian(
            "linear_angle_change",
            x0,
            0.0,
            np.deg2rad(beta_in),
            np.deg2rad(beta_out),
            chord,
            loc_max,
            t_max,
            t_te,
            wedge,
            r_le,
            N_points,
        )

        for i in range(N_blades_plot):
            fig.add_trace(
                go.Scatter(
                    x=x_b,
                    y=y_b + i * spacing,
                    mode="lines",
                    line=dict(color=color, width=LINE_WIDTH),
                    showlegend=False,
                )
            )

    # -------------------------------------------------
    # Plot stator and rotor
    # -------------------------------------------------
    plot_row(
        0.0,
        out["geometry"]["stator"],
        out["flow_stations"][1]["alpha"],
        out["flow_stations"][2]["alpha"],
        COLOR_STATOR,
    )

    plot_row(
        out["geometry"]["stator"]["chord"] + out["geometry"]["stator"]["opening"],
        out["geometry"]["rotor"],
        out["flow_stations"][3]["beta"],
        out["flow_stations"][4]["beta"],
        COLOR_ROTOR,
    )

    # -------------------------------------------------
    # Axis ranges and scaling
    # -------------------------------------------------
    x_max = (
        out["geometry"]["stator"]["chord"]
        + out["geometry"]["stator"]["opening"]
        + out["geometry"]["rotor"]["chord"]
    )

    pitch = out["geometry"]["stator"]["spacing"]
    y_center = 0.5 * (N_blades_plot - 1) * pitch

    dx = x_max * 0.1
    fig.update_xaxes(
        range=[0.0 - dx, x_max + dx],
        constrain="domain",
        ticks="inside",
        ticklen=TICK_LENGTH,
        showline=True,
        mirror=True,
        linewidth=AXIS_LINE_WIDTH,
        linecolor="black",
        title_text="Axial direction",
        title_font=dict(size=AXIS_FONT_SIZE),
        tickfont=dict(size=TICK_FONT_SIZE),
    )

    fig.update_yaxes(
        range=[y_center - 2 * pitch, y_center + pitch],
        scaleanchor="x",
        scaleratio=1.0,
        constrain="domain",
        ticks="inside",
        ticklen=TICK_LENGTH,
        showline=True,
        mirror=True,
        linewidth=AXIS_LINE_WIDTH,
        linecolor="black",
        title_text="Tangential direction",
        title_font=dict(size=AXIS_FONT_SIZE),
        tickfont=dict(size=TICK_FONT_SIZE),
    )

    fig.update_xaxes(tickangle=0)
    fig.update_yaxes(tickangle=0)

    fig.update_layout(
        template="simple_white",
        margin=dict(l=30, r=30, t=30, b=30),
    )

    return fig


# =====================================================================
# Core efficiency plotting function
# =====================================================================
def plot_efficiency(
    out,
    *,
    metric,
    nu_min=1e-9,
    nu_max=2.0,
    n_nu=200,
    R_list=None,
):
    """
    Plot an efficiency curve as a function of blade velocity ratio ν
    for multiple values of degree of reaction R.

    Parameters
    ----------
    out : dict
        Output dictionary from compute_stage_meanline
    metric : {"eta_ts", "eta_tt"}
        Efficiency metric to plot
    """

    if metric not in {"eta_ts", "eta_tt"}:
        raise ValueError(f"Invalid metric '{metric}'. Expected 'eta_ts' or 'eta_tt'.")

    inputs = out["inputs"]

    alpha1 = inputs["stator_inlet_angle"]
    alpha2 = inputs["stator_exit_angle"]
    nu0 = inputs["blade_velocity_ratio"]
    R0 = inputs["degree_reaction"]
    rr_34 = inputs["radius_ratio_34"]
    xi_stator = inputs["loss_coeff_stator"]
    xi_rotor = inputs["loss_coeff_rotor"]

    if R_list is None:
        R_list = [1e-9, 0.25, 0.5, 0.75, 1.0 - 1e-9]

    nu_vals = np.linspace(nu_min, nu_max, n_nu)

    fig = go.Figure()

    colors = [
        sample_colorscale("Magma", 0.2 + 0.6 * i / (len(R_list) - 1))[0]
        for i in range(len(R_list))
    ]

    # -------------------------------------------------
    # Efficiency curves
    # -------------------------------------------------
    for Ri, color in zip(R_list, colors):
        perf = compute_performance_stage(
            stator_inlet_angle=alpha1,
            stator_exit_angle=alpha2,
            degree_reaction=Ri,
            blade_velocity_ratio=nu_vals,
            radius_ratio_34=rr_34,
            loss_coeff_stator=xi_stator,
            loss_coeff_rotor=xi_rotor,
        )

        fig.add_trace(
            go.Scatter(
                x=nu_vals,
                y=perf[metric],
                mode="lines",
                line=dict(color=color, width=2),
                name=f"R = {Ri:.2f}",
            )
        )

    # -------------------------------------------------
    # Current operating point
    # -------------------------------------------------
    perf_now = compute_performance_stage(
        stator_inlet_angle=alpha1,
        stator_exit_angle=alpha2,
        degree_reaction=R0,
        blade_velocity_ratio=np.array([nu0]),
        radius_ratio_34=rr_34,
        loss_coeff_stator=xi_stator,
        loss_coeff_rotor=xi_rotor,
    )

    fig.add_trace(
        go.Scatter(
            x=[nu0],
            y=[perf_now[metric][0]],
            mode="markers",
            marker=dict(size=10, color="black"),
            name="Current",
        )
    )

    # -------------------------------------------------
    # Labels and styling
    # -------------------------------------------------
    ylabel = {
        "eta_ts": "Total-to-static efficiency",
        "eta_tt": "Total-to-total efficiency",
    }[metric]

    fig.update_layout(
        template="simple_white",
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
        title_text="Blade velocity ratio",
        range=[0.0, 2.0],
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=True,
    )

    fig.update_yaxes(
        title_text=ylabel,
        range=[0.0, 1.1],
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=True,
    )

    return fig


# =====================================================================
# Public convenience wrappers
# =====================================================================
def plot_eta_ts(out):
    """Plot total-to-static efficiency."""
    return plot_efficiency(out, metric="eta_ts")


def plot_eta_tt(out):
    """Plot total-to-total efficiency."""
    return plot_efficiency(out, metric="eta_tt")
