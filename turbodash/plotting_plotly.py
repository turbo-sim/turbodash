import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .geom_blade import (
    compute_blade_coordinates_radial,
    compute_blade_coordinates_cartesian,
)

# ---------------------------------------------------------------------
# Styling constants
# ---------------------------------------------------------------------
COLOR_STATOR = "#1f77b4"   # blue
COLOR_ROTOR  = "#ff7f0e"   # orange
LINE_WIDTH = 1.6

# =====================================================================
# Meridional channel plots
# =====================================================================

def plot_meridional_channel(
    out,
    *,
    fig_size=600,
):
    """
    Automatically plot meridional channel depending on stage_type.

    - stage_type == "axial"  -> axial meridional channel
    - stage_type == "radial" -> radial meridional channel

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """

    stage_type = out.get("stage_type", None)

    if stage_type == "axial":
        return plot_meridional_channel_axial(
            out,
            fig_size=fig_size,
        )

    elif stage_type == "radial":
        return plot_meridional_channel_radial(
            out,
            fig_size=fig_size,
        )

    else:
        raise ValueError(
            f"Invalid or missing stage_type: {stage_type!r}. "
            "Expected 'axial' or 'radial'."
        )

def plot_meridional_channel_radial(
    out,
    *,
    fig_size=600,
):
    """
    Standalone radial meridional channel plot with true 1:1 metric scaling.
    """

    fig = go.Figure()

    r1, r2, r3, r4 = (
        out["station_1.r"],
        out["station_2.r"],
        out["station_3.r"],
        out["station_4.r"],
    )

    H1, H2, H3, H4 = (
        out["station_1.H"],
        out["station_2.H"],
        out["station_3.H"],
        out["station_4.H"],
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

    fig.update_xaxes(range=[-1.5 * H_max, 1.5 * H_max])
    fig.update_yaxes(
        range=[0.0, r_max],
        scaleanchor="x",
        scaleratio=1.0,
    )

    # -----------------------------
    # Layout and styling
    # -----------------------------
    fig.update_layout(
        width=fig_size,
        height=fig_size,
        template="simple_white",
        margin=dict(l=40, r=40, t=40, b=40),
        # title_text="Radial meridional channel",
    )

    fig.update_xaxes(
        title_text="Axial direction",
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=True,
    )

    fig.update_yaxes(
        title_text="Radial direction",
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=True,
    )

    return fig

def plot_meridional_channel_axial(
    out,
    *,
    fig_size=600,
):
    """
    Standalone axial meridional channel plot with true 1:1 metric scaling.
    """

    fig = go.Figure()

    r1, r2, r3, r4 = (
        out["station_1.r"],
        out["station_2.r"],
        out["station_3.r"],
        out["station_4.r"],
    )

    H1, H2, H3, H4 = (
        out["station_1.H"],
        out["station_2.H"],
        out["station_3.H"],
        out["station_4.H"],
    )

    x1 = 0.0
    x2 = x1 + out["stator.chord"]
    x3 = x2 + out["stator.opening"]
    x4 = x3 + out["rotor.chord"]

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
    x_max = x4
    y_max = 1.05 * max(r1 + H1, r2 + H2, r3 + H3, r4 + H4)

    fig.update_xaxes(range=[0.0, x_max])
    fig.update_yaxes(
        range=[0.0, y_max],
        scaleanchor="x",
        scaleratio=1.0,
    )

    # -----------------------------
    # Layout and styling
    # -----------------------------
    fig.update_layout(
        width=fig_size,
        height=fig_size,
        template="simple_white",
        margin=dict(l=40, r=40, t=40, b=40),
        # title_text="Axial meridional channel",
    )

    fig.update_xaxes(
        title_text="Axial direction",
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=True,
    )

    fig.update_yaxes(
        title_text="Radius",
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=True,
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
    fig_size=600,
):
    """
    Automatically plot blade-to-blade view depending on stage_type.

    - stage_type == "axial"  -> axial blade-to-blade
    - stage_type == "radial" -> radial blade-to-blade

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """

    stage_type = out.get("stage_type", None)

    if stage_type == "axial":
        return plot_blades_axial(
            out,
            N_points=N_points,
            N_blades_plot=N_blades_plot,
            fig_size=fig_size,
        )

    elif stage_type == "radial":
        return plot_blades_radial(
            out,
            N_points=N_points,
            fig_size=fig_size,
        )

    else:
        raise ValueError(
            f"Invalid or missing stage_type: {stage_type!r}. "
            "Expected 'axial' or 'radial'."
        )



def plot_blades_radial(
    out,
    N_points=500,
    fig_size=600,
):
    """
    Standalone radial  blade-to-blade plot with true 1:1 metric scaling.
    """

    fig = go.Figure()

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

    def plot_row(r_in, r_out, a_in, a_out, N, color):
        loc_max = 0.30
        chord = r_out - r_in
        t_max = 0.30 * chord
        r_le = 0.50 * t_max
        t_te = 0.02 * chord
        wedge = np.deg2rad(10.0)

        x_b, y_b, *_ = compute_blade_coordinates_radial(
            "linear_angle_change",
            r_in,
            r_out,
            np.deg2rad(a_in),
            np.deg2rad(a_out),
            0.0,
            loc_max,
            t_max,
            t_te,
            wedge,
            r_le,
            N_points,
        )

        # ---- annulus boundaries
        add_circumference(fig, r_in, color)
        add_circumference(fig, r_out, color)

        N = int(round(N))
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
                ),
            )

    # -------------------------------------------------
    # Plot stator and rotor
    # -------------------------------------------------
    plot_row(
        out["station_1.r"],
        out["station_2.r"],
        out["station_1.alpha"],
        out["station_2.alpha"],
        out["stator.N_blades"],
        COLOR_STATOR,
    )

    plot_row(
        out["station_3.r"],
        out["station_4.r"],
        out["station_3.beta"],
        out["station_4.beta"],
        out["rotor.N_blades"],
        COLOR_ROTOR,
    )

    # -------------------------------------------------
    # Axis ranges (physically meaningful window)
    # -------------------------------------------------
    r_max = 1.05 * out["station_4.r"]

    fig.update_xaxes(range=[0, r_max])
    fig.update_yaxes(
        range=[0, r_max],
        scaleanchor="x",
        scaleratio=1.0,
    )

    # -------------------------------------------------
    # Layout and styling (identical philosophy to axial)
    # -------------------------------------------------
    fig.update_layout(
        width=fig_size,
        height=fig_size,
        template="simple_white",
        margin=dict(l=40, r=40, t=40, b=40),
        # title_text="Radial blade-to-blade view",
    )

    fig.update_xaxes(
        title_text="x direction",
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=True,
    )

    fig.update_yaxes(
        title_text="y direction",
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=True,
    )

    return fig




def plot_blades_axial(
    out,
    N_points=500,
    N_blades_plot=10,
    fig_size=600,
):
    """
    Standalone axial blade-to-blade plot with true 1:1 metric scaling.

    A square in data space appears square on screen.
    """

    fig = go.Figure()

    # -------------------------------------------------
    # Blade geometry helper
    # -------------------------------------------------
    def plot_row(x0, b1, b2, chord, spacing, opening, color):
        loc_max = 0.30
        t_max = 0.30 * chord
        r_le = 0.50 * t_max
        t_te = 0.1 * opening
        wedge = np.deg2rad(5.0)

        x_b, y_b, *_ = compute_blade_coordinates_cartesian(
            "linear_angle_change",
            x0,
            0.0,
            np.deg2rad(b1),
            np.deg2rad(b2),
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
        out["station_1.alpha"],
        out["station_2.alpha"],
        out["stator.chord"],
        out["stator.spacing"],
        out["stator.opening"],
        COLOR_STATOR,
    )

    plot_row(
        out["stator.chord"] + out["stator.opening"],
        out["station_3.beta"],
        out["station_4.beta"],
        out["rotor.chord"],
        out["rotor.spacing"],
        out["rotor.opening"],
        COLOR_ROTOR,
    )

    # -------------------------------------------------
    # Axis ranges (physically meaningful window)
    # -------------------------------------------------
    x_min = 0.0
    x_max = (
        out["stator.chord"]
        + out["stator.opening"]
        + out["rotor.chord"]
    )

    pitch = out["stator.spacing"]

    y_center = 0.5 * (N_blades_plot - 1) * pitch
    y_min = y_center - 2 * pitch
    y_max = y_center + 1 * pitch

    # -------------------------------------------------
    # Apply the SAME commands that made the square square
    # -------------------------------------------------
    fig.update_xaxes(range=[x_min, x_max])

    fig.update_yaxes(
        range=[y_min, y_max],
        scaleanchor="x",
        scaleratio=1.0,
    )

    fig.update_layout(
        width=fig_size,
        height=fig_size,
        template="simple_white",
        margin=dict(l=40, r=40, t=40, b=40),
        # title_text="Axial blade-to-blade view",
    )

    fig.update_xaxes(
        title_text="Axis direction",
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=True,
    )

    fig.update_yaxes(
        title_text="Tangential direction",
        showline=True,
        linewidth=1,
        linecolor="black",
        mirror=True,
    )

    return fig
