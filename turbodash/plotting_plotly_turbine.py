import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.colors import sample_colorscale

from .geom_blade import (
    compute_blade_coordinates_radial,
    compute_blade_coordinates_cartesian,
)


# =============================================================================
# Styling constants (mirror of the mpl module's intent, plotly idiom)
# =============================================================================
COLOR_STATOR = "#ff7f0e"  # orange (matches mpl "tab:orange")
COLOR_ROTOR = "#1f77b4"  # blue   (matches mpl "tab:blue")
COLOR_INLET = "#1f77b4"  # rotor inlet  (blue)
COLOR_OUTLET = "#ff7f0e"  # rotor outlet (orange)
LINE_WIDTH = 1.6

AXIS_FONT_SIZE = 18
TICK_FONT_SIZE = 16
AXIS_LINE_WIDTH = 2
TICK_LENGTH = 6


def _style_axes(fig, *, xtitle, ytitle, equal=False, row=None, col=None):
    """Apply the shared simple_white axis styling used across the module."""
    xkw = dict(
        ticks="inside",
        ticklen=TICK_LENGTH,
        showline=True,
        mirror=True,
        linewidth=AXIS_LINE_WIDTH,
        linecolor="black",
        title_text=xtitle,
        title_font=dict(size=AXIS_FONT_SIZE),
        tickfont=dict(size=TICK_FONT_SIZE),
        tickangle=0,
    )
    ykw = dict(xkw)
    ykw["title_text"] = ytitle
    if equal:
        ykw["scaleanchor"] = "x"
        ykw["scaleratio"] = 1.0
    if row is not None:
        fig.update_xaxes(**xkw, row=row, col=col)
        fig.update_yaxes(**ykw, row=row, col=col)
    else:
        fig.update_xaxes(**xkw)
        fig.update_yaxes(**ykw)


# =============================================================================
# Plot turbine meridional channel (multistage)
# =============================================================================
def plot_turbine_meridional_channel(results):
    """
    Meridional channel for a full multistage turbine, axial or radial.
    Mirror of the mpl function of the same name.

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    stages = results["stages_performance"]
    turbine_type = results["inputs"]["turbine_type"]
    fig = go.Figure()

    def _line(x, y, color):
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                line=dict(color=color, width=LINE_WIDTH),
                showlegend=False,
            )
        )

    if turbine_type == "axial":
        intercascade_gap_factor = 2

        def draw_stage(out, x_offset):
            fs = out["flow_stations"]
            r = [fs[k]["r"] for k in range(4)]
            H = [fs[k]["H"] for k in range(4)]
            x1 = x_offset
            x2 = x1 + out["geometry"]["stator"]["chord_meridional"]
            x3 = x2 + out["geometry"]["stator"]["opening"] * intercascade_gap_factor
            x4 = x3 + out["geometry"]["rotor"]["chord_meridional"]
            hub = [r[k] - H[k] / 2 for k in range(4)]
            tip = [r[k] + H[k] / 2 for k in range(4)]
            _line([x1, x2], [hub[0], hub[1]], COLOR_STATOR)
            _line([x1, x2], [tip[0], tip[1]], COLOR_STATOR)
            _line([x1, x1], [hub[0], tip[0]], COLOR_STATOR)
            _line([x2, x2], [hub[1], tip[1]], COLOR_STATOR)
            _line([x3, x4], [hub[2], hub[3]], COLOR_ROTOR)
            _line([x3, x4], [tip[2], tip[3]], COLOR_ROTOR)
            _line([x3, x3], [hub[2], tip[2]], COLOR_ROTOR)
            _line([x4, x4], [hub[3], tip[3]], COLOR_ROTOR)
            return x4

        x_cursor, r_max = 0.0, 0.0
        for st in stages:
            x_exit = draw_stage(st, x_offset=x_cursor)
            last = st["flow_stations"][-1]
            r_max = max(r_max, last["r"] + last["H"] / 2)
            x_cursor = (
                x_exit + intercascade_gap_factor * st["geometry"]["rotor"]["opening"]
            )
        x_end = (
            x_cursor
            - intercascade_gap_factor * stages[-1]["geometry"]["rotor"]["opening"]
        )

        dx = x_end * 0.05
        fig.update_xaxes(range=[-dx, x_end + dx])
        fig.update_yaxes(range=[0.0, 1.05 * r_max])
        _style_axes(
            fig, xtitle="Axial direction", ytitle="Radial direction", equal=True
        )

    elif turbine_type == "radial":

        def draw_stage(out):
            fs = out["flow_stations"]
            r = [fs[k]["r"] for k in range(4)]
            H = [fs[k]["H"] for k in range(4)]
            xspan = [[-H[k] / 2, +H[k] / 2] for k in range(4)]
            _line(xspan[0], [r[0], r[0]], COLOR_STATOR)
            _line(xspan[1], [r[1], r[1]], COLOR_STATOR)
            _line([xspan[0][0], xspan[1][0]], [r[0], r[1]], COLOR_STATOR)
            _line([xspan[0][1], xspan[1][1]], [r[0], r[1]], COLOR_STATOR)
            _line(xspan[2], [r[2], r[2]], COLOR_ROTOR)
            _line(xspan[3], [r[3], r[3]], COLOR_ROTOR)
            _line([xspan[2][0], xspan[3][0]], [r[2], r[3]], COLOR_ROTOR)
            _line([xspan[2][1], xspan[3][1]], [r[2], r[3]], COLOR_ROTOR)

        r_min, r_max, x_min, x_max = np.inf, 0.0, 0.0, 0.0
        for st in stages:
            draw_stage(st)
            for fs in st["flow_stations"]:
                r_min = min(r_min, fs["r"])
                r_max = max(r_max, fs["r"])
                x_min = min(x_min, -fs["H"] / 2)
                x_max = max(x_max, fs["H"] / 2)

        x_pad = 0.2 * (x_max - x_min)
        r_pad = 0.2 * (r_max - r_min)
        fig.update_xaxes(range=[x_min - x_pad, x_max + x_pad])
        fig.update_yaxes(range=[0.0, r_max + r_pad])
        _style_axes(
            fig, xtitle="Spanwise direction", ytitle="Radial direction", equal=True
        )

    else:
        raise ValueError(f"Invalid stage type: {turbine_type!r}")

    fig.update_layout(template="simple_white", margin=dict(l=30, r=30, t=30, b=30))
    return fig


# =============================================================================
# Plot turbine blades (multistage)
# =============================================================================
def plot_turbine_blades(results, N_points=51, N_blades_plot=8):
    """
    Blade cascades for a full multistage turbine, axial or radial.
    Mirror of the mpl function of the same name.

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    stages = results["stages_performance"]
    turbine_type = results["inputs"]["turbine_type"]
    fig = go.Figure()

    # Track which legend entries have been shown so each appears once only.
    _legend_shown = {"Stator": False, "Rotor": False}

    def _blade_trace(x, y, color, label):
        show = not _legend_shown[label]
        _legend_shown[label] = True
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                line=dict(color=color, width=LINE_WIDTH),
                name=label,
                legendgroup=label,
                showlegend=show,
            )
        )

    if turbine_type == "axial":

        def draw_stage(out, x_offset):
            def draw_row(*, geom, x0, beta_in, beta_out, color, label):
                x_b, y_b, *_ = compute_blade_coordinates_cartesian(
                    camberline_type="linear_angle_change",
                    x1=x0,
                    y1=0.0,
                    beta1=np.deg2rad(beta_in),
                    beta2=np.deg2rad(beta_out),
                    chord_ax=geom["chord"],
                    loc_max=geom["maximum_thickness_location"],
                    thickness_max=geom["maximum_thickness"],
                    thickness_trailing=geom["trailing_edge_thickness"],
                    wedge_trailing=np.deg2rad(geom["trailing_edge_wedge_angle"]),
                    radius_leading=geom["leading_edge_radius"],
                    N_points=N_points,
                )
                for i in range(5 * N_blades_plot):
                    y_shift = (i - 10.5) * geom["spacing"]
                    _blade_trace(x_b, y_b + y_shift, color, label)

            stator_geom = out["geometry"]["stator"]
            rotor_geom = out["geometry"]["rotor"]
            draw_row(
                geom=stator_geom,
                x0=x_offset,
                beta_in=out["flow_stations"][0]["alpha"],
                beta_out=out["flow_stations"][1]["alpha"],
                color=COLOR_STATOR,
                label="Stator",
            )
            draw_row(
                geom=rotor_geom,
                x0=x_offset + stator_geom["chord"] + stator_geom["opening"],
                beta_in=out["flow_stations"][2]["beta"],
                beta_out=out["flow_stations"][3]["beta"],
                color=COLOR_ROTOR,
                label="Rotor",
            )
            return (
                x_offset
                + stator_geom["chord"]
                + stator_geom["opening"]
                + rotor_geom["chord"]
            )

        x_cursor = 0.0
        for st in stages:
            x_cursor = draw_stage(st, x_offset=x_cursor)
            x_cursor += st["geometry"]["rotor"]["opening"]

        y_top = N_blades_plot * stages[0]["geometry"]["stator"]["spacing"]
        fig.update_yaxes(range=[0, y_top])
        _style_axes(
            fig, xtitle="Axial direction", ytitle="Tangential direction", equal=True
        )

    elif turbine_type == "radial":

        def draw_stage(out):
            def draw_row(geom, color, label):
                x_b, y_b, *_ = compute_blade_coordinates_radial(
                    "linear_angle_change",
                    geom["radius_in"],
                    geom["radius_out"],
                    np.deg2rad(geom["metal_angle_in"]),
                    np.deg2rad(geom["metal_angle_out"]),
                    0.0,
                    geom["maximum_thickness_location"],
                    geom["maximum_thickness"],
                    geom["trailing_edge_thickness"],
                    np.deg2rad(geom["maximum_thickness_location"]),
                    geom["leading_edge_radius"],
                    N_points,
                )
                th = np.linspace(0.0, 2.0 * np.pi, 400)
                _line_black = dict(color="black", width=1.0)
                fig.add_trace(
                    go.Scatter(
                        x=geom["radius_in"] * np.cos(th),
                        y=geom["radius_in"] * np.sin(th),
                        mode="lines",
                        line=_line_black,
                        showlegend=False,
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=geom["radius_out"] * np.cos(th),
                        y=geom["radius_out"] * np.sin(th),
                        mode="lines",
                        line=_line_black,
                        showlegend=False,
                    )
                )

                N = int(round(geom["blade_count"]))
                dtheta = 2.0 * np.pi / N
                for i in range(N):
                    ct, st_ = np.cos(i * dtheta), np.sin(i * dtheta)
                    _blade_trace(
                        ct * x_b - st_ * y_b, st_ * x_b + ct * y_b, color, label
                    )

            draw_row(out["geometry"]["stator"], COLOR_STATOR, "Stator")
            draw_row(out["geometry"]["rotor"], COLOR_ROTOR, "Rotor")

        for st in stages:
            draw_stage(st)

        # r_max = 1.05 * stages[-1]["flow_stations"][-1]["r"]
        # fig.update_xaxes(range=[0.0, r_max])
        # fig.update_yaxes(range=[0.0, r_max])
        # _style_axes(fig, xtitle="x direction", ytitle="y direction", equal=True)

        r_max = 1.05 * stages[-1]["flow_stations"][-1]["r"]
        _style_axes(fig, xtitle="x direction", ytitle="y direction", equal=True)
        # Set ranges AFTER scaleanchor, and constrain the domain so the box holds.
        fig.update_xaxes(range=[0.0, r_max], constrain="domain")
        fig.update_yaxes(range=[0.0, r_max], constrain="domain")


    else:
        raise ValueError(f"Invalid stage type: {turbine_type!r}")

    fig.update_layout(template="simple_white", margin=dict(l=30, r=30, t=30, b=30))
    return fig


# =============================================================================
# Plot turbine velocity triangles (one stage per row, top to bottom)
# =============================================================================
def _get_velocity_components(fs, use_mach):
    """
    Tangential/meridional components of absolute and relative velocity at one
    station, as (v_t, v_m, w_t, w_m). Mach if use_mach else m/s.
    Mirror of the mpl helper.
    """
    norm = fs["a"] if use_mach else 1.0
    v, w = fs["v"] / norm, fs["w"] / norm
    alpha, beta = np.deg2rad(fs["alpha"]), np.deg2rad(fs["beta"])
    return (
        v * np.sin(alpha),
        v * np.cos(alpha),
        w * np.sin(beta),
        w * np.cos(beta),
    )


def _add_triangle(fig, v_t, v_m, w_t, w_m, color, show_legend, name):
    """
    Draw one velocity triangle on a single (non-subplot) figure: v and w from
    the shared origin, and the blade vector u closing from the tip of w to the
    tip of v.

    The meridional component (y) is stored with its true positive sign; the
    downward-pointing appearance is produced by inverting the y-axis range in
    the caller, not by negating the data. This way the tick labels read positive
    meridional Mach increasing downward.
    """

    def seg(x0, y0, x1, y1, legend):
        fig.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                line=dict(color=color, width=2),
                name=name,
                legendgroup=name,
                showlegend=legend,
            )
        )

    seg(0.0, 0.0, v_t, v_m, show_legend)  # absolute v (carries legend)
    seg(0.0, 0.0, w_t, w_m, False)  # relative w
    seg(w_t, w_m, v_t, v_m, False)  # blade u (closes triangle)


def plot_velocity_triangle_stage(st, scale_tangen, scale_merid, use_mach, title=None):
    """
    Build ONE self-contained figure with the rotor velocity triangles (inlet and
    outlet) of a single stage.

    This replaces the previous subplot-stacking approach: rather than packing all
    stages into one figure with shared axes (which fought the responsive GUI
    layout), each stage gets its own standalone figure. The caller stacks them in
    the page, one per row. Each figure carries its own legend and axis titles and
    is fully independent.

    Parameters
    ----------
    st : dict
        One entry of result["stages_performance"].
    scale_tangen, scale_merid : float
        Shared tangential/meridional magnitudes (max across all stages), so every
        stage's figure uses the SAME axis extents and the triangles are directly
        comparable from row to row.
    use_mach : bool
        True -> Mach numbers, False -> m/s.
    title : str, optional
        Title shown above the figure (e.g. "Stage 2").

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    if use_mach:
        xlabel, ylabel = "Tangential Mach [-]", "Meridional Mach [-]"
    else:
        xlabel, ylabel = "Tangential velocity [m/s]", "Meridional velocity [m/s]"

    v3_t, v3_m, w3_t, w3_m = _get_velocity_components(st["flow_stations"][2], use_mach)
    v4_t, v4_m, w4_t, w4_m = _get_velocity_components(st["flow_stations"][3], use_mach)

    fig = go.Figure()
    _add_triangle(fig, v3_t, v3_m, w3_t, w3_m, COLOR_INLET, True, "Rotor inlet")
    _add_triangle(fig, v4_t, v4_m, w4_t, w4_m, COLOR_OUTLET, True, "Rotor outlet")

    # Origin crosshairs.
    fig.add_hline(y=0.0, line=dict(color="black", width=0.5))
    fig.add_vline(x=0.0, line=dict(color="black", width=0.5))

    # Padding factors. y_top_frac gives a little headroom ABOVE the origin
    # (negative meridional side); pad gives generous room on the triangle side so
    # the heads/tips are never clipped at the axis edge.
    pad = 1.25
    y_top_frac = 0.25

    fig.update_xaxes(
        range=[-pad * scale_tangen, pad * scale_tangen],
        title_text=xlabel,
        title_font=dict(size=AXIS_FONT_SIZE),
        showline=True,
        mirror=True,
        linewidth=1,
        linecolor="black",
        ticks="inside",
        tickfont=dict(size=TICK_FONT_SIZE),
    )
    fig.update_yaxes(
        # Inverted axis: the FIRST element (top of the axis) is the small
        # negative-side headroom, the SECOND element (bottom) is the padded
        # positive meridional extent. Because element[0] < element[1] would
        # normally plot upward, we write them high-to-low so the axis descends:
        # i.e. top = +y_top_frac*..., bottom = -pad*... in display terms. The
        # explicit padded range (instead of autorange="reversed") guarantees the
        # triangle tips at +scale_merid are fully inside the frame.
        range=[pad * scale_merid, -y_top_frac * pad * scale_merid],
        title_text=ylabel,
        title_font=dict(size=AXIS_FONT_SIZE),
        showline=True,
        mirror=True,
        linewidth=1,
        linecolor="black",
        ticks="inside",
        tickfont=dict(size=TICK_FONT_SIZE),
    )

    fig.update_layout(
        template="simple_white",
        height=340,
        autosize=True,
        margin=dict(l=70, r=30, t=40 if title else 30, b=55),
        title=dict(text=title, font=dict(size=AXIS_FONT_SIZE)) if title else None,
        legend=dict(
            orientation="v",
            x=0.98,
            y=0.98,
            xanchor="right",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="black",
            borderwidth=1,
        ),
    )
    return fig


def plot_velocity_triangles_turbine(result, mode="mach"):
    """
    Rotor velocity triangles for every stage, as a LIST of standalone figures
    (one per stage), stage 1 first.

    The shared axis extents are computed once across all stages so the per-stage
    figures are directly comparable. The caller (GUI) places each figure in its
    own row.

    Parameters
    ----------
    result : dict
        Output of compute_turbine_performance().
    mode : {"mach", "velocity"}

    Returns
    -------
    list[plotly.graph_objects.Figure]
        One figure per stage, in stage order.
    """
    use_mach = {"mach": True, "velocity": False}.get(mode)
    if use_mach is None:
        raise ValueError(f"mode must be 'mach' or 'velocity', got {mode!r}")

    stages = result["stages_performance"]

    # Shared extents across all stages and both rotor stations, so every stage's
    # figure uses identical axis limits and the triangles are comparable.
    tangential_mags, meridional_mags = [], []
    for st in stages:
        for idx in (2, 3):
            v_t, v_m, w_t, w_m = _get_velocity_components(
                st["flow_stations"][idx], use_mach
            )
            tangential_mags += [abs(v_t), abs(w_t)]
            meridional_mags += [abs(v_m), abs(w_m)]
    scale_tangen = max((t for t in tangential_mags if np.isfinite(t)))
    scale_merid = max((m for m in meridional_mags if np.isfinite(m)))

    figs = []
    for i, st in enumerate(stages, 1):
        figs.append(
            plot_velocity_triangle_stage(
                st, scale_tangen, scale_merid, use_mach, title=f"Stage {i}"
            )
        )
    return figs


# =============================================================================
# Plot loss distribution (multistage stacked bars)
# =============================================================================
_LOSS_COMPONENTS = [
    "loss_profile",
    "loss_trailing",
    "loss_secondary",
    "loss_clearance",
]

_LOSS_LABELS = {
    "loss_profile": "Profile",
    "loss_trailing": "Trailing edge",
    "loss_secondary": "Secondary",
    "loss_clearance": "Tip clearance",
}

# Magma samples in [0.15, 0.85], matching the mpl module.
_pts = np.linspace(0.15, 0.85, len(_LOSS_COMPONENTS))
_LOSS_COLORS = {
    comp: sample_colorscale("Magma", float(p))[0]
    for comp, p in zip(_LOSS_COMPONENTS, _pts)
}


def plot_turbine_loss_distribution(results):
    """
    Stacked bar plot of the loss-coefficient breakdown for every blade row.
    One bar per blade row in flow order (S1, R1, S2, R2, ...). Mirror of the
    mpl function.

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    stages = results["stages_performance"]

    labels, row_losses = [], []
    for i, st in enumerate(stages, 1):
        labels.append(f"S{i}")
        row_losses.append(st["losses"]["stator"])
        labels.append(f"R{i}")
        row_losses.append(st["losses"]["rotor"])

    fig = go.Figure()
    for comp in _LOSS_COMPONENTS:
        heights = [float(rl.get(comp, 0.0)) for rl in row_losses]
        fig.add_trace(
            go.Bar(
                x=labels,
                y=heights,
                name=_LOSS_LABELS[comp],
                marker=dict(
                    color=_LOSS_COLORS[comp],
                    line=dict(color="black", width=0.75),
                ),
            )
        )

    totals = [sum(float(rl.get(c, 0.0)) for c in _LOSS_COMPONENTS) for rl in row_losses]
    y_top = 1.2 * max(totals) if totals else 1.0

    fig.update_layout(
        barmode="stack",
        template="simple_white",
        margin=dict(l=70, r=20, t=30, b=60),
        bargap=0.5,
        legend=dict(
            orientation="v",
            x=0.98,
            y=0.98,
            xanchor="right",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="black",
            borderwidth=1,
        ),
    )
    fig.update_xaxes(
        title_text="Blade row",
        title_font=dict(size=AXIS_FONT_SIZE),
        tickfont=dict(size=TICK_FONT_SIZE),
        showline=True,
        mirror=True,
        linewidth=AXIS_LINE_WIDTH,
        linecolor="black",
    )
    fig.update_yaxes(
        title_text="Loss coefficient [-]",
        range=[0.0, y_top],
        title_font=dict(size=AXIS_FONT_SIZE),
        tickfont=dict(size=TICK_FONT_SIZE),
        showline=True,
        mirror=True,
        linewidth=AXIS_LINE_WIDTH,
        linecolor="black",
        gridcolor="#D9D9D9",
    )
    return fig