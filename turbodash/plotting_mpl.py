import numpy as np
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from matplotlib.ticker import FormatStrFormatter

from .geom_blade import (
    compute_blade_coordinates_radial,
    compute_blade_coordinates_cartesian,
)


# =============================================================================
# Plot turbine geometry
# =============================================================================

COLOR_STATOR = "tab:orange"
COLOR_ROTOR = "tab:blue"


def plot_turbine_meridional_channel(results, ax=None):
    stages = results["stages_performance"]
    turbine_type = results["inputs"]["turbine_type"]

    def plot_axial(ax):
        intercascade_gap_factor = 2  # number openings to use as gap between stages

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
            ax.plot([x1, x2], [hub[0], hub[1]], color=COLOR_STATOR, lw=1.5)
            ax.plot([x1, x2], [tip[0], tip[1]], color=COLOR_STATOR, lw=1.5)
            ax.plot([x1, x1], [hub[0], tip[0]], color=COLOR_STATOR, lw=1.5)
            ax.plot([x2, x2], [hub[1], tip[1]], color=COLOR_STATOR, lw=1.5)
            ax.plot([x3, x4], [hub[2], hub[3]], color=COLOR_ROTOR, lw=1.5)
            ax.plot([x3, x4], [tip[2], tip[3]], color=COLOR_ROTOR, lw=1.5)
            ax.plot([x3, x3], [hub[2], tip[2]], color=COLOR_ROTOR, lw=1.5)
            ax.plot([x4, x4], [hub[3], tip[3]], color=COLOR_ROTOR, lw=1.5)
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
        ax.set_xlim(-dx, x_end + dx)
        ax.set_ylim(0.0, 1.05 * r_max)
        ax.set_xlabel("Axial direction")

    def plot_radial(ax):
        def draw_stage(out):
            fs = out["flow_stations"]
            r = [fs[k]["r"] for k in range(4)]
            H = [fs[k]["H"] for k in range(4)]
            xspan = [[-H[k] / 2, +H[k] / 2] for k in range(4)]
            ax.plot(xspan[0], [r[0], r[0]], color=COLOR_STATOR, lw=1.5)
            ax.plot(xspan[1], [r[1], r[1]], color=COLOR_STATOR, lw=1.5)
            ax.plot(
                [xspan[0][0], xspan[1][0]], [r[0], r[1]], color=COLOR_STATOR, lw=1.5
            )
            ax.plot(
                [xspan[0][1], xspan[1][1]], [r[0], r[1]], color=COLOR_STATOR, lw=1.5
            )
            ax.plot(xspan[2], [r[2], r[2]], color=COLOR_ROTOR, lw=1.5)
            ax.plot(xspan[3], [r[3], r[3]], color=COLOR_ROTOR, lw=1.5)
            ax.plot([xspan[2][0], xspan[3][0]], [r[2], r[3]], color=COLOR_ROTOR, lw=1.5)
            ax.plot([xspan[2][1], xspan[3][1]], [r[2], r[3]], color=COLOR_ROTOR, lw=1.5)

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
        ax.set_xlim(x_min - x_pad, x_max + x_pad)
        ax.set_ylim(0.0, r_max + r_pad)
        ax.set_xlabel("Spanwise direction")

    if ax is None:
        figsize = (6, 5)
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    if turbine_type == "axial":
        plot_axial(ax)
    elif turbine_type == "radial":
        plot_radial(ax)
    else:
        raise ValueError(f"Invalid stage type: {turbine_type!r}")

    ax.set_ylabel("Radial direction")
    ax.set_aspect("equal", adjustable="box")
    fig.tight_layout(pad=1)
    return fig


def plot_turbine_blades(results, ax=None, N_points=200, N_blades_plot=8):
    """
    Plot blade cascades for a full turbine, axial or radial.
    Stage type is read from the first stage's inputs.

    Parameters
    ----------
    results : dict
        Meanline results container, output of compute_turbine_performance().
    ax : matplotlib axis, optional
        Axis to draw on; created if None.
    N_points : int
        Points per blade camberline. Default 500.
    N_blades_plot : int
        Number of blades to plot per row (axial only). Default 4.

    Returns
    -------
    fig, ax
    """
    stages = results["stages_performance"]
    turbine_type = results["inputs"]["turbine_type"]

    def plot_axial(ax):
        def draw_stage(out, x_offset, stage_idx):
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
                    ax.plot(
                        x_b,
                        y_b + y_shift,
                        color=color,
                        lw=1.5,
                        label=label if (stage_idx == 0 and i == 0) else None,
                    )

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
        for j, st in enumerate(stages):
            x_cursor = draw_stage(st, x_offset=x_cursor, stage_idx=j)
            x_cursor += st["geometry"]["rotor"]["opening"]  # interstage gap

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Axial direction")
        ax.set_ylabel("Tangential direction")
        ax.set_ylim(0, N_blades_plot * stages[0]["geometry"]["stator"]["spacing"])

    def plot_radial(ax):
        def draw_stage(out, stage_idx):
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
                ax.plot(
                    geom["radius_in"] * np.cos(th),
                    geom["radius_in"] * np.sin(th),
                    "k-",
                    lw=1,
                )
                ax.plot(
                    geom["radius_out"] * np.cos(th),
                    geom["radius_out"] * np.sin(th),
                    "k-",
                    lw=1,
                )

                dtheta = 2.0 * np.pi / geom["blade_count"]
                for i in range(geom["blade_count"]):
                    rot = i * dtheta
                    ct, st = np.cos(rot), np.sin(rot)
                    ax.plot(
                        ct * x_b - st * y_b,
                        st * x_b + ct * y_b,
                        color=color,
                        lw=1.5,
                        label=label if (stage_idx == 0 and i == 0) else None,
                    )

            draw_row(out["geometry"]["stator"], color=COLOR_STATOR, label="Stator")
            draw_row(out["geometry"]["rotor"], color=COLOR_ROTOR, label="Rotor")

        for j, st in enumerate(stages):
            draw_stage(st, stage_idx=j)

        r_max = 1.05 * stages[-1]["flow_stations"][-1]["r"]
        ax.set_xlim(0.0, r_max)
        ax.set_ylim(0.0, r_max)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(r"$x$ direction")
        ax.set_ylabel(r"$y$ direction")

    if ax is None:
        figsize = (6, 5)
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    if turbine_type == "axial":
        plot_axial(ax)
    elif turbine_type == "radial":
        plot_radial(ax)
    else:
        raise ValueError(f"Invalid stage type: {turbine_type!r}")

    ax.legend()
    fig.tight_layout(pad=1)
    return fig


# =============================================================================
# Plot turbine velocity triangles (one stage per row, from top to bottom)
# =============================================================================

COLOR_INLET = "tab:blue"
COLOR_OUTLET = "tab:orange"


def _match_tick_decimals(ax):
    """
    Force the x and y tick labels to share the same number of decimal places
    (the larger of what each axis would use on its own), so the two axes read
    consistently. Call after the axis limits/ticks are set.
    """

    def n_decimals(ticks):
        d = 0
        for t in ticks:
            s = f"{t:.10f}".rstrip("0").rstrip(".")
            if "." in s:
                d = max(d, len(s.split(".")[1]))
        return d

    # Use the ticks that fall within the current view, since off-view ticks
    # can otherwise inflate the decimal count.
    x0, x1 = sorted(ax.get_xlim())
    y0, y1 = sorted(ax.get_ylim())
    xt = [t for t in ax.get_xticks() if x0 <= t <= x1]
    yt = [t for t in ax.get_yticks() if y0 <= t <= y1]

    decimals = max(n_decimals(xt), n_decimals(yt))
    fmt = FormatStrFormatter(f"%.{decimals}f")
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)


def _get_velocity_components(fs, use_mach):
    """
    Return the (tangential, meridional) components of the absolute and relative
    velocities at one flow station, as (v_t, v_m, w_t, w_m).

    Components are in Mach number if use_mach is True (divided by the local
    speed of sound), otherwise in m/s. Used by both the per-stage plot and the
    turbine-level scale computation so the convention lives in one place.
    """
    norm = fs["a"] if use_mach else 1.0
    v, w = fs["v"] / norm, fs["w"] / norm
    alpha, beta = np.deg2rad(fs["alpha"]), np.deg2rad(fs["beta"])
    return (
        v * np.sin(alpha),  # v_t
        v * np.cos(alpha),  # v_m
        w * np.sin(beta),  # w_t
        w * np.cos(beta),  # w_m
    )


def _plot_velocity_triangles_stage(
    out,
    ax,
    use_mach=True,
    show_legend=True,
):
    """
    Plot rotor inlet (station 3) and rotor outlet (station 4) velocity
    triangles, with:

      - x-axis = tangential component
      - y-axis = meridional component, pointing DOWNWARD (inverted y-axis)

    Triangle construction (per station): the absolute (v) and relative (w)
    vectors share a common origin at (0, 0); the blade (u) vector closes the
    triangle, running from the tip of w to the tip of v (purely tangential,
    since v = w + u). The inlet and outlet triangles share the same origin, so
    the blade vector is NOT shared between them.

    Colors: blue = rotor inlet (3), orange = rotor outlet (4). All vectors are
    solid; the triangle is identified by color.

    Axis labels are set by the caller (the turbine wrapper) so the same drawing
    works for Mach or m/s.

    Parameters
    ----------
    out : dict
        Per-stage result dict (one entry of result["stages"]).
    ax : matplotlib axis, optional
        If given, draw on it; otherwise create a new figure.
    use_mach : bool, optional
        Components in Mach (v/a) if True, else m/s. Default True.
    show_legend : bool, optional
        Draw the inlet/outlet legend on this panel. Default True.
    """

    # Components at the two rotor stations: (v_t, v_m, w_t, w_m).
    v3_t, v3_m, w3_t, w3_m = _get_velocity_components(out["flow_stations"][2], use_mach)
    v4_t, v4_m, w4_t, w4_m = _get_velocity_components(out["flow_stations"][3], use_mach)

    # Arrowhead sized relative to the plot extent, so it is visible in both Mach and m/s
    t_ext = max(abs(t) for t in (v3_t, w3_t, v4_t, w4_t) if np.isfinite(t))
    m_ext = max((m for m in (v3_m, w3_m, v4_m, w4_m, 0.0) if np.isfinite(m)))
    head = 0.02 * max(t_ext, m_ext)

    # Draw one triangle: v and w from the shared origin
    # Coordinates are (x=tangential, y=meridional).
    def draw_triangle(v_t, v_m, w_t, w_m, color):
        kw = dict(
            color=color,
            head_width=head,
            head_length=1.5 * head,
            length_includes_head=True,
            lw=1.5,
        )
        ax.arrow(0.0, 0.0, v_t, v_m, **kw)  # absolute v
        ax.arrow(0.0, 0.0, w_t, w_m, **kw)  # relative w
        ax.arrow(w_t, w_m, v_t - w_t, v_m - w_m, **kw)  # blade u (closes triangle)

    draw_triangle(v3_t, v3_m, w3_t, w3_m, COLOR_INLET)
    draw_triangle(v4_t, v4_m, w4_t, w4_m, COLOR_OUTLET)
    ax.axhline(0.0, color="k", lw=0.5)
    ax.axvline(0.0, color="k", lw=0.5)

    # Axis limits (set before inverting y)
    pad = 1.25
    y_top_frac = 0.25
    ax.set_xlim(-pad * t_ext, pad * t_ext)
    ax.set_ylim(-y_top_frac * pad * m_ext, pad * m_ext)

    # Invert y so the meridional component points downward.
    ax.invert_yaxis()
    ax.set_aspect("equal", adjustable="box")

    # Empty lines for the legend (one per triangle, so the color identifies the station)
    if show_legend:
        ax.plot([], [], color=COLOR_INLET, label="Rotor inlet")
        ax.plot([], [], color=COLOR_OUTLET, label="Rotor outlet")
        ax.legend(loc="upper right", fontsize=9)

    return ax


def plot_velocity_triangles_turbine(result, mode="mach"):
    """
    Plot the rotor velocity triangles of every stage in a single column,
    top-to-bottom in cascade order (stage 1 at the top). All panels use equal
    aspect and share identical xlim/ylim, so the stages are directly comparable.

    Layout: no per-panel titles, a single shared x-label at the bottom, a
    legend only on the first (top) panel, and minimal vertical spacing.

    Parameters
    ----------
    result : dict
        Meanline results container, output of compute_turbine_performance().
    mode : {"mach", "velocity"}
        Whether the axes show Mach number (v/a) or velocity in m/s.

    Returns
    -------
    fig, axes
    """
    use_mach = {"mach": True, "velocity": False}.get(mode)
    if use_mach is None:
        raise ValueError(f"mode must be 'mach' or 'velocity', got {mode!r}")
    if use_mach:
        xlabel, ylabel = "Tangential Mach [-]", "Meridional Mach [-]"
    else:
        xlabel, ylabel = "Tangential velocity [m/s]", "Meridional velocity [m/s]"

    stages = result["stages_performance"]
    n_stages = len(stages)

    # Shared extents across all stages and both rotor stations
    tangential_mags, meridional_mags = [], []
    for st in stages:
        for idx in (2, 3):  # rotor inlet (3) and outlet (4)
            v_t, v_m, w_t, w_m = _get_velocity_components(
                st["flow_stations"][idx], use_mach
            )
            tangential_mags += [abs(v_t), abs(w_t)]
            meridional_mags += [abs(v_m), abs(w_m)]
    scale_tangen = max((t for t in tangential_mags if np.isfinite(t)))
    scale_merid = max((m for m in meridional_mags if np.isfinite(m)))

    # Panel height matched to the data box aspect, so equal-aspect panels fill
    # their width instead of collapsing to a thin strip with blank space above
    # and below. The box spans 2*scale_tangen wide and (1 + y_top_frac)*scale_merid
    # tall (matching the limits set in the per-stage function). The +0.8 slack per
    # panel leaves room for tick labels, the legend and the shared x-label.
    y_top_frac = 0.4
    fig_w = 6.0
    box_aspect = (1.0 + y_top_frac) * scale_merid / (2.0 * scale_tangen)
    panel_h = fig_w * box_aspect
    fig_h = n_stages * panel_h + 0.8 * n_stages

    # Create the figure
    fig, axes = plt.subplots(
        nrows=n_stages,
        ncols=1,
        figsize=(fig_w, fig_h),
        squeeze=False,
    )
    axes = axes[:, 0]  # single column

    for i, (ax, st) in enumerate(zip(axes, stages)):
        _plot_velocity_triangles_stage(
            st,
            ax,
            use_mach=use_mach,
            show_legend=(i == 0),
        )
        _match_tick_decimals(ax)  # x and y share the same decimal count

    # rect bounds the axes block; center the super-labels on THAT block
    # (not the figure) so they align with the axes, then nudge them inward.
    rect = (0.05, 0.05, 1.0, 1.0)
    x_mid = 0.5 * (rect[0] + rect[2])  # horizontal center of the axes block
    y_mid = 0.5 * (rect[1] + rect[3])  # vertical center of the axes block
    fig.supxlabel(xlabel, size=14, x=x_mid, y=0.06)  # centered on axes, nudged up
    fig.supylabel(ylabel, size=14, y=y_mid, x=0.06)  # centered on axes, nudged right
    fig.tight_layout(pad=1.0, rect=rect)
    return fig


# =============================================================================
# Plot loss distribution
# =============================================================================


# Loss components to stack (bottom-to-top). Incidence is omitted: in the
# forward, isentropic-reconstruction setup it is identically zero, so it only
# adds clutter. loss_definition / loss_error are likewise excluded, and
# loss_total is not stacked (it would double-count) but overlaid as a check.
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

# Sample magma at evenly spaced points, trimming the near-black/near-white
# extremes (0.15..0.85) which read poorly as bar fills.
_magma = mpl.colormaps["magma"]
_pts = np.linspace(0.15, 0.85, len(_LOSS_COMPONENTS))
_LOSS_COLORS = {
    comp: mpl.colors.to_hex(_magma(p)) for comp, p in zip(_LOSS_COMPONENTS, _pts)
}


def plot_turbine_loss_distribution(results, ax=None):
    """
    Stacked bar plot of the loss-coefficient breakdown for every blade row.

    One bar per blade row in flow order (S1, R1, S2, R2, ...). Within each bar
    the loss components (profile, trailing edge, secondary, tip clearance) are
    stacked; their sum equals loss_total, overlaid as a thin check marker.

    Values are the kinetic-energy / enthalpy loss coefficients returned by the
    loss model (dimensionless), read from results[...]["losses"].

    Parameters
    ----------
    results : dict
        Meanline results container, output of compute_turbine_performance().
    ax : matplotlib axis, optional
        Axis to draw on; created if None.

    Returns
    -------
    fig, ax
    """
    stages = results["stages_performance"]

    # Columns in flow order: stator then rotor for each stage.
    labels, row_losses = [], []
    for i, st in enumerate(stages, 1):
        labels.append(f"S{i}")
        row_losses.append(st["losses"]["stator"])
        labels.append(f"R{i}")
        row_losses.append(st["losses"]["rotor"])

    n_cols = len(row_losses)
    x = np.arange(n_cols)
    bar_w = 0.5

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.get_figure()

    # Build the stack.
    bottoms = np.zeros(n_cols)
    for comp in _LOSS_COMPONENTS:
        heights = np.array([float(rl.get(comp, 0.0)) for rl in row_losses])
        ax.bar(
            x,
            heights,
            bar_w,
            bottom=bottoms,
            color=_LOSS_COLORS[comp],
            label=_LOSS_LABELS[comp],
            edgecolor="black",
            linewidth=0.75,
        )
        bottoms += heights

    # Cosmetics
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Blade row")
    ax.set_ylabel("Loss coefficient [-]")
    ax.set_ylim(0.0, 1.2 * float(np.nanmax(bottoms)))

    # Grid: off everywhere, then horizontal only, behind the bars.
    ax.set_axisbelow(True)
    ax.grid(False)
    ax.grid(True, axis="y", color="0.85", linewidth=0.5)

    ax.legend(
        loc="upper right",
        frameon=True,
        ncol=1,
        fontsize=11,
    )

    fig.tight_layout(pad=1)
    return fig
