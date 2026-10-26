import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import PchipInterpolator
from plotly.subplots import make_subplots
from turbodash.core.turbine import get_blade_row_geometry
from turbodash.core.interspaces import get_interspace_geometry
from turbodash.core.diffuser import diffuser_settings

from turbodash.geometry.meanline import state_for_stage_row, section_from_state
from turbodash.geometry.blade import section_coordinates

# =============================================================================
# Styling constants (mirror of the mpl module's intent, plotly idiom)
# =============================================================================
COLOR_STATOR = "#ff7f0e"  # orange (matches mpl "tab:orange")
COLOR_ROTOR = "#1f77b4"  # blue   (matches mpl "tab:blue")
COLOR_INLET = COLOR_STATOR  # rotor inlet  (blue)
COLOR_OUTLET = COLOR_ROTOR  # rotor outlet (orange)
LINE_WIDTH = 1.6

AXIS_FONT_SIZE = 18
TICK_FONT_SIZE = 16
AXIS_LINE_WIDTH = 2
TICK_LENGTH = 6

CHANNEL_EXTENSION_CHORDS = 1.5
_ROW_FILLS = {"Stator": "rgba(255,127,14,0.16)", "Rotor": "rgba(31,119,180,0.18)"}


_PLOT_THEMES = {
    "light": dict(
        surface="#FFFFFF",
        text="#23364D",
        axis="#9AAABD",
        grid="#E3E9F1",
        wall="#647892",
        solid="rgba(100,120,146,0.12)",
        channel="rgba(36,98,209,0.035)",
        guide="rgba(68,118,188,0.48)",
    ),
    "dark": dict(
        surface="#17212E",
        text="#E2E8F1",
        axis="#586A80",
        grid="#2C3B4D",
        wall="#A0B2CA",
        solid="rgba(160,178,202,0.14)",
        channel="rgba(109,161,255,0.055)",
        guide="rgba(128,183,255,0.58)",
    ),
}


def apply_plot_theme(fig, theme="light"):
    """Restyle an existing figure without changing scientific data or axis ranges.

    The app can call this on retained results when switching themes, without
    recalculating the turbine. Colored series retain their scientific identity;
    formerly black/white framing and reference lines follow the selected theme.
    """
    if theme not in _PLOT_THEMES:
        raise ValueError("theme must be 'light' or 'dark'.")
    palette = _PLOT_THEMES[theme]
    fig.update_layout(
        template="plotly_dark" if theme == "dark" else "simple_white",
        paper_bgcolor=palette["surface"],
        plot_bgcolor=palette["surface"],
        font=dict(color=palette["text"], family="Arial, sans-serif"),
        legend=dict(
            bgcolor=palette["surface"],
            bordercolor=palette["axis"],
            font=dict(color=palette["text"]),
            borderwidth=0,
        ),
        hoverlabel=dict(
            bgcolor=palette["surface"],
            bordercolor=palette["axis"],
            font=dict(color=palette["text"]),
        ),
    )
    for update in (fig.update_xaxes, fig.update_yaxes):
        update(
            linecolor=palette["axis"],
            gridcolor=palette["grid"],
            zerolinecolor=palette["axis"],
            color=palette["text"],
            linewidth=1,
            title_font=dict(size=14),
            tickfont=dict(size=12),
        )
    scene_axis = dict(
        backgroundcolor=palette["surface"],
        color=palette["text"],
        gridcolor=palette["grid"],
        zerolinecolor=palette["axis"],
    )
    fig.update_scenes(
        bgcolor=palette["surface"], xaxis=scene_axis, yaxis=scene_axis, zaxis=scene_axis
    )
    # Use metadata for endwalls so repeated light/dark toggles remain reversible.
    for trace in fig.data:
        role = trace.meta.get("theme_role") if isinstance(trace.meta, dict) else None
        if role == "endwall":
            trace.line.color = palette["wall"]
        elif role == "solid":
            trace.fillcolor = palette["solid"]
        elif role == "channel":
            trace.fillcolor = palette["channel"]
        elif role == "flow_guide":
            trace.line.color = palette["guide"]
        elif role == "efficiency_tt":
            trace.line.color = "#69a9e2" if theme == "dark" else COLOR_ETA_TT
        elif role == "efficiency_ts":
            trace.line.color = "#ffad5c" if theme == "dark" else COLOR_ETA_TS
        elif role == "efficiency_recovery":
            trace.line.color = "#75ce9a" if theme == "dark" else COLOR_ETA_RECOVERY
        elif role in _LOSS_COLORS:
            trace.marker.color = (
                _LOSS_COLORS_DARK if theme == "dark" else _LOSS_COLORS
            )[role]
            trace.marker.line.color = palette["surface"]
        for part_name in ("line", "marker"):
            part = getattr(trace, part_name, None)
            color = getattr(part, "color", None)
            neutral = {"black", "#000000", *[p["text"] for p in _PLOT_THEMES.values()]}
            if isinstance(color, str) and color in neutral:
                part.color = palette["text"]
            elif isinstance(color, str) and color in {
                "white",
                "#ffffff",
                *[p["surface"] for p in _PLOT_THEMES.values()],
            }:
                part.color = palette["surface"]
            border = getattr(part, "line", None)
            if (
                border is not None
                and isinstance(border.color, str)
                and border.color in neutral
            ):
                border.color = palette["text"]
    for shape in fig.layout.shapes:
        if shape.line.color in {
            "black",
            "#000000",
            *[p["axis"] for p in _PLOT_THEMES.values()],
        }:
            shape.line.color = palette["axis"]
    for annotation in fig.layout.annotations:
        annotation.font.color = palette["text"]
        if annotation.showarrow:
            annotation.arrowcolor = palette["text"]
    return fig


def _axial_row_layout(stages, results=None):
    """Lightweight nominal row placements shared by channel and cascade views.

    Units remain metres here. Unlike physical assembly/export placement this
    uses meanline chords and the same interspace lengths as the calculation.
    Unannotated legacy results retain their original nominal gap convention.
    """
    results = results if results is not None else {"inputs": {"turbine_type": "axial"},
                                                  "stages_performance": stages}
    rows, cursor = [], 0.0
    for stage_number, stage in enumerate(stages, 1):
        for row_name, station_index in (("stator", 0), ("rotor", 2)):
            if rows:
                cursor += get_interspace_geometry(results, stage_number - 1, row_name)["meridional_length"]
            geometry = get_blade_row_geometry(stage, row_name)
            end = cursor + geometry["chord_meridional"]
            rows.append(
                dict(
                    stage=stage,
                    stage_number=stage_number,
                    row=row_name,
                    station_index=station_index,
                    geometry=geometry,
                    start=cursor,
                    end=end,
                )
            )
            cursor = end
    return rows


def _axial_view_range(rows):
    start = -CHANNEL_EXTENSION_CHORDS * rows[0]["geometry"]["chord_meridional"]
    end = rows[-1]["end"] + CHANNEL_EXTENSION_CHORDS * rows[-1]["geometry"]["chord_meridional"]
    padding = 0.05 * (end - start)
    return [1000 * (start - padding), 1000 * (end + padding)]


def _extend_channel(hub, shroud, rows, turbine_type):
    """Add display-only inlet/outlet passages at constant endpoint dimensions."""
    first = get_blade_row_geometry(rows[0]["stage"], rows[0]["row"])
    last = get_blade_row_geometry(rows[-1]["stage"], rows[-1]["row"])
    inlet_length = CHANNEL_EXTENSION_CHORDS * first["chord_meridional"]
    outlet_length = CHANNEL_EXTENSION_CHORDS * last["chord_meridional"]
    if turbine_type == "axial":
        inlet_delta, outlet_delta = [-inlet_length, 0], [outlet_length, 0]
    else:
        radial_steps = np.diff(hub[:, 1])
        nonzero = radial_steps[np.abs(radial_steps) > np.finfo(float).eps]
        inlet_direction = np.sign(nonzero[0]) if len(nonzero) else 1.0
        outlet_direction = np.sign(nonzero[-1]) if len(nonzero) else inlet_direction
        inlet_delta = [0, -inlet_direction * inlet_length]
        outlet_delta = [0, outlet_direction * outlet_length]
        # Keep every displayed radial station strictly above the rotation axis.
        inlet_delta[1] = max(inlet_delta[1], -0.9 * hub[0, 1])
        outlet_delta[1] = max(outlet_delta[1], -0.9 * hub[-1, 1])
    return tuple(np.vstack((points[0] + inlet_delta, points, points[-1] + outlet_delta))
                 for points in (hub, shroud))


def _channel_guides(hub, shroud, turbine_type):
    """Smooth illustrative paths constrained within station-derived endwalls.

    Parameterize in flow order, so inward and outward radial designs work alike.
    PCHIP gives gentle transitions; clipping to the local channel preserves the
    physical walls even across very sharp expansions or contractions. These
    curves illustrate passage shape, not a solved velocity field.
    """
    center = (hub + shroud) / 2
    distance = np.linalg.norm(np.diff(center, axis=0), axis=1)
    # Coincident stations can occur at a row interface. Retain a positive
    # parameter increment without moving either physical station.
    parameter = np.r_[0, np.cumsum(np.maximum(distance, 1e-12))]
    samples = np.unique(np.r_[parameter, np.linspace(0, parameter[-1], 241)])
    inner = np.column_stack([np.interp(samples, parameter, hub[:, axis]) for axis in (0, 1)])
    outer = np.column_stack([np.interp(samples, parameter, shroud[:, axis]) for axis in (0, 1)])
    span_axis = 1 if turbine_type == "axial" else 0
    for fraction in np.linspace(.16, .84, 5):
        guide = inner * (1 - fraction) + outer * fraction
        station_values = hub[:, span_axis] * (1 - fraction) + shroud[:, span_axis] * fraction
        smooth = PchipInterpolator(parameter, station_values)(samples)
        height = outer[:, span_axis] - inner[:, span_axis]
        guide[:, span_axis] = np.clip(smooth, inner[:, span_axis] + .035 * height,
                                      outer[:, span_axis] - .035 * height)
        yield guide


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
def plot_turbine_meridional_channel(results, *, theme="light"):
    """Plot the station-derived meanline channel, not a generated mesh.

    Piecewise-linear hub/shroud boundaries connect every row and stage gap.
    Constant-section inlet/outlet extensions frame the blade rows. Smooth guide
    streamlines illustrate passage shape; they are not computed CFD results.
    Coordinates are in millimetres with physical aspect ratio preserved. The
    initial/reset radial range includes zero; users remain free to zoom in.
    """
    stages = results["stages_performance"]
    if not stages:
        raise ValueError("A solved turbine with at least one stage is required.")
    turbine_type = results["inputs"]["turbine_type"]
    if turbine_type not in ("axial", "radial"):
        raise ValueError(f"Invalid stage type: {turbine_type!r}")
    palette = _PLOT_THEMES[theme]
    fig = go.Figure()
    if turbine_type == "axial":
        rows = _axial_row_layout(stages, results)
    else:
        rows = [
            dict(stage=stage, stage_number=number, row=row, station_index=index)
            for number, stage in enumerate(stages, 1)
            for row, index in (("stator", 0), ("rotor", 2))
        ]

    hub, shroud, row_points = [], [], []
    for row in rows:
        stations = row["stage"]["flow_stations"][
            row["station_index"] : row["station_index"] + 2
        ]
        if turbine_type == "axial":
            inner = [
                (x, station["r"] - station["H"] / 2)
                for x, station in zip((row["start"], row["end"]), stations)
            ]
            outer = [
                (x, station["r"] + station["H"] / 2)
                for x, station in zip((row["start"], row["end"]), stations)
            ]
        else:
            # Radial span is axial width; r +/- H/2 would invent wrong radii.
            inner = [(-station["H"] / 2, station["r"]) for station in stations]
            outer = [(+station["H"] / 2, station["r"]) for station in stations]
        hub.extend(inner)
        shroud.extend(outer)
        row_points.append((row, np.asarray(inner), np.asarray(outer)))
    hub, shroud = _extend_channel(np.asarray(hub), np.asarray(shroud), rows, turbine_type)

    def polygon(points, *, color, name, role=None, linecolor=None, hover=None):
        points = np.vstack((points, points[0])) * 1000.0
        fig.add_trace(
            go.Scatter(
                x=points[:, 0],
                y=points[:, 1],
                mode="lines",
                fill="toself",
                fillcolor=color,
                line=dict(color=linecolor or color, width=1 if linecolor else 0),
                name=name,
                legendgroup=name,
                showlegend=False,
                meta=dict(theme_role=role),
                hoverinfo="skip" if hover is None else None,
                hovertemplate=hover,
            )
        )

    polygon(
        np.vstack((hub, shroud[::-1])),
        color=palette["channel"],
        name="Meanline channel",
        role="channel",
    )
    r_max = max(hub[:, 1].max(), shroud[:, 1].max())
    if turbine_type == "axial":
        # Restrained solid-surface shading ends at the physical channel walls.
        polygon(
            np.vstack((hub, [[hub[-1, 0], 0], [hub[0, 0], 0]])),
            color=palette["solid"],
            name="Hub surface",
            role="solid",
        )
        wall_thickness = 0.035 * r_max
        polygon(
            np.vstack((shroud, shroud[::-1] + [0, wall_thickness])),
            color=palette["solid"],
            name="Shroud surface",
            role="solid",
        )
        xrange = _axial_view_range(rows)
        radial_top = 1.08 * r_max
        xtitle = "Axial direction [mm]"
    else:
        wall_thickness = 0.10 * np.max(shroud[:, 0] - hub[:, 0])
        polygon(
            np.vstack((hub, hub[::-1] - [wall_thickness, 0])),
            color=palette["solid"],
            name="Hub surface",
            role="solid",
        )
        polygon(
            np.vstack((shroud, shroud[::-1] + [wall_thickness, 0])),
            color=palette["solid"],
            name="Shroud surface",
            role="solid",
        )
        width = shroud[:, 0].max() - hub[:, 0].min()
        xrange = [
            1000 * (hub[:, 0].min() - 2 * width),
            1000 * (shroud[:, 0].max() + 2 * width),
        ]
        radial_top = r_max + max(0.05 * r_max, 0.25 * np.ptp(hub[:, 1]))
        xtitle = "Spanwise direction [mm]"

    for row, inner, outer in row_points:
        is_stator = row["row"] == "stator"
        color = COLOR_STATOR if is_stator else COLOR_ROTOR
        label = row["row"].title()
        stage_name = row["stage"].get("name") or f"Stage {row['stage_number']}"
        polygon(
            np.vstack((inner, outer[::-1])),
            color=_ROW_FILLS[label],
            linecolor=color,
            name=label,
            hover=f"{stage_name} · {label}<br>Meanline channel<extra></extra>",
        )

    for guide in _channel_guides(hub, shroud, turbine_type):
        fig.add_trace(go.Scatter(
            x=1000 * guide[:, 0], y=1000 * guide[:, 1], mode="lines",
            name="Flow guides", legendgroup="Flow guides", showlegend=False,
            line=dict(color=palette["guide"], width=1.15),
            meta=dict(theme_role="flow_guide"),
            hovertemplate="Geometry-based guide streamline<br>Illustrative passage path; not a CFD solution<extra></extra>",
        ))

    for points, label in ((hub, "Hub"), (shroud, "Shroud")):
        fig.add_trace(
            go.Scatter(
                x=1000 * points[:, 0],
                y=1000 * points[:, 1],
                mode="lines",
                name=label,
                line=dict(color=palette["wall"], width=2, dash="solid"),
                showlegend=False,
                meta=dict(theme_role="endwall"),
                hovertemplate=f"{label}<br>%{{x:.3f}} mm, %{{y:.3f}} mm<extra></extra>",
            )
        )
    _style_axes(fig, xtitle=xtitle, ytitle="Radial direction [mm]", equal=True)
    fig.update_xaxes(range=xrange, constrain="domain", tickangle=0)
    fig.update_yaxes(
        range=[0.0, 1000.0 * radial_top],
        rangemode="tozero",
        fixedrange=False,
        constrain="domain",
    )
    fig.update_layout(
        margin=dict(l=45, r=20, t=20, b=55),
        showlegend=False,
    )
    return apply_plot_theme(fig, theme)


# =============================================================================
# Plot turbine blades (multistage)
# =============================================================================
def plot_turbine_blades(results, N_points=601, N_blades_plot=8, *, theme="light"):
    """Render effective shared blade states with actual sampled kernel curves."""
    stages = results["stages_performance"]
    if not stages:
        raise ValueError("A solved turbine with at least one stage is required.")
    turbine_type = results["inputs"]["turbine_type"]
    fig = go.Figure()
    cursor, greatest_radius, first_pitch = 0., 0., None
    starts, chords = [], []
    for stage_index, stage in enumerate(stages):
        for row in ("stator", "rotor"):
            state = state_for_stage_row(stage, row, turbine_type)
            section = section_from_state(state)
            params = section.parameters
            if params["geometry"]["type"] != turbine_type:
                raise ValueError("The edited row stream surface differs from the turbine type.")
            points = section_coordinates(section)
            label, color = ("Stator", COLOR_STATOR) if row == "stator" else ("Rotor", COLOR_ROTOR)
            count = params["cascade"]["blade_count"]
            if turbine_type == "axial":
                if starts:
                    cursor += get_interspace_geometry(results, stage_index, row)["meridional_length"]
                pitch = 2*np.pi*params["geometry"]["mean_radius"]/count
                first_pitch = pitch if first_pitch is None else first_pitch
                copies = [points+[cursor, (index-1)*pitch] for index in range(N_blades_plot+3)]
                starts.append(cursor)
                chords.append(section.meridional_chord)
                cursor += section.meridional_chord
            elif turbine_type == "radial":
                def rotations():
                    for angle in np.arange(count)*2*np.pi/count:
                        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
                        yield points@rotation.T
                copies = rotations()
                greatest_radius = max(greatest_radius, float(np.linalg.norm(points, axis=1).max()))
            else:
                raise ValueError(f"Invalid stage type: {turbine_type!r}")
            for copy in copies:
                fig.add_trace(go.Scatter(x=1000*copy[:, 0], y=1000*copy[:, 1], mode="lines",
                    line=dict(color=color, width=LINE_WIDTH, simplify=False), fill="toself",
                    fillcolor=_ROW_FILLS[label], name=label, legendgroup=label, showlegend=False))
    if turbine_type == "axial":
        _style_axes(fig, xtitle="Axial direction [mm]", ytitle="Tangential direction [mm]", equal=True)
        first = -CHANNEL_EXTENSION_CHORDS*chords[0]
        last = starts[-1]+(1+CHANNEL_EXTENSION_CHORDS)*chords[-1]
        padding = .05*(last-first)
        fig.update_xaxes(range=[1000*(first-padding), 1000*(last+padding)], constrain="domain")
        fig.update_yaxes(range=[0, 1000*N_blades_plot*first_pitch])
    else:
        _style_axes(fig, xtitle="x direction [mm]", ytitle="y direction [mm]", equal=True)
        fig.update_xaxes(range=[0., 1050*greatest_radius], constrain="domain")
        fig.update_yaxes(range=[0., 1050*greatest_radius], constrain="domain")
    fig.update_layout(template="simple_white", margin=dict(l=30, r=30, t=30, b=30), showlegend=False)
    return apply_plot_theme(fig, theme)


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

    def seg(x0, y0, x1, y1, legend, vector):
        # Screen-sized heads remain readable when toggling Mach/m/s or zooming.
        # Plotly derives their direction from the preceding point, including
        # the reversed meridional axis. A zero vector has no direction to draw.
        head_size = 12 if np.hypot(x1 - x0, y1 - y0) > np.finfo(float).eps else 0
        fig.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines+markers",
                line=dict(color=color, width=2),
                marker=dict(
                    symbol="arrow",
                    angleref="previous",
                    size=[0, head_size],
                    color=color,
                ),
                name=name,
                legendgroup=name,
                showlegend=legend,
                meta=dict(vector=vector),
                hovertemplate=(
                    f"{name} · {vector}<br>Tangential: %{{x:.3f}}"
                    "<br>Meridional: %{y:.3f}<extra></extra>"
                ),
            )
        )

    seg(0.0, 0.0, v_t, v_m, show_legend, "Absolute velocity")  # v
    seg(0.0, 0.0, w_t, w_m, False, "Relative velocity")  # w
    seg(w_t, w_m, v_t, v_m, False, "Blade velocity")  # u closes w -> v


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
        scaleanchor="x",
        scaleratio=1.0,
        constrain="domain",
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
    "loss_incidence",
]

_LOSS_LABELS = {
    "loss_profile": "Profile",
    "loss_trailing": "Trailing edge",
    "loss_secondary": "Secondary",
    "loss_clearance": "Tip clearance",
    "loss_incidence": "Incidence",
}

_LOSS_COLORS = {
    # Plasma samples at 0.30, 0.425, 0.55, 0.675 and 0.80. These are fixed
    # categorical colors; they do not encode the magnitude of each loss.
    "loss_profile": "#8f0da4",
    "loss_trailing": "#b7318a",
    "loss_secondary": "#d6556d",
    "loss_clearance": "#ee7b51",
    "loss_incidence": "#fca636",
    "loss_exit_kinetic": "#64748B",
    "loss_unallocated": "#8995A5",
}
_LOSS_COLORS_DARK = dict(_LOSS_COLORS, loss_exit_kinetic="#A9BACD", loss_unallocated="#BCC6D4")


def plot_turbine_loss_distribution(results):
    r"""Plot contributions to the efficiency deficit after exhaust recovery.

    Every component uses the same recovery-adjusted reference drop, so the
    blade-row columns are directly comparable across stages:

    ``delta_eta = loss_component * 0.5 * row_exit_speed**2 / (dh_is_total - f * KE_exit)``.

    Stators use absolute exit speed and rotors use relative exit speed, exactly
    as in ``compute_stage_performance``. Only the final rotor's absolute exit
    kinetic energy remaining after prescribed recovery is included; intermediate exit kinetic energy is
    carried into the next stage and must not be counted again. The sum of all
    columns therefore equals ``100 * (1 - overall efficiency_ts_with_recovery)`` in efficiency
    percentage points for the solved model. Coefficients and energies in the
    result dictionary retain their original units.

    The five named components include incidence and already contain any Benner
    penetration weighting. Diagnostic/unweighted values are not added again.
    If a custom or tuned row total differs from the named component sum, show
    that signed difference explicitly as ``Other / unallocated`` rather than
    silently rescaling the available components.

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    stages = results["stages_performance"]
    if not stages:
        raise ValueError("A solved turbine with at least one stage is required.")
    reference_drop = float(results["overall_performance"]["isentropic_enthalpy_drop"])
    if not np.isfinite(reference_drop) or reference_drop <= 0:
        raise ValueError("The overall isentropic enthalpy drop must be positive and finite.")
    recovery = diffuser_settings(results.get("inputs", {}))["kinetic_energy_recovery_fraction"]
    leaving_energy = 0.5 * float(stages[-1]["flow_stations"][3]["v"]) ** 2
    reference_drop -= recovery * leaving_energy
    if not np.isfinite(reference_drop) or reference_drop <= 0:
        raise ValueError("The enthalpy drop with exhaust recovery must be positive and finite.")
    exit_label = "Exit"

    labels, row_losses, row_energy = [], [], []
    for i, st in enumerate(stages, 1):
        for label, row, station, velocity in (
            (f"S{i}", "stator", 1, "v"), (f"R{i}", "rotor", 3, "w")
        ):
            labels.append(label)
            row_losses.append(st["losses"][row])
            row_energy.append(0.5 * float(st["flow_stations"][station][velocity]) ** 2)
    labels.append(exit_label)
    component_energy = {
        component: [
            float(losses.get(component, 0.0)) * energy
            for losses, energy in zip(row_losses, row_energy)
        ] + [0.0]
        for component in _LOSS_COMPONENTS
    }
    remainder = [
        float(losses["loss_total"]) * energy
        - sum(component_energy[component][index] for component in _LOSS_COMPONENTS)
        for index, (losses, energy) in enumerate(zip(row_losses, row_energy))
    ] + [0.0]
    # Ignore floating-point summation noise only; custom totals remain explicit.
    has_remainder = any(abs(value / reference_drop) > 1e-12 for value in remainder)

    fig = go.Figure()

    def add_component(name, energy, role):
        fig.add_trace(
            go.Bar(
                x=labels,
                y=100.0 * np.asarray(energy) / reference_drop,
                customdata=energy,
                name=name,
                meta=dict(theme_role=role),
                marker=dict(
                    color=_LOSS_COLORS[role],
                    line=dict(color=_PLOT_THEMES["light"]["surface"], width=0.75),
                ),
                hovertemplate=(
                    "%{x}<br>Δη: %{y:.2f}% (efficiency percentage points)"
                    "<br>Specific energy: %{customdata:.3f} J/kg"
                    "<extra>%{fullData.name}</extra>"
                ),
            )
        )

    for component in _LOSS_COMPONENTS:
        add_component(
            _LOSS_LABELS[component], component_energy[component], component
        )
    if has_remainder:
        add_component("Other / unallocated", remainder, "loss_unallocated")
    add_component(
        "Exit kinetic energy",
        [0.0] * len(row_losses) + [(1 - recovery) * leaving_energy],
        "loss_exit_kinetic",
    )
    fig.data[-1].hovertemplate = (
        "Exit: unrecovered kinetic energy after diffuser recovery"
        "<br>Δη: %{y:.2f}% (efficiency percentage points)"
        "<br>Specific energy: %{customdata:.3f} J/kg<extra></extra>"
    )
    contributions = np.asarray([trace.y for trace in fig.data])
    positive_totals = np.maximum(contributions, 0).sum(axis=0)
    negative_totals = np.minimum(contributions, 0).sum(axis=0)
    y_top = max(1.2 * positive_totals.max(), 1.0)
    y_bottom = 1.2 * negative_totals.min()

    fig.update_layout(
        barmode="relative",
        template="simple_white",
        margin=dict(l=70, r=20, t=75, b=60),
        bargap=0.5,
        legend=dict(
            orientation="h",
            entrywidth=140,
            entrywidthmode="pixels",
            x=0.0,
            y=1.03,
            xanchor="left",
            yanchor="bottom",
            font=dict(size=11),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="black",
            borderwidth=1,
        ),
    )
    fig.update_xaxes(
        title_text="Blade row / final exit",
        tickangle=0,
        title_font=dict(size=AXIS_FONT_SIZE),
        tickfont=dict(size=TICK_FONT_SIZE),
        showline=True,
        mirror=True,
        linewidth=AXIS_LINE_WIDTH,
        linecolor="black",
    )
    fig.update_yaxes(
        title_text="Efficiency penalty Δη [%]",
        range=[y_bottom, y_top],
        title_font=dict(size=AXIS_FONT_SIZE),
        tickfont=dict(size=TICK_FONT_SIZE),
        showline=True,
        mirror=True,
        linewidth=AXIS_LINE_WIDTH,
        linecolor="black",
        gridcolor="#D9D9D9",
    )
    return fig


# =============================================================================
# Plot overall efficiency trends vs blade velocity ratio
# =============================================================================
# Use the same blue/orange series colors as the blade and velocity views.
COLOR_ETA_TT = COLOR_ROTOR
COLOR_ETA_TS = "#d97706"
COLOR_ETA_RECOVERY = "#238b62"


def plot_turbine_efficiency_trends(trends, results=None):
    """
    Single plot of overall total-to-total, total-to-static and exhaust-recovery efficiency
    trends in percent as a function of the (overall) blade-velocity ratio.
    Input efficiencies remain fractions; only plotted values are multiplied by 100.

    Parameters
    ----------
    trends : dict
        Output of ``compute_turbine_efficiency_trends`` (core module). Must
        provide ``"blade_velocity_ratio"``, ``"eta_tt"``, ``"eta_ts"``
        and ``"eta_ts_with_recovery"``.

    results : dict, optional
        Output of ``compute_turbine_performance``. If given, the current
        design operating point (overall ``nu`` and its efficiencies) is marked
        on the curves.

    Returns
    -------
    fig : plotly.graph_objects.Figure
    """
    nu = np.asarray(trends["blade_velocity_ratio"], dtype=float)
    eta_tt = 100.0 * np.asarray(trends["eta_tt"], dtype=float)
    eta_ts = 100.0 * np.asarray(trends["eta_ts"], dtype=float)
    eta_recovery = 100.0 * np.asarray(trends["eta_ts_with_recovery"], dtype=float)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=nu,
            y=eta_tt,
            mode="lines",
            line=dict(color=COLOR_ETA_TT, width=2),
            name="Total-to-total",
            meta=dict(theme_role="efficiency_tt"),
            hovertemplate="Blade velocity ratio: %{x:.3f}<br>Efficiency: %{y:.2f}%"
                          "<extra>%{fullData.name}</extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=nu,
            y=eta_ts,
            mode="lines",
            line=dict(color=COLOR_ETA_TS, width=2),
            name="Total-to-static",
            meta=dict(theme_role="efficiency_ts"),
            hovertemplate="Blade velocity ratio: %{x:.3f}<br>Efficiency: %{y:.2f}%"
                          "<extra>%{fullData.name}</extra>",
        )
    )

    fig.add_trace(go.Scatter(
        x=nu, y=eta_recovery, mode="lines",
        line=dict(color=COLOR_ETA_RECOVERY, width=2),
        name="With exhaust recovery", meta=dict(theme_role="efficiency_recovery"),
        hovertemplate="Blade velocity ratio: %{x:.3f}<br>Efficiency: %{y:.2f}%"
                      "<extra>%{fullData.name}</extra>",
    ))

    # Mark the current design operating point if the full results are provided.
    if results is not None:

        nu0 = float(results["inputs"]["blade_velocity_ratio"])
        eta_tt0 = 100.0 * float(results["overall_performance"]["efficiency_tt"])
        eta_ts0 = 100.0 * float(results["overall_performance"]["efficiency_ts"])
        eta_recovery0 = 100.0 * float(results["overall_performance"]["efficiency_ts_with_recovery"])

        # Vertical line at design blade velocity ratio
        fig.add_vline(
            x=nu0,
            line=dict(color="black", width=1.5, dash="dash"),
        )

        fig.add_trace(
            go.Scatter(
                x=[nu0, nu0, nu0],
                y=[eta_tt0, eta_ts0, eta_recovery0],
                mode="markers",
                marker=dict(
                    size=9,
                    color="white",
                    line=dict(color="black", width=1.5),
                ),
                name="Design point",
                hovertemplate="Blade velocity ratio: %{x:.3f}<br>Efficiency: %{y:.2f}%"
                              "<extra>Design point</extra>",
            )
        )

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
        title_text="Blade velocity ratio [-]",
        title_font=dict(size=AXIS_FONT_SIZE),
        tickfont=dict(size=TICK_FONT_SIZE),
        range=[0.0, float(np.nanmax(nu))],
        showline=True,
        mirror=True,
        linewidth=AXIS_LINE_WIDTH,
        linecolor="black",
        ticks="inside",
        ticklen=TICK_LENGTH,
    )

    fig.update_yaxes(
        title_text="Efficiency [%]",
        title_font=dict(size=AXIS_FONT_SIZE),
        tickfont=dict(size=TICK_FONT_SIZE),
        range=[0.0, 100.0],
        showline=True,
        mirror=True,
        linewidth=AXIS_LINE_WIDTH,
        linecolor="black",
        ticks="inside",
        ticklen=TICK_LENGTH,
        gridcolor="#D9D9D9",
    )

    return fig
