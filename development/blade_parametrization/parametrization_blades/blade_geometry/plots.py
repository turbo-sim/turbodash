"""Plotly views of one immutable V4 section, independent of Dash callbacks.

The two conformal plots use equal scales for m' and theta. Physical thickness
is the paired intrinsic distance measured by the geometry kernel, in mm.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from copy import deepcopy

from .geometry import build_section

PLOT_BASE_POINTS = 601

BLUE = "#3478ac"
TEAL = "#258375"
ORANGE = "#c77b32"
INK = "#263d4a"
MUTED = "#748995"


def _figure(xlabel: str, ylabel: str, *, equal: bool = False) -> go.Figure:
    figure = go.Figure()
    figure.update_layout(
        template="none", paper_bgcolor="white", plot_bgcolor="white",
        font={"family": "Inter, Segoe UI, sans-serif", "size": 11, "color": INK},
        margin={"l": 55, "r": 24, "t": 18, "b": 57}, height=330,
        hovermode="closest", dragmode="pan", uirevision="v4-section",
        legend={"orientation": "h", "yanchor": "top", "y": -0.18,
                "x": 0, "font": {"size": 10}},
        xaxis={"title": xlabel, "gridcolor": "#edf1f4", "zerolinecolor": "#dce5ea"},
        yaxis={"title": ylabel, "gridcolor": "#edf1f4", "zerolinecolor": "#dce5ea"},
    )
    if equal:
        figure.update_yaxes(scaleanchor="x", scaleratio=1)
    return figure


def empty_figure(message: str = "Preparing geometry…") -> go.Figure:
    figure = _figure("", "")
    figure.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper",
                          showarrow=False, font={"color": MUTED})
    figure.update_xaxes(visible=False)
    figure.update_yaxes(visible=False)
    return figure


def camberline_figure(section) -> go.Figure:
    figure = _figure("m′ [rad]", "θ [rad]", equal=True)
    points = np.asarray(section.camber)
    angles = np.rad2deg(section.beta)
    figure.add_trace(go.Scatter(
        x=points[:, 0], y=points[:, 1], mode="lines", name="Camberline",
        line={"color": BLUE, "width": 2.6},
        customdata=np.column_stack((section.xi, angles)),
        hovertemplate="m′ %{x:.4f}<br>θ %{y:.4f}<br>ξ′ %{customdata[0]:.3f}"
                      "<br>β %{customdata[1]:.2f}°<extra>Camberline</extra>",
    ))
    for index, label in ((0, "LE"), (-1, "TE")):
        point = points[index]
        tangent = np.array([np.cos(section.beta[index]), np.sin(section.beta[index])])
        length = 0.11 * section.prime_chord
        ends = point + np.array([-length, length])[:, None] * tangent
        figure.add_trace(go.Scatter(
            x=ends[:, 0], y=ends[:, 1], mode="lines", showlegend=False,
            line={"color": MUTED, "width": 1, "dash": "dot"}, hoverinfo="skip",
        ))
        figure.add_trace(go.Scatter(
            x=[point[0]], y=[point[1]], mode="markers", showlegend=False,
            marker={"color": BLUE, "size": 6},
            hovertemplate=f"{label}: β = {angles[index]:.2f}°<extra></extra>",
        ))
    figure.add_annotation(
        text=f"β<sub>in</sub> {angles[0]:.1f}° → β<sub>out</sub> {angles[-1]:.1f}°",
        x=0.02, y=0.98, xref="paper", yref="paper", xanchor="left", yanchor="top",
        showarrow=False, bgcolor="rgba(255,255,255,.85)", font={"size": 11},
    )
    return figure


def thickness_figure(section) -> go.Figure:
    figure = _figure("Conformal meridional station ξ′ [−]", "Paired thickness [mm]")
    thickness = section.thickness
    baseline = 1000 * section.meridional_chord * np.asarray(thickness["denton_ratio"])
    curves = [
        ("Denton baseline × cₘ", baseline, MUTED, "dot", 1.4),
        ("Physical target after edge relief", 1000 * np.asarray(thickness["target_physical"]),
         ORANGE, "dash", 1.8),
        ("Mapped paired thickness", 1000 * np.asarray(thickness["measured_physical"]),
         BLUE, "solid", 2.5),
    ]
    for name, values, color, dash, width in curves:
        figure.add_trace(go.Scatter(
            x=section.xi, y=values, mode="lines", name=name,
            line={"color": color, "dash": dash, "width": width},
            hovertemplate="ξ′ %{x:.4f}<br>%{y:.4f} mm<extra>" + name + "</extra>",
        ))
    pressure_is_plus = section.pressure_side == "plus"
    for side, label, color in (("plus", "Pressure" if pressure_is_plus else "Suction", ORANGE),
                               ("minus", "Suction" if pressure_is_plus else "Pressure", TEAL)):
        values = 1000 * np.asarray(section.radius) * np.asarray(thickness[f"{side}_chart_offset"])
        figure.add_trace(go.Scatter(
            x=section.xi, y=values, name=f"{label} offset × r (first order)",
            visible="legendonly", line={"color": color, "width": 1.2},
            hovertemplate="ξ′ %{x:.4f}<br>%{y:.4f} mm<extra>First-order side offset</extra>",
        ))
    figure.update_layout(hovermode="x unified", legend={"font": {"size": 9}, "y": -0.2},
                         margin={"b": 82})
    return figure


def blade_to_blade_figure(section) -> go.Figure:
    figure = _figure("m′ [rad]", "θ [rad]", equal=True)
    boundary = np.asarray(section.boundary)
    closed = np.vstack((boundary, boundary[:1]))
    pitch = float(section.diagnostics["pitch_angle"])
    for copy in (-1, 1):
        figure.add_trace(go.Scatter(
            x=closed[:, 0], y=closed[:, 1] + copy * pitch, mode="lines",
            fill="toself", fillcolor="rgba(135,156,168,.09)",
            line={"color": "#b8c7cf", "width": 1.1},
            name="Neighboring blades", showlegend=copy == -1,
            hovertemplate="m′ %{x:.4f}<br>θ %{y:.4f}<extra>Neighbor</extra>",
        ))
    figure.add_trace(go.Scatter(
        x=closed[:, 0], y=closed[:, 1], mode="lines", fill="toself",
        fillcolor="rgba(52,120,172,.10)", line={"width": 0},
        showlegend=False, hoverinfo="skip",
    ))
    for side in ("plus", "minus"):
        points = np.asarray(getattr(section, side))
        pressure = side == section.pressure_side
        label = "Pressure side" if pressure else "Suction side"
        figure.add_trace(go.Scatter(
            x=points[:, 0], y=points[:, 1], mode="lines", name=label,
            line={"color": ORANGE if pressure else BLUE, "width": 2.1},
            customdata=np.asarray(section.xi),
            hovertemplate="m′ %{x:.4f}<br>θ %{y:.4f}<br>ξ′ %{customdata:.3f}<extra>" + label + "</extra>",
        ))
    camber = np.asarray(section.camber)
    figure.add_trace(go.Scatter(
        x=camber[:, 0], y=camber[:, 1], mode="lines", name="Camberline",
        line={"color": MUTED, "width": 1, "dash": "dash"}, hoverinfo="skip",
    ))
    return figure


def passage_traces(passage: dict) -> list[go.Scatter]:
    """Batch every geometric crosscut into one trace, with physical hover data.

    The calculation is independent of display. Keeping only two extra traces
    avoids a Plotly trace per cut while allowing both cuts and center points to
    show the measured opening. Missing cuts are never bridged by a centerline.
    """
    if not passage.get("available") or not passage.get("valid_count"):
        return []
    cuts = passage["cuts_chart"]
    widths = np.asarray(passage["widths_mm"], dtype=float)
    stations = np.asarray(passage["stations"], dtype=float)
    x, y, custom = [], [], []
    for cut, width, station in zip(cuts, widths, stations):
        points = np.asarray(cut)
        x.extend([*points[:, 0], None])
        y.extend([*points[:, 1], None])
        custom.extend([[float(width), float(station)]] * len(points) + [[None, None]])
    hover = ("Physical opening %{customdata[0]:.3f} mm"
             "<br>Center station ξ′ %{customdata[1]:.3f}<extra>Geometric crosscut</extra>")
    centers = np.asarray(passage["center_chart"])
    return [
        go.Scatter(x=x, y=y, mode="lines", name="Passage crosscuts",
                   line={"color": TEAL, "width": 1.2, "simplify": False},
                   customdata=custom, hovertemplate=hover, connectgaps=False),
        go.Scatter(x=centers[:, 0], y=centers[:, 1], mode="markers",
                   name="Passage centers", showlegend=False,
                   marker={"color": TEAL, "size": 4},
                   customdata=np.column_stack((widths, stations)), hovertemplate=hover),
    ]


def meridional_figure(section) -> go.Figure:
    """Channel dimensions with the same physical meridional reference path."""
    figure = _figure("Axial coordinate z [mm]", "Radius r [mm]", equal=True)
    geometry = section.parameters["geometry"]
    kind = geometry["type"]
    if kind == "axial":
        chord = 1000 * section.meridional_chord
        hub = 1000 * (geometry["mean_radius"] - geometry["blade_height"] / 2)
        shroud = 1000 * (geometry["mean_radius"] + geometry["blade_height"] / 2)
        polygon_z = [0, chord, chord, 0, 0]
        polygon_r = [hub, hub, shroud, shroud, hub]
        z = 1000 * np.asarray(section.meridional_distance)
        radii = 1000 * np.asarray(section.radius)
    else:
        height = 1000 * geometry["blade_height"]
        inlet = 1000 * geometry["inlet_radius"]
        outlet = 1000 * geometry["outlet_radius"]
        polygon_z = [0, 0, height, height, 0]
        polygon_r = [inlet, outlet, outlet, inlet, inlet]
        z = np.full_like(section.radius, height / 2, dtype=float)
        radii = 1000 * np.asarray(section.radius)
    figure.add_trace(go.Scatter(
        x=polygon_z, y=polygon_r, mode="lines", fill="toself",
        fillcolor="rgba(52,120,172,.075)", line={"color": "#8198a5", "width": 1.5},
        name="Channel extent", hoverinfo="skip",
    ))
    # Project the entire mapped boundary, including rounded-edge protrusions.
    # For the supported unswept cylinders/radial planes every span has the same
    # physical meridional footprint; these are its hub/shroud extrema.
    station = np.asarray(section.boundary)[:, 0]
    if kind == "axial":
        mapped_station = 1000 * station * float(np.asarray(section.radius)[0])
        lower, upper = float(mapped_station.min()), float(mapped_station.max())
        blade_z = [lower, upper, upper, lower, lower]
        blade_r = [hub, hub, shroud, shroud, hub]
    else:
        direction = np.sign(geometry["outlet_radius"] - geometry["inlet_radius"])
        mapped_station = 1000 * geometry["inlet_radius"] * np.exp(direction * station)
        lower, upper = float(mapped_station.min()), float(mapped_station.max())
        blade_z = [0, 0, height, height, 0]
        blade_r = [lower, upper, upper, lower, lower]
    figure.add_trace(go.Scatter(
        x=blade_z, y=blade_r, mode="lines", fill="toself",
        fillcolor="rgba(52,120,172,.12)", line={"color": BLUE, "width": 1.5, "dash": "dot"},
        name="Full blade projection",
        hovertemplate="z %{x:.3f} mm<br>r %{y:.3f} mm<extra>Full blade projection</extra>",
    ))
    figure.add_trace(go.Scatter(
        x=z, y=radii, mode="lines", name="Reference stream surface",
        line={"color": BLUE, "width": 2.5},
        customdata=np.asarray(section.xi),
        hovertemplate="z %{x:.3f} mm<br>r %{y:.3f} mm<br>ξ′ %{customdata:.3f}<extra></extra>",
    ))
    figure.add_trace(go.Scatter(
        x=[z[0], z[-1]], y=[radii[0], radii[-1]], mode="markers+text",
        marker={"color": [TEAL, ORANGE], "size": 7}, text=["LE", "TE"],
        textposition=["top left", "bottom right"], showlegend=False, hoverinfo="skip",
    ))
    figure.add_annotation(
        text=f"cₘ = {1000 * section.meridional_chord:.2f} mm",
        x=0.02, y=0.98, xref="paper", yref="paper", xanchor="left", yanchor="top",
        showarrow=False, bgcolor="rgba(255,255,255,.85)",
    )
    return figure


def build_figures(section) -> dict[str, go.Figure]:
    """Densely sample the same analytic geometry for display, without refitting.

    Validation/diagnostics retain the requested scientific resolution. Display
    refinement is inexpensive: it doesn't repeat quadratic intersection checks
    or build CAD. It never interpolates a cosmetic spline over a real corner.
    """
    display = section
    if section.parameters["numerics"]["number_of_points"] < PLOT_BASE_POINTS:
        parameters = deepcopy(section.parameters)
        parameters["numerics"]["number_of_points"] = PLOT_BASE_POINTS
        display = build_section(parameters, span_fraction=section.diagnostics["span_fraction"],
                                check_intersections=False)
    figures = {
        "camberline": camberline_figure(display),
        "thickness": thickness_figure(display),
        "blade_to_blade": blade_to_blade_figure(display),
        "meridional": meridional_figure(section),
    }
    for figure in figures.values():
        figure.update_traces(line_simplify=False)
    return figures
