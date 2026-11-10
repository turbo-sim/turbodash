"""Engineering figures from evaluated geometry; no duplicate shape laws."""

import numpy as np
import plotly.graph_objects as go
from dash import html

HUB_COLOR = "#176b89"
SHROUD_COLOR = "#d16b27"
MID_COLOR = "#8b9aa5"

GRAPH_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "scrollZoom": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    "toImageButtonOptions": {"format": "svg", "filename": "impeller_geometry"},
}


def _figure(key, xlabel, ylabel):
    figure = go.Figure()
    figure.update_layout(
        template="plotly_white",
        margin=dict(l=58, r=18, t=35, b=48),
        font=dict(family="Segoe UI, Arial, sans-serif", size=11, color="#294553"),
        paper_bgcolor="white",
        plot_bgcolor="white",
        hovermode="closest",
        legend=dict(orientation="h", x=0, y=1.02, yanchor="bottom"),
        uirevision=f"impeller-{key}",
    )
    figure.update_xaxes(title=xlabel, gridcolor="#e9eef1", zerolinecolor="#cbd8df")
    figure.update_yaxes(title=ylabel, gridcolor="#e9eef1", zerolinecolor="#cbd8df")
    return figure


def build_figures(geometry, *, disk=None):
    figures = {
        "angles": _figure("angles", "Conformal station u", "Metal angle β [deg]"),
        "thickness": _figure("thickness", "Conformal station u", "Thickness [mm]"),
        "channel": _figure("channel", "Axial coordinate z [mm]", "Radius r [mm]"),
        "stacking": _figure(
            "stacking",
            "Span v · hub = 0, shroud = 1",
            "Circumferential position θ [deg]",
        ),
    }
    for index, name, color in ((0, "Hub", HUB_COLOR), (-1, "Shroud", SHROUD_COLOR)):
        section = geometry.sections[index]
        figures["angles"].add_trace(
            go.Scatter(
                x=geometry.u,
                y=np.degrees(section.beta),
                name=name,
                mode="lines",
                line=dict(color=color, width=2.3),
                hovertemplate=name + "<br>u=%{x:.3f}<br>β=%{y:.3f}°<extra></extra>",
            )
        )
        figures["thickness"].add_trace(
            go.Scatter(
                x=geometry.u,
                y=section.target_thickness_m * 1000,
                name=name,
                mode="lines",
                line=dict(color=color, width=2.3),
                hovertemplate=name + "<br>u=%{x:.3f}<br>t=%{y:.4f} mm<extra></extra>",
            )
        )
        # Dashed shroud remains distinguishable when both thickness laws coincide.
        if index == -1:
            figures["thickness"].data[-1].line.dash = "dash"
        points = (
            geometry.channel.point(np.linspace(0, 1, 201), 0 if index == 0 else 1)
            * 1000
        )
        figures["channel"].add_trace(
            go.Scatter(
                x=points[:, 0],
                y=points[:, 1],
                name=name,
                mode="lines",
                line=dict(color=color, width=2.4),
                hovertemplate=name
                + "<br>z=%{x:.2f} mm<br>r=%{y:.2f} mm<extra></extra>",
            )
        )
    for station, label in ((0, "Inlet"), (1, "Outlet")):
        points = geometry.channel.point(station, np.array([0.0, 1.0])) * 1000
        figures["channel"].add_trace(
            go.Scatter(
                x=points[:, 0],
                y=points[:, 1],
                mode="lines",
                line=dict(color="#899ca8", width=1.3, dash="dot"),
                name=label,
                showlegend=False,
                hoverinfo="skip",
            )
        )
    if disk is None and geometry.config.get("disk", {}).get("nose_length_m", 0) > 0:
        from turbodash.geometry.impeller.disk import build_disk

        disk = build_disk(geometry, stream_samples=41, angular_samples=32)
    nose = getattr(disk, "nose_profile_zr", None)
    if nose is not None and len(nose):
        points = np.asarray(nose) * 1000
        figures["channel"].add_trace(
            go.Scatter(
                x=points[:, 0],
                y=points[:, 1],
                name="Hub nose",
                mode="lines",
                line=dict(color=MID_COLOR, width=1.8, dash="dot"),
                hovertemplate="Hub nose<br>z=%{x:.2f} mm<br>r=%{y:.2f} mm<extra></extra>",
            )
        )
    for edge, index, color in (
        ("Leading edge", 0, HUB_COLOR),
        ("Trailing edge", -1, SHROUD_COLOR),
    ):
        theta = np.degrees([section.theta[index] for section in geometry.sections])
        figures["stacking"].add_trace(
            go.Scatter(
                x=geometry.v,
                y=theta,
                name=edge,
                mode="lines",
                line=dict(color=color, width=2.2),
                hovertemplate=edge + "<br>v=%{x:.3f}<br>θ=%{y:.3f}°<extra></extra>",
            )
        )
    figures["channel"].update_yaxes(scaleanchor="x", scaleratio=1)
    return figures


def metric_cards(geometry):
    rake = geometry.diagnostics["rake"]
    lean = np.degrees(geometry.sections[-1].theta[0] - geometry.sections[0].theta[0])
    requested = rake["target_angle_deg"]
    items = [
        ("Main blades", str(geometry.blade_count), "One repeated blade family"),
        (
            "Exit rake",
            "Free" if requested is None else f"{requested:g}°",
            (
                "Prescribed trailing-edge angle"
                if requested is not None
                else "Result of the section wraps"
            ),
        ),
        ("LE azimuth shift", f"{lean:+.2f}°", "Shroud relative to hub; not local lean"),
        (
            "Thickness residual",
            f"{100*geometry.diagnostics['maximum_relative_thickness_error']:.4f}%",
            "Maximum sampled paired-distance error",
        ),
    ]
    return [
        html.Div(
            [html.Span(label), html.Strong(value), html.Small(detail)],
            className="impeller-metric",
        )
        for label, value, detail in items
    ]
