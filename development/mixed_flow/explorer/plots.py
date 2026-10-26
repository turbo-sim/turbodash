"""Plotly figures for interactive exploration of the shared meridional kernel."""

import numpy as np
import plotly.graph_objects as go

HUB, SHROUD, EDGE, SPLITTER, REFERENCE = (
    "#176b89",
    "#d16b27",
    "#334d60",
    "#8752a1",
    "#a2b1bc",
)


def _figure(xlabel, ylabel, key):
    fig = go.Figure()
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=58, r=20, t=14, b=50),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="Segoe UI, Arial, sans-serif", color="#385466", size=11),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.015, x=0, font=dict(size=10)
        ),
        xaxis=dict(title=xlabel, gridcolor="#edf1f4", zeroline=False),
        yaxis=dict(title=ylabel, gridcolor="#edf1f4", zeroline=False),
        hovermode="closest",
        uirevision=key,
    )
    return fig


def _curve(fig, points, name, color, *, dash=None, width=2.5, legend=True, group=None):
    fig.add_trace(
        go.Scatter(
            x=(points[..., 0] * 1000).tolist(),
            y=(points[..., 1] * 1000).tolist(),
            mode="lines",
            name=name,
            legendgroup=group or name,
            showlegend=legend,
            line=dict(color=color, width=width, dash=dash, simplify=False),
            hovertemplate=f"{name}<br>z = %{{x:.3f}} mm<br>r = %{{y:.3f}} mm<extra></extra>",
        )
    )


def channel_figure(channel, blade, splitter, reference=None):
    fig = _figure("Axial coordinate z [mm]", "Radius r [mm]", "meridional-channel")
    s, eta = np.linspace(0, 1, 301), np.linspace(0, 1, 101)
    if reference is not None:
        r_channel, r_blade, r_splitter = reference
        for i, wall in enumerate((r_channel.hub, r_channel.shroud)):
            _curve(
                fig,
                wall.point_at(s),
                "Reference",
                REFERENCE,
                dash="dash",
                width=1.5,
                legend=i == 0,
            )
        for edge in (
            r_blade.leading_edge(eta),
            r_blade.trailing_edge(eta),
            r_splitter.leading_edge(eta),
        ):
            _curve(fig, edge, "Reference", REFERENCE, dash="dot", width=1, legend=False)
    boundary = np.concatenate(
        [
            blade.point(s, 0),
            blade.trailing_edge(eta),
            blade.point(s[::-1], 1),
            blade.leading_edge(eta[::-1]),
        ]
    )
    fig.add_trace(
        go.Scatter(
            x=(boundary[:, 0] * 1000).tolist(),
            y=(boundary[:, 1] * 1000).tolist(),
            mode="lines",
            line=dict(width=0),
            fill="toself",
            fillcolor="rgba(92,130,153,.11)",
            name="Blade extent",
            hoverinfo="skip",
        )
    )
    for position in np.linspace(0, 1, 9):
        points = channel.point(position, eta)
        fig.add_trace(
            go.Scatter(
                x=(points[:, 0] * 1000).tolist(),
                y=(points[:, 1] * 1000).tolist(),
                mode="lines",
                line=dict(color="#cbd7df", width=0.8),
                hoverinfo="skip",
                showlegend=False,
            )
        )
    _curve(fig, channel.hub.point_at(s), "Hub", HUB, width=3)
    _curve(fig, channel.shroud.point_at(s), "Shroud", SHROUD, width=3)
    _curve(fig, blade.leading_edge(eta), "Leading edge", EDGE, width=2)
    _curve(fig, blade.trailing_edge(eta), "Trailing edge", EDGE, width=2, dash="dash")
    _curve(
        fig,
        splitter.leading_edge(eta),
        "Splitter start",
        SPLITTER,
        dash="dash",
        width=2,
    )
    for station, angle in (
        (0, channel.dimensions.inlet_flow_angle_deg),
        (1, channel.dimensions.outlet_flow_angle_deg),
    ):
        centre = channel.point(station, 0.5) * 1000
        direction = np.array([np.cos(np.radians(angle)), np.sin(np.radians(angle))])
        tail, tip = centre - 5 * direction, centre + 5 * direction
        fig.add_annotation(
            x=tip[0],
            y=tip[1],
            ax=tail[0],
            ay=tail[1],
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            text="",
            showarrow=True,
            arrowhead=2,
            arrowwidth=1.5,
            arrowcolor="#698496",
        )
    fig.update_yaxes(scaleanchor="x", scaleratio=1, constrain="domain")
    fig.update_xaxes(constrain="domain")
    return fig


def _series(fig, x, y, name, color, dash=None):
    fig.add_trace(
        go.Scatter(
            x=np.asarray(x).tolist(),
            y=np.asarray(y).tolist(),
            name=name,
            mode="lines",
            line=dict(color=color, width=2, dash=dash, simplify=False),
            hovertemplate=f"{name}<br>s = %{{x:.3f}}<br>%{{y:.4g}}<extra></extra>",
        )
    )


def build_figures(channel, blade, splitter, reference=None):
    data = channel.diagnostics(301)
    area = _figure("Normalized wall arclength s", "Geometric area [mm²]", "area")
    curvature = _figure(
        "Normalized wall arclength s", "Signed curvature [1/mm]", "curvature"
    )
    width = _figure("Normalized wall arclength s", "Connector length [mm]", "width")
    if reference is not None:
        rd = reference[0].diagnostics(301)
        _series(area, rd["s"], rd["area"] * 1e6, "Reference cut", REFERENCE, "dash")
        _series(
            curvature,
            rd["s"],
            rd["hub_curvature"] / 1000,
            "Reference hub",
            REFERENCE,
            "dash",
        )
        _series(
            curvature,
            rd["s"],
            rd["shroud_curvature"] / 1000,
            "Reference shroud",
            "#c5b5a9",
            "dash",
        )
        _series(width, rd["s"], rd["gap"] * 1000, "Reference", REFERENCE, "dash")
    _series(area, data["s"], data["area"] * 1e6, "Revolved cut", HUB)
    _series(
        area,
        data["s"],
        data["projected_area"] * 1e6,
        "Midline-normal projection",
        SHROUD,
        "dash",
    )
    _series(curvature, data["s"], data["hub_curvature"] / 1000, "Hub", HUB)
    _series(curvature, data["s"], data["shroud_curvature"] / 1000, "Shroud", SHROUD)
    _series(width, data["s"], data["gap"] * 1000, "Hub–shroud connector", HUB)
    return dict(
        channel=channel_figure(channel, blade, splitter, reference),
        area=area,
        curvature=curvature,
        width=width,
    )
