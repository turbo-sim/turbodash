"""Two-dimensional row geometry and saved mesh projections.

All 3D geometry and mesh viewers use TurboMoC's VTK scene implementation.
TurboMoC also owns reference-mesh edges and meridional mapping. This module
only adapts their plots to TurboDash's selected-row axes and presentation.
"""

import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

BOUNDARY_COLORS = {
    "blade": "#1f77b4",
    "inlet": "#17a2b8",
    "outlet": "#9467bd",
    "periodic_left": "#ffb347",
    "periodic_right": "#ffb347",
}


def _empty_figure(message):
    figure = go.Figure()
    figure.update_layout(template="simple_white", height=500)
    _annotation(figure, message)
    return figure


def _annotation(figure, message):
    figure.add_annotation(
        text=message, x=0.5, y=1.02, xref="paper", yref="paper",
        showarrow=False, xanchor="center", yanchor="bottom",
        font={"size": 11, "color": "#5b6570"},
    )


def _axis_fonts(figure):
    """Keep labels legible without crowding the paired, compact app plots."""
    figure.update_xaxes(title_font_size=12, tickfont_size=11, title_standoff=8, automargin=True)
    figure.update_yaxes(title_font_size=12, tickfont_size=11, title_standoff=8, automargin=True)


def plot_profile_preview(payload):
    """Show the blade and periodic passage in the physical reference chart.

    The adapter supplies ``q=(tangential, -axial)`` in millimetres. Rotating this
    chart into the familiar axial/tangential axes does not change its lengths.
    """
    section = (payload or {}).get("section", {})
    if not section or not len(section.get("blade", [])):
        return _empty_figure("Generate a blade-row preview to view its 2D profile.")
    if section.get("coordinate_units") != "mm":
        raise ValueError("The 2D preview requires reference-chart coordinates in millimetres.")
    figure = go.Figure()
    for name in ("blade", "periodic_left", "periodic_right", "inlet", "outlet"):
        raw = section.get(name, [])
        if not len(raw):
            continue
        points = np.asarray(raw, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or not np.all(np.isfinite(points)):
            raise ValueError(f"{name} must contain finite 2D reference-chart coordinates.")
        if name == "blade" and not np.array_equal(points[0], points[-1]):
            points = np.vstack((points, points[0]))
        figure.add_trace(go.Scatter(
            x=-points[:, 1], y=points[:, 0], mode="lines",
            name=name.replace("_", " ").title(),
            line={"color": BOUNDARY_COLORS[name], "width": 2 if name == "blade" else 1.5,
                  "dash": "dash" if "periodic" in name else "solid"},
            fill="toself" if name == "blade" else None,
            fillcolor="rgba(31,119,180,0.12)",
            hovertemplate="Axial %{x:.4g} mm<br>Tangential %{y:.4g} mm<extra>%{fullData.name}</extra>",
        ))
    figure.update_layout(
        template="simple_white", height=520,
        margin={"l": 55, "r": 15, "t": 35, "b": 90},
        xaxis={"title": "Axial reference [mm]", "constrain": "domain"},
        yaxis={"title": "Tangential reference [mm]", "scaleanchor": "x",
               "scaleratio": 1},
        legend={"orientation": "h", "x": .5, "xanchor": "center",
                "y": -.22, "yanchor": "top", "font": {"size": 11}},
        uirevision="turbodash-profile-view",
    )
    _axis_fonts(figure)
    return figure


def _meridional_style(
    figure, metadata, *, mesh=False, start_at_zero=False, show_note=True
):
    """Focus an individual row while retaining physical radius coordinates."""
    hub = float(metadata["hub_radius_mm"])
    shroud = float(metadata["shroud_radius_mm"])
    padding = 0.08 * (shroud - hub)
    figure.update_layout(
        height=520,
        margin={"l": 65, "r": 20, "t": 45, "b": 60},
        xaxis_title="Axial coordinate Z [mm]",
        yaxis_title="Radius [mm]",
        uirevision=f"turbodash-meridional-{'mesh' if mesh else 'geometry'}-{metadata.get('source_fingerprint', '')}",
    )
    figure.update_yaxes(
        range=[0.0 if start_at_zero else max(0.0, hub - padding), shroud + padding],
        scaleanchor="x",
        scaleratio=1,
        autorange=False,
    )
    if start_at_zero:
        figure.update_yaxes(rangemode="tozero", fixedrange=False)
    _axis_fonts(figure)
    if show_note:
        _annotation(
            figure,
            (
                "Projection of actual swept mesh stations."
                if mesh
                else "Cylindrical row; shaded blade envelope."
            ),
        )
    return figure


def plot_geometry_meridional(preview, *, start_at_zero=False, show_note=True):
    """Show the selected row's physical meridional geometry, without meshing.

    ``preview_blade_row`` prepares this figure with TurboMoC's
    ``meridional_figure`` from the same Design used for the 3D surfaces. Plot
    inputs remain immutable, including when the preview is cached by the app.
    """
    if not preview or not preview.get("meridional_figure"):
        return _empty_figure(
            "Generate a blade-row preview to view its meridional geometry."
        )
    return _meridional_style(
        go.Figure(preview["meridional_figure"]),
        preview["metadata"],
        start_at_zero=start_at_zero,
        show_note=show_note,
    )


def mesh_figures(directory, *, edge_width=0.35, start_at_zero=False, show_notes=True):
    """Return actual reference-mesh and meridional-grid figures for one job.

    Loading the saved section does not initialize Gmsh or regenerate anything.
    The blade-to-blade chart is displayed in reference millimetres, with axial
    and tangential axes matching ``plot_profile_preview``. It is not a physical
    Cartesian slice of the annular volume. The meridional view projects the
    actual periodic-boundary mesh stations through the saved span distribution.
    """
    if not np.isfinite(edge_width) or edge_width <= 0:
        raise ValueError("Mesh edge width must be finite and positive.")
    # Import lazily so meanline/2D profile viewing needs no optional mesh stack.
    from turbo_moc.app.plots import section_mesh_figure, meridional_figure
    from turbo_moc.meshing.mesh_3d import load_section

    directory = Path(directory)
    section, design = load_section(directory)
    config = json.loads(
        (directory / "mesh_configuration.json").read_text(encoding="utf-8")
    )
    metadata = json.loads(
        (directory / "geometry_metadata.json").read_text(encoding="utf-8")
    )
    reference = section_mesh_figure(section, design, edge_width=edge_width)
    scale = (
        1.0
        if design.meridional_channel.is_linear
        else design.meridional_channel.reference_radius
    )
    for trace in reference.data:
        pitchwise, downstream_negative = np.asarray(trace.x, float), np.asarray(
            trace.y, float
        )
        trace.x = -downstream_negative * scale
        trace.y = pitchwise * scale
    reference.update_layout(
        height=520,
        margin={"l": 60, "r": 20, "t": 45, "b": 60},
        xaxis_title="Axial reference [mm]",
        yaxis_title="Tangential reference [mm]",
        uirevision=f"turbodash-reference-mesh-{metadata.get('source_fingerprint', '')}",
        meta={
            "coordinate_units": "mm",
            "coordinate_basis": "annular_conformal_reference",
            "section_nodes": len(section["xy"]),
            "section_cells": sum(len(cells) for cells in section["cells"].values()),
        },
    )
    _axis_fonts(reference)
    if show_notes:
        _annotation(reference, "Reference mesh in millimetres.")
    meridional, grid = meridional_figure(
        design, config["mesh"]["numerics"], section, edge_width=edge_width
    )
    meridional.update_layout(
        meta={
            name: grid[name]
            for name in (
                "basis",
                "coordinate_basis",
                "span_layers",
                "streamwise_stations",
            )
        }
    )
    return reference, _meridional_style(
        meridional,
        metadata,
        mesh=True,
        start_at_zero=start_at_zero,
        show_note=show_notes,
    )


def plot_reference_mesh(directory, *, edge_width=0.35):
    """Show the saved blade-to-blade mesh; use ``mesh_figures`` for both views."""
    return mesh_figures(directory, edge_width=edge_width)[0]


def plot_mesh_meridional(directory, *, edge_width=0.35):
    """Show the saved meridional projection; use ``mesh_figures`` for both views."""
    return mesh_figures(directory, edge_width=edge_width)[1]
