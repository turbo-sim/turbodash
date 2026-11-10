"""Turbodash-style workspace: engineering diagrams above the 3D assembly."""

from dash import dcc, html

from .controls import sidebar
from .plots import GRAPH_CONFIG


def _plot(identifier, title, figure, caption):
    return html.Div(
        [
            html.H5(title),
            dcc.Graph(
                id=identifier,
                figure=figure,
                config=GRAPH_CONFIG,
                responsive=True,
                style={"height": "100%"},
            ),
            html.P(caption, className="td-blade-caption impeller-plot-caption"),
        ],
        className="td-engineering-plot impeller-engineering-plot",
    )


def build_layout(config, initial_figures, wall_layout, viewer_layout, *, stores=None):
    """Root callbacks supply stores, evaluated figures and both shared viewers."""
    return html.Div(
        [
            *(stores or []),
            dcc.Download(id="impeller-yaml-download"),
            dcc.Download(id="impeller-stl-download"),
            html.Header(
                [
                    html.Div(
                        [
                            html.Span("◒", className="impeller-brand-mark"),
                            html.Strong("Turbodash"),
                            html.Span("/", className="impeller-brand-divider"),
                            html.Span("Impeller geometry"),
                        ],
                        className="impeller-brand",
                    ),
                    html.Div(
                        [
                            html.Span(
                                "GEOMETRY WORKBENCH", className="impeller-header-tag"
                            ),
                            html.A(
                                "Theory & workflow",
                                href="/_impeller/docs",
                                target="_blank",
                                className="impeller-doc-link",
                            ),
                        ],
                        className="impeller-header-actions",
                    ),
                ],
                className="impeller-header",
            ),
            html.Div(
                [
                    sidebar(config),
                    html.Main(
                        [
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Div(
                                                "CENTRIFUGAL IMPELLER",
                                                className="impeller-eyebrow",
                                            ),
                                            html.H1("Shape the channel and blade"),
                                            html.P(
                                                "Independent hub and shroud sections, explicit rake, and a shared three-dimensional geometry.",
                                                className="impeller-subtitle",
                                            ),
                                        ]
                                    ),
                                    html.Div(
                                        "Ready",
                                        id="impeller-status",
                                        role="status",
                                        className="impeller-status",
                                    ),
                                ],
                                className="impeller-heading",
                            ),
                            html.Div(
                                id="impeller-error",
                                role="alert",
                                className="impeller-error",
                            ),
                            html.Div(
                                id="impeller-metrics", className="impeller-metrics"
                            ),
                            html.Section(
                                [
                                    html.Div(
                                        [
                                            _plot(
                                                "impeller-angle-plot",
                                                "Metal-angle distribution · hub & shroud",
                                                initial_figures["angles"],
                                                "Both sections use the same conformal station u. Changes in turning remain visible together.",
                                            ),
                                            _plot(
                                                "impeller-thickness-plot",
                                                "Thickness distribution · hub & shroud",
                                                initial_figures["thickness"],
                                                "Hub is solid blue; shroud is dashed orange so matching thickness laws remain distinguishable.",
                                            ),
                                        ],
                                        className="td-plot-pair",
                                    ),
                                    html.Div(
                                        [
                                            _plot(
                                                "impeller-channel-plot",
                                                "Meridional channel",
                                                initial_figures["channel"],
                                                "Equal axial and radial scale. Edit the engineering dimensions or drag the wall points below.",
                                            ),
                                            _plot(
                                                "impeller-stacking-plot",
                                                "Edge stacking · hub to shroud",
                                                initial_figures["stacking"],
                                                "Circumferential positions across the span. A horizontal trailing-edge curve has zero rake.",
                                            ),
                                        ],
                                        className="td-plot-pair",
                                    ),
                                ],
                                className="td-plot-group impeller-diagrams",
                            ),
                            html.Section(
                                [
                                    html.Div(
                                        [
                                            html.H2("Hub & shroud control points"),
                                            html.P(
                                                "Select a wall, then drag its interior points. The inlet and outlet stations and tangent points stay fixed. Editor axes: x = axial z [mm], y = radius r [mm].",
                                                className="td-blade-caption",
                                            ),
                                        ],
                                        className="impeller-section-heading",
                                    ),
                                    html.Div(wall_layout, id="impeller-wall-editor"),
                                    html.Div(
                                        [
                                            html.Button(
                                                "Apply wall edits",
                                                id="wall-apply",
                                                type="button",
                                                className="impeller-primary",
                                            ),
                                            html.Button(
                                                "Clear wall refinements",
                                                id="wall-clear",
                                                type="button",
                                            ),
                                            html.Div(
                                                id="wall-notice",
                                                role="status",
                                                className="td-blade-caption",
                                            ),
                                        ],
                                        className="impeller-wall-toolbar",
                                    ),
                                ],
                                className="td-plot-group impeller-wall-panel",
                            ),
                            html.Section(
                                [
                                    html.Div(
                                        [
                                            html.H2("Resulting impeller geometry"),
                                            html.P(
                                                "The accepted channel and blade controls build this view. Use the orientation cube to inspect the inlet, outlet and leaned leading edge.",
                                                className="td-blade-caption",
                                            ),
                                        ],
                                        className="impeller-section-heading",
                                    ),
                                    viewer_layout,
                                    html.Div(
                                        [
                                            html.Button(
                                                "Download full STL",
                                                id="impeller-stl-download-button",
                                                disabled=True,
                                                type="button",
                                                className="impeller-primary",
                                            ),
                                            html.Div(
                                                "Full disk and blade row, fused into one solid. STL coordinates are millimetres.",
                                                id="impeller-stl-status",
                                                role="status",
                                                className="td-blade-caption",
                                            ),
                                        ],
                                        className="impeller-wall-toolbar impeller-export-toolbar",
                                    ),
                                ],
                                className="td-plot-group impeller-assembly-panel",
                            ),
                            html.Footer(
                                "Turbodash · geometry exploration · hub nose and disk dimensions are geometric design inputs",
                                className="impeller-footer",
                            ),
                        ],
                        className="td-workspace-main impeller-main",
                    ),
                ],
                className="td-workspace impeller-workspace",
            ),
        ],
        className="impeller-app",
    )
