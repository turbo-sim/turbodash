"""Display controls for the station-derived turbine channel."""

from dash import Input, Output, dcc
from dash.exceptions import PreventUpdate

from turbodash.visualization.plotly_turbine import plot_turbine_meridional_channel


CONTOUR_HELP = (
    "Choose linear or smooth cubic hub and shroud contours for all blade rows. "
    "All contours pass through the row inlet and outlet stations. "
    "Smooth contours match the straight interspaces and meet the axial or "
    "radial inlet and outlet extensions tangentially."
)


def contour_control():
    """A global display setting with no solver input identity."""
    return dcc.RadioItems(
        id="td-contour-degree",
        options=[
            {"label": "Linear", "value": 1},
            {"label": "Smooth", "value": 3},
        ],
        value=1,
        inline=True,
        className="td-segmented",
        style={"display": "flex", "gap": "6px"},
        inputStyle={"marginRight": "4px"},
        persistence="linear-smooth",
        persistence_type="session",
    )


def register_meridional_callbacks(app):
    """Redraw accepted results without invoking a turbine calculation.

    This callback is the sole writer of the meridional figure. A calculation
    publishes its result to ``result_store``; it never returns a figure made
    with an earlier contour selection.
    """
    @app.callback(
        Output("meridional_plot", "figure"),
        Input("result_store", "data"),
        Input("td-contour-degree", "value"),
    )
    def update_meridional(results, degree):
        if not results:
            raise PreventUpdate
        return plot_turbine_meridional_channel(results, contour_degree=degree)

    return update_meridional
