"""Connect accepted results, the blade modeller, and downstream consumers.

Only the calculation callback writes ``result_store``. Geometry edits never
enter its input graph, and the immutable result remains the meanline solution.
"""

from dash import Input, Output, State
from dash.exceptions import PreventUpdate

from turbodash.geometry.meanline import effective_results
from turbodash.visualization.plotly_turbine import plot_turbine_blades


def pending_geometry_reason(results, event):
    """Block downstream actions until this baseline's journal is projected.

    An event from a replaced meanline baseline must not disable the new
    geometry. The projection marker belongs only to downstream result copies;
    the solver result and its inputs never receive editor state.
    """
    version = (results or {}).get("_geometry_projection", {})
    epoch = version.get("epoch")
    if (event and epoch and event.get("epoch") == epoch
            and event.get("sequence", 0) > version.get("revision", 0)):
        return "Wait for geometry edits to finish before exporting or generating a mesh."
    return ""


def register_geometry_workflow(app, *, loaded_config_store=None, normalize_inputs=None):
    from .components.blade_modeller import register_callbacks

    register_callbacks(app, loaded_config_store=loaded_config_store,
                       normalize_inputs=normalize_inputs)

    @app.callback(
        Output("geometry_results_store", "data"),
        Input("geometry_committed", "data"),
        State("result_store", "data"),
        prevent_initial_call=True,
    )
    def downstream_geometry(state, results):
        if not state or not results:
            raise PreventUpdate
        try:
            projected = effective_results(results, state)
            projected["_geometry_projection"] = {
                "epoch": state.get("_epoch"), "revision": state.get("revision", 0),
            }
            return projected
        except ValueError:
            # A new accepted calculation is replacing this older baseline.
            raise PreventUpdate

    @app.callback(
        Output("blades_plot", "figure", allow_duplicate=True),
        Input("geometry_state", "data"),
        Input("meanline_pane", "style"),
        State("result_store", "data"),
        prevent_initial_call=True,
    )
    def meanline_blades(state, pane_style, results):
        if not state or not results:
            raise PreventUpdate
        if isinstance(pane_style, dict) and pane_style.get("display") == "none":
            # A multistage blade figure can exceed several MB. Opening the tab
            # renders the latest state without serializing that figure per drag.
            raise PreventUpdate
        try:
            current = effective_results(results, state)
        except ValueError:
            raise PreventUpdate
        try:
            return plot_turbine_blades(current, N_points=601, N_blades_plot=8)
        except ValueError as error:
            return {"data": [], "layout": {"annotations": [{
                "text": f"Blade geometry unavailable: {error}", "showarrow": False,
                "xref": "paper", "yref": "paper", "x": .5, "y": .5,
            }]}}
