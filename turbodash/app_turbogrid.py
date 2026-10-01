"""Optional TurboGrid download panel and callbacks for the turbine app."""

import logging
from datetime import datetime
from math import isfinite

import dash_bootstrap_components as dbc
from dash import Input, Output, State, dcc, html, no_update
from dash.exceptions import PreventUpdate

from .export_turbogrid import export_turbogrid_zip
from .export_turbogrid_assembly import export_turbogrid_assembly_zip


def turbogrid_export_panel():
    return dbc.Card(
        [
            dbc.CardHeader(html.H5("TurboGrid export", className="mb-0")),
            dbc.CardBody(
                [
                    html.P(
                        "Download the latest successfully calculated design as a ZIP, "
                        "with separate stator and rotor files for every stage. "
                        "Axial turbines only."
                    ),
                    dbc.Label("Export mode", html_for="turbogrid_export_mode"),
                    dcc.Dropdown(
                        id="turbogrid_export_mode",
                        options=[
                            {"label": "Independent rows", "value": "independent"},
                            {"label": "Whole-turbine assembly", "value": "assembly"},
                        ],
                        value="independent",
                        clearable=False,
                        className="mb-3",
                    ),
                    html.Div(
                        [
                            dbc.Label("Row gaps [mm]", html_for="turbogrid_row_gaps"),
                            dbc.Input(
                                id="turbogrid_row_gaps",
                                type="text",
                                value="",
                                placeholder="Example: 10 or 10, 15, 10",
                            ),
                            html.P(
                                "Enter one positive gap per adjacent row pair, separated by commas: "
                                "stator 1 to rotor 1, rotor 1 to stator 2, and so on. "
                                "Examples are illustrative, not recommended spacing. "
                                "Assembly files use global positions and common endwalls; "
                                "mesh each row separately and configure coupling in CFX. "
                                "Follow the ZIP README for required TurboGrid settings.",
                                className="small text-muted mt-2",
                            ),
                        ],
                        id="turbogrid_assembly_options",
                        style={"display": "none"},
                    ),
                    dbc.Button(
                        "Download TurboGrid ZIP",
                        id="turbogrid_export_button",
                        n_clicks=0,
                        disabled=True,
                        color="primary",
                    ),
                    html.Div(
                        id="turbogrid_export_help",
                        className="small text-muted mt-2",
                        role="status",
                    ),
                    dcc.Loading(
                        children=[
                            dcc.Download(id="download_turbogrid"),
                            dbc.Alert(
                                id="turbogrid_export_error",
                                color="danger",
                                is_open=False,
                                className="mt-3 mb-0",
                            ),
                        ],
                        type="circle",
                    ),
                ]
            ),
        ],
        style={"marginTop": "24px", "marginBottom": "24px"},
    )


def _export_unavailable_reason(results, turbine_type):
    if turbine_type != "axial":
        return "TurboGrid export currently supports axial turbines only."
    if not results or not results.get("stages_performance"):
        return "Calculate an axial turbine design to enable the download."
    if results.get("inputs", {}).get("turbine_type") != "axial":
        return "Wait for an axial turbine design to finish calculating."
    return ""


def _parse_row_gaps(value, results):
    count = 2 * len(results["stages_performance"]) - 1
    message = f"Enter {count} positive row gap(s) in millimetres, separated by commas."
    if not isinstance(value, str):
        raise ValueError(message)
    try:
        gaps = [float(part.strip()) / 1000.0 for part in value.split(",")]
    except ValueError as error:
        raise ValueError(message) from error
    if len(gaps) != count or any(not isfinite(gap) or gap <= 0.0 for gap in gaps):
        raise ValueError(message)
    return gaps


def register_turbogrid_callbacks(app):
    @app.callback(
        Output("turbogrid_export_button", "disabled"),
        Output("turbogrid_export_help", "children"),
        Output("turbogrid_assembly_options", "style"),
        Input("result_store", "data"),
        Input({"scope": "overall", "key": "turbine_type"}, "value"),
        Input("turbogrid_export_mode", "value"),
        Input("turbogrid_row_gaps", "value"),
    )
    def update_export_availability(results, turbine_type, export_mode, gap_text):
        style = {} if export_mode == "assembly" else {"display": "none"}
        reason = _export_unavailable_reason(results, turbine_type)
        if not reason:
            if export_mode == "assembly":
                try:
                    _parse_row_gaps(gap_text, results)
                except ValueError as error:
                    reason = str(error)
            elif export_mode != "independent":
                reason = "Select a valid export mode."
        help_text = (
            "Includes positioned rows for all stages. Read the ZIP setup instructions before meshing."
            if export_mode == "assembly" else "Includes independent blade rows for all stages."
        )
        return bool(reason), reason or help_text, style

    @app.callback(
        Output("download_turbogrid", "data"),
        Output("turbogrid_export_error", "children"),
        Output("turbogrid_export_error", "is_open"),
        Input("turbogrid_export_button", "n_clicks"),
        State("result_store", "data"),
        State({"scope": "overall", "key": "turbine_type"}, "value"),
        State("turbogrid_export_mode", "value"),
        State("turbogrid_row_gaps", "value"),
        prevent_initial_call=True,
    )
    def download_turbogrid(n_clicks, results, turbine_type, export_mode, gap_text):
        if not n_clicks:
            raise PreventUpdate
        reason = _export_unavailable_reason(results, turbine_type)
        if reason:
            return no_update, reason, True
        try:
            if export_mode == "assembly":
                archive = export_turbogrid_assembly_zip(
                    results, row_gaps=_parse_row_gaps(gap_text, results)
                )
            elif export_mode == "independent":
                archive = export_turbogrid_zip(results)
            else:
                raise ValueError("Select a valid export mode.")
        except (ValueError, KeyError, TypeError) as error:
            return no_update, f"Could not export TurboGrid geometry: {error}", True
        except Exception:
            logging.getLogger(__name__).exception("TurboGrid ZIP export failed")
            return (
                no_update,
                "TurboGrid export failed. Check the application log for details.",
                True,
            )
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        prefix = "turbogrid_assembly" if export_mode == "assembly" else "turbogrid"
        return (
            dcc.send_bytes(archive, f"{prefix}_{timestamp}.zip", type="application/zip"),
            "",
            False,
        )
