"""Local Dash workbench for the meridional geometry engineering controls.

Importing this module starts no server. The geometry and its validation come from
the same model used by scripts; invalid drafts retain explicitly marked figures.
"""

from __future__ import annotations

import argparse
import base64
import binascii
from copy import deepcopy
import json
import math
from pathlib import Path

from dash import ALL, Dash, Input, Output, State, ctx, dcc, html, no_update
from dash.exceptions import PreventUpdate

from . import model

HERE = Path(__file__).resolve().parent
FIELD_PATHS = tuple(
    path
    for group in model.GROUPS
    for path, spec in model.FIELD_SPECS.items()
    if spec["group"] == group
)
ANGLE_PATHS = tuple(path for path in FIELD_PATHS if path in model.ANGLE_PATHS)
GRAPH_NAMES = ("channel", "area", "curvature", "width")
PRESETS = [
    {"label": "Centrifugal baseline", "value": "baseline"},
    {"label": "Inclined inlet + mixed-flow outlet", "value": "inclined"},
    {"label": "Romei-inspired wall angles", "value": "romei"},
]
GRAPH_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "scrollZoom": False,
    "toImageButtonOptions": {"format": "svg", "filename": "meridional_geometry"},
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}


def _finite_number(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def _control_view(config):
    """Display invalid numeric drafts without trying to clip them into validity."""
    numbers, sliders, disabled, lows, highs = [], [], [], [], []
    for path in FIELD_PATHS:
        spec = model.FIELD_SPECS[path]
        try:
            value = model.display_value(config, path)
        except (TypeError, ValueError, KeyError):
            value = None
        slider = (
            value
            if _finite_number(value)
            else model.display_value(model.default_config(), path)
        )
        numbers.append(value)
        sliders.append(slider)
        disabled.append(path in ANGLE_PATHS and model.is_linked(config, path))
        lows.append(min(spec["min"], slider))
        highs.append(max(spec["max"], slider))
    links = [
        ["linked"] if model.is_linked(config, path) else [] for path in ANGLE_PATHS
    ]
    return numbers, sliders, disabled, lows, highs, links


def _field(path, config):
    spec = model.FIELD_SPECS[path]
    value = model.display_value(config, path)
    linked = path in ANGLE_PATHS and model.is_linked(config, path)
    label = spec["label"] + (f" [{spec['unit']}]" if spec["unit"] else "")
    children = [
        html.Div(
            [
                html.Label(
                    label,
                    htmlFor=json.dumps(
                        {"kind": "number", "path": path},
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    title=spec["help"],
                ),
                dcc.Input(
                    id={"kind": "number", "path": path},
                    type="number",
                    value=value,
                    debounce=True,
                    step=spec["step"],
                    disabled=linked,
                    className="mc-number",
                    persistence=False,
                ),
            ],
            className="mc-field-heading",
        ),
        dcc.Slider(
            id={"kind": "slider", "path": path},
            value=value,
            min=min(spec["min"], value),
            max=max(spec["max"], value),
            step=spec["step"],
            marks=None,
            updatemode="drag",
            disabled=linked,
            tooltip={"placement": "bottom", "always_visible": False},
            className="mc-slider",
        ),
    ]
    if path in ANGLE_PATHS:
        station = "inlet" if ".inlet_" in path else "outlet"
        children.append(
            dcc.Checklist(
                id={"kind": "link", "path": path},
                options=[{"label": f"Follow {station} direction", "value": "linked"}],
                value=["linked"] if linked else [],
                className="mc-follow",
            )
        )
    children.append(html.P(spec["help"], className="mc-field-help"))
    return html.Div(children, className="mc-field", **{"data-parameter": path})


def _metrics(items=None):
    items = items or [
        {"label": label, "value": "—", "detail": "Current inputs invalid"}
        for label in (
            "Inlet area",
            "Outlet area",
            "Minimum connector",
            "Wall lengths · hub / shroud",
        )
    ]
    return [
        html.Div(
            [
                html.Span(item["label"], className="mc-metric-label"),
                html.Strong(item["value"]),
                html.Small(item.get("detail", "")),
            ],
            className="mc-metric",
        )
        for item in items
    ]


def uploaded_config(contents):
    """Decode a small local YAML upload before changing any browser state."""
    try:
        encoded = contents.split(",", 1)[1]
        if len(encoded) > 1_400_000:
            raise ValueError("Choose a YAML file smaller than 1 MB.")
        return model.parse_yaml(
            base64.b64decode(encoded, validate=True).decode("utf-8-sig")
        )
    except (IndexError, AttributeError, binascii.Error, UnicodeError) as error:
        raise ValueError("Could not read the uploaded YAML file.") from error


def create_app(initial_config=None):
    config = model.normalize_config(
        model.default_config() if initial_config is None else initial_config
    )
    reference = model.default_config()
    initial = model.evaluate_config(config, reference=reference)
    if not initial["valid"]:
        raise ValueError(f"The initial configuration is invalid: {initial['error']}")
    config = initial["config"]
    app = Dash(
        __name__,
        assets_folder=str(HERE / "assets"),
        title="Meridional channel explorer",
    )
    app.layout = html.Div(
        [
            dcc.Store(id="draft-config", data=config),
            dcc.Store(id="reference-config", data=reference),
            dcc.Store(id="generation-state", data={"valid": True}),
            dcc.Download(id="yaml-download"),
            html.Header(
                [
                    html.Div(
                        [
                            html.Span("◒", className="mc-mark"),
                            html.Strong("TurboDash"),
                            html.Span("/", className="mc-divider"),
                            html.Span("Meridional channel"),
                        ],
                        className="mc-brand",
                    ),
                    html.Span("GEOMETRY EXPLORER", className="mc-tag"),
                ],
                className="mc-header",
            ),
            html.Div(
                [
                    html.Aside(
                        [
                            html.Div(
                                [
                                    html.H2("Engineering controls"),
                                    html.P(
                                        "Drag to explore. Type a value for a precise edit."
                                    ),
                                ],
                                className="mc-sidebar-heading",
                            ),
                            html.Div(
                                [
                                    html.Label(
                                        "Starting geometry", className="mc-small-label"
                                    ),
                                    dcc.Dropdown(
                                        id="preset",
                                        options=PRESETS,
                                        value="baseline",
                                        clearable=False,
                                        searchable=False,
                                    ),
                                    html.Div(
                                        [
                                            html.Button(
                                                "Load preset",
                                                id="load-preset",
                                                className="mc-button mc-button-primary",
                                            ),
                                            html.Button(
                                                "Reset baseline",
                                                id="reset-all",
                                                className="mc-button",
                                            ),
                                        ],
                                        className="mc-action-row",
                                    ),
                                    html.Div(
                                        [
                                            dcc.Upload(
                                                id="yaml-upload",
                                                children=html.Button(
                                                    "Load YAML", className="mc-button"
                                                ),
                                                accept=".yaml,.yml",
                                                multiple=False,
                                            ),
                                            html.Button(
                                                "Save YAML",
                                                id="save-yaml",
                                                className="mc-button",
                                            ),
                                        ],
                                        className="mc-action-row",
                                    ),
                                    html.Div(
                                        id="file-notice",
                                        className="mc-file-notice",
                                        role="status",
                                    ),
                                ],
                                className="mc-preset-box",
                            ),
                            *[
                                html.Details(
                                    [
                                        html.Summary(group),
                                        html.Div(
                                            [
                                                _field(path, config)
                                                for path in FIELD_PATHS
                                                if model.FIELD_SPECS[path]["group"]
                                                == group
                                            ],
                                            className="mc-group-body",
                                        ),
                                    ],
                                    open=index < 2,
                                    className="mc-group",
                                )
                                for index, group in enumerate(model.GROUPS)
                            ],
                            html.P(
                                "Dimensions display in mm. YAML stores lengths in metres. Angles are measured from +z towards +r.",
                                className="mc-sidebar-note",
                            ),
                        ],
                        className="mc-sidebar",
                    ),
                    html.Main(
                        [
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.P(
                                                "CENTRIFUGAL / MIXED-FLOW IMPELLER",
                                                className="mc-eyebrow",
                                            ),
                                            html.H1("Shape the meridional channel"),
                                            html.P(
                                                "Independent walls. Independent blade edges. One live geometry.",
                                                className="mc-subtitle",
                                            ),
                                        ]
                                    ),
                                    html.Div(
                                        "Valid geometry",
                                        id="status-chip",
                                        className="mc-status mc-status-valid",
                                        role="status",
                                    ),
                                ],
                                className="mc-main-heading",
                            ),
                            html.Div(
                                initial["message"],
                                id="generation-notice",
                                className="mc-notice",
                                role="status",
                            ),
                            html.Div(
                                _metrics(initial["metrics"]),
                                id="metrics",
                                className="mc-metrics",
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.Div(
                                                        [
                                                            html.H3(
                                                                "Meridional geometry"
                                                            ),
                                                            html.P(
                                                                "Hub, shroud, main-blade extent, and splitter leading edge"
                                                            ),
                                                        ]
                                                    ),
                                                    html.Span(
                                                        "z–r / mm", className="mc-units"
                                                    ),
                                                ],
                                                className="mc-card-heading",
                                            ),
                                            html.Div(
                                                [
                                                    dcc.Checklist(
                                                        id="show-reference",
                                                        options=[
                                                            {
                                                                "label": "Reference overlay",
                                                                "value": "show",
                                                            }
                                                        ],
                                                        value=["show"],
                                                        className="mc-check",
                                                    ),
                                                    html.Button(
                                                        "Pin current",
                                                        id="pin-reference",
                                                        className="mc-button mc-button-small",
                                                    ),
                                                    html.Button(
                                                        "Use baseline",
                                                        id="reset-reference",
                                                        className="mc-button mc-button-small",
                                                    ),
                                                    html.Span(
                                                        "Baseline reference",
                                                        id="reference-caption",
                                                        className="mc-reference-caption",
                                                    ),
                                                ],
                                                className="mc-reference-bar",
                                            ),
                                            dcc.Graph(
                                                id="plot-channel",
                                                figure=initial["figures"]["channel"],
                                                config=GRAPH_CONFIG,
                                                className="mc-channel-plot",
                                            ),
                                            html.P(
                                                "Use the plot toolbar to zoom, fit, or download an SVG. A reference remains fixed while you edit.",
                                                className="mc-card-footnote",
                                            ),
                                        ],
                                        className="mc-card mc-geometry-card",
                                    ),
                                    html.Div(
                                        [
                                            html.Div(
                                                [
                                                    html.H3("Passage diagnostics"),
                                                    html.P(
                                                        "See how each edit changes the flow-path geometry"
                                                    ),
                                                ],
                                                className="mc-card-heading",
                                            ),
                                            dcc.Tabs(
                                                id="diagnostic-tab",
                                                value="area",
                                                className="mc-tabs",
                                                children=[
                                                    dcc.Tab(
                                                        label=label,
                                                        value=name,
                                                        className="mc-tab",
                                                        selected_className="mc-tab-selected",
                                                        children=[
                                                            dcc.Graph(
                                                                id=f"plot-{name}",
                                                                figure=initial[
                                                                    "figures"
                                                                ][name],
                                                                config=GRAPH_CONFIG,
                                                                className="mc-diagnostic-plot",
                                                            ),
                                                        ],
                                                    )
                                                    for name, label in (
                                                        ("area", "Area"),
                                                        ("curvature", "Curvature"),
                                                        ("width", "Width"),
                                                    )
                                                ],
                                            ),
                                            html.Div(
                                                [
                                                    html.H4("Reading the geometry"),
                                                    html.P(
                                                        "Wall angles can follow the station direction or be tuned independently. Uncheck “Follow” to edit a wall angle."
                                                    ),
                                                    html.P(
                                                        "Blade-edge positions use normalized wall arclength. Changing them moves the blade inside the channel without moving either wall."
                                                    ),
                                                    html.P(
                                                        "Area excludes blade blockage. These plots describe geometry; they do not predict throat area, efficiency, surge, or choking."
                                                    ),
                                                ],
                                                className="mc-reading-note",
                                            ),
                                        ],
                                        className="mc-card mc-diagnostics-card",
                                    ),
                                ],
                                className="mc-plots",
                            ),
                            html.Footer(
                                "Illustrative baseline · inspect actual blade inlet/outlet dimensions after moving the edges.",
                                className="mc-footer",
                            ),
                        ],
                        className="mc-main",
                    ),
                ],
                className="mc-workspace",
            ),
        ],
        className="mc-app",
    )

    @app.callback(
        Output("draft-config", "data"),
        Output({"kind": "number", "path": ALL}, "value"),
        Output({"kind": "slider", "path": ALL}, "value"),
        Output({"kind": "number", "path": ALL}, "disabled"),
        Output({"kind": "slider", "path": ALL}, "disabled"),
        Output({"kind": "slider", "path": ALL}, "min"),
        Output({"kind": "slider", "path": ALL}, "max"),
        Output({"kind": "link", "path": ALL}, "value"),
        Output("file-notice", "children"),
        Input({"kind": "number", "path": ALL}, "value"),
        Input({"kind": "slider", "path": ALL}, "value"),
        Input({"kind": "link", "path": ALL}, "value"),
        Input("load-preset", "n_clicks"),
        Input("reset-all", "n_clicks"),
        Input("yaml-upload", "contents"),
        State("draft-config", "data"),
        State("preset", "value"),
        State("yaml-upload", "filename"),
        prevent_initial_call=True,
    )
    def synchronize(
        numbers,
        sliders,
        links,
        load_clicks,
        reset_clicks,
        contents,
        draft,
        preset,
        filename,
    ):
        trigger = ctx.triggered_id
        updated = deepcopy(draft)
        notice = ""
        try:
            if isinstance(trigger, dict):
                path, kind = trigger["path"], trigger["kind"]
                if kind == "link":
                    updated = model.set_link(
                        updated,
                        path,
                        "linked" in (links[ANGLE_PATHS.index(path)] or []),
                    )
                else:
                    if path in ANGLE_PATHS and model.is_linked(updated, path):
                        raise PreventUpdate
                    values = numbers if kind == "number" else sliders
                    value = values[FIELD_PATHS.index(path)]
                    if value is None:
                        # A cleared widget is an invalid draft, not the YAML
                        # null convention for inherited angles/default length.
                        if "." in path:
                            group, name = path.split(".", 1)
                            updated[group][name] = ""
                        else:
                            updated[path] = ""
                    else:
                        updated = model.set_display_value(updated, path, value)
            elif trigger == "load-preset":
                updated = model.preset_config(preset)
                notice = "Preset loaded. All controls now describe this geometry."
            elif trigger == "reset-all":
                updated = model.default_config()
                notice = "Reset to the centrifugal baseline."
            elif trigger == "yaml-upload":
                candidate = uploaded_config(contents)
                validation = model.evaluate_config(candidate)
                if not validation["valid"]:
                    raise ValueError(validation["error"])
                updated = validation["config"]
                notice = f"Loaded {Path(filename or 'configuration.yaml').name}."
            else:
                raise PreventUpdate
        except (ValueError, TypeError, KeyError) as error:
            # Dash expands ALL outputs into lists of concrete widget outputs.
            # Every expanded list needs its own no_update entries; a scalar
            # sentinel fails callback return validation on these Dash versions.
            return (
                no_update,
                *[[no_update] * len(FIELD_PATHS) for _ in range(6)],
                [no_update] * len(ANGLE_PATHS),
                f"Could not apply this change: {error}",
            )
        numbers, sliders, disabled, lows, highs, links = _control_view(updated)
        return updated, numbers, sliders, disabled, disabled, lows, highs, links, notice

    @app.callback(
        *[Output(f"plot-{name}", "figure") for name in GRAPH_NAMES],
        Output("metrics", "children"),
        Output("generation-state", "data"),
        Output("generation-notice", "children"),
        Output("generation-notice", "className"),
        Output("status-chip", "children"),
        Output("status-chip", "className"),
        Output("save-yaml", "disabled"),
        Output("pin-reference", "disabled"),
        Input("draft-config", "data"),
        Input("reference-config", "data"),
        Input("show-reference", "value"),
        prevent_initial_call=True,
    )
    def generate(draft, reference, selection):
        result = model.evaluate_config(
            draft, reference=reference if "show" in (selection or []) else None
        )
        if not result["valid"]:
            return (
                *([no_update] * 4),
                _metrics(),
                {"valid": False},
                f"Last valid geometry is shown. Current inputs were rejected: {result['error']}",
                "mc-notice mc-notice-error",
                "Invalid inputs · view is stale",
                "mc-status mc-status-error",
                True,
                True,
            )
        return (
            *[result["figures"][name] for name in GRAPH_NAMES],
            _metrics(result["metrics"]),
            {"valid": True},
            result["message"],
            "mc-notice",
            "Valid geometry",
            "mc-status mc-status-valid",
            False,
            False,
        )

    @app.callback(
        Output("reference-config", "data"),
        Output("reference-caption", "children"),
        Input("pin-reference", "n_clicks"),
        Input("reset-reference", "n_clicks"),
        State("draft-config", "data"),
        prevent_initial_call=True,
    )
    def reference_action(pin_clicks, reset_clicks, draft):
        if ctx.triggered_id == "reset-reference":
            return model.default_config(), "Baseline reference"
        result = model.evaluate_config(draft)
        if not result["valid"]:
            raise PreventUpdate
        return result["config"], "Pinned reference"

    @app.callback(
        Output("yaml-download", "data"),
        Input("save-yaml", "n_clicks"),
        State("draft-config", "data"),
        prevent_initial_call=True,
    )
    def save_yaml(clicks, draft):
        if not clicks:
            raise PreventUpdate
        result = model.evaluate_config(draft)
        if not result["valid"]:
            raise PreventUpdate
        return {
            "content": model.dump_yaml(result["config"]),
            "filename": "meridional_channel.yaml",
            "type": "application/yaml",
        }

    return app


def _check_server_address(host, port):
    import socket

    if not 1 <= port <= 65535:
        raise ValueError("Port must be between 1 and 65535.")
    for family, kind, protocol, _, address in socket.getaddrinfo(
        host, port, type=socket.SOCK_STREAM
    ):
        with socket.socket(family, kind, protocol) as probe:
            if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            probe.bind(address)


def _open_browser_when_ready(url):
    import time
    from urllib.error import URLError
    from urllib.request import urlopen
    import webbrowser

    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        try:
            with urlopen(url + "_dash-layout", timeout=1) as response:
                if response.status == 200:
                    webbrowser.open(url)
                    return
        except (URLError, OSError):
            pass
        time.sleep(0.2)


def main(argv=None, *, open_browser=False):
    parser = argparse.ArgumentParser(
        description="Explore centrifugal and mixed-flow meridional geometry"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8065, type=int)
    parser.add_argument("--config", type=Path, help="Initial YAML configuration")
    parser.add_argument(
        "--preset", choices=("baseline", "inclined", "romei"), default="baseline"
    )
    parser.add_argument("--open-browser", action="store_true", default=open_browser)
    parser.add_argument(
        "--no-browser", action="store_true", help="Serve without opening a browser"
    )
    args = parser.parse_args(argv)
    try:
        _check_server_address(args.host, args.port)
        config = (
            model.parse_yaml(args.config.read_text(encoding="utf-8-sig"))
            if args.config
            else model.preset_config(args.preset)
        )
        app = create_app(config)
    except (OSError, ValueError) as error:
        parser.error(
            f"Cannot start: {error}. If the address is occupied, choose another --port."
        )
    if args.open_browser and not args.no_browser:
        from threading import Thread

        browser_host = "127.0.0.1" if args.host in {"0.0.0.0", "::"} else args.host
        if ":" in browser_host:
            browser_host = f"[{browser_host}]"
        Thread(
            target=_open_browser_when_ready,
            args=(f"http://{browser_host}:{args.port}/",),
            daemon=True,
        ).start()
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
