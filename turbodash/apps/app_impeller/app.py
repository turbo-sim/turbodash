"""Interactive impeller geometry using Turbodash's kernels and shared editors.

Importing the module starts no server. create_app() creates an independent
Dash instance; all design state lives in that browser's stores.
"""

import argparse
from pathlib import Path

from dash import ALL, ClientsideFunction, Dash, Input, Output, State, ctx, dcc
from dash.exceptions import PreventUpdate
from flask import Response, send_file

from turbodash.apps.shared.assets import register_app_assets
from turbodash.apps.shared.paths import SHARED_ASSETS_DIRECTORY
from . import controls, layout, model, viewer, walls

HERE = Path(__file__).resolve().parent


def create_app(initial_config=None):
    initial = model.evaluate(
        model.default_config() if initial_config is None else initial_config, 0
    )
    if not initial["valid"]:
        raise ValueError(f"Invalid initial impeller: {initial['error']}")
    config = initial["config"]
    app = Dash(
        __name__,
        title="Turbodash · Impeller",
        update_title=None,
        assets_folder=str(HERE / "assets"),
    )
    register_app_assets(app, SHARED_ASSETS_DIRECTORY, "impeller-shared")
    viewer.register(app)
    wall_layout = walls.register(app, config)
    stores = [
        dcc.Store(id="draft-config", data=config),
        dcc.Store(id="impeller-baseline", data=config),
        dcc.Store(id="impeller-revision", data=0),
        dcc.Store(id="impeller-fields", data=controls.FIELD_SPECS),
        dcc.Store(id="impeller-evaluation", data=initial),
        dcc.Store(id="impeller-accepted"),
        dcc.Store(id="impeller-config-replacement"),
        dcc.Store(id="impeller-viewer-data"),
        dcc.Store(id="wall-commit"),
        dcc.Store(id="impeller-stl-request"),
        dcc.Store(id="impeller-stl-result"),
    ]
    app.layout = layout.build_layout(
        config,
        initial["figures"],
        wall_layout,
        viewer.viewer_layout(app),
        stores=stores,
    )
    numbers = {"type": "impeller-number", "path": ALL}
    sliders = {"type": "impeller-slider", "path": ALL}
    app.clientside_callback(
        ClientsideFunction(namespace="impeller", function_name="edit"),
        Output("draft-config", "data"),
        Output("impeller-revision", "data"),
        Output(numbers, "value"),
        Output(sliders, "value"),
        Output(sliders, "min"),
        Output(sliders, "max"),
        Output(numbers, "disabled"),
        Output(sliders, "disabled"),
        Output("impeller-rake-mode", "value"),
        Output("impeller-angle-interpolation", "value"),
        Output("impeller-angle-interpolation", "disabled"),
        Output("impeller-stacking-help", "children"),
        Output("wall-notice", "children"),
        Input(numbers, "value"),
        Input(sliders, "value"),
        Input("impeller-rake-mode", "value"),
        Input("impeller-angle-interpolation", "value"),
        Input("impeller-reset", "n_clicks"),
        Input("impeller-config-replacement", "data"),
        State("draft-config", "data"),
        State("impeller-revision", "data"),
        State("impeller-baseline", "data"),
        State("impeller-fields", "data"),
        State(numbers, "id"),
        State(sliders, "id"),
        prevent_initial_call=True,
    )

    @app.callback(
        Output("impeller-evaluation", "data"),
        Input("impeller-revision", "data"),
        State("draft-config", "data"),
        prevent_initial_call=True,
    )
    def evaluate(revision, draft):
        return model.evaluate(draft, revision)

    app.clientside_callback(
        ClientsideFunction(namespace="impeller", function_name="accept"),
        Output("impeller-angle-plot", "figure"),
        Output("impeller-thickness-plot", "figure"),
        Output("impeller-channel-plot", "figure"),
        Output("impeller-stacking-plot", "figure"),
        Output("impeller-metrics", "children"),
        Output("impeller-viewer-data", "data"),
        Output("impeller-accepted", "data"),
        Output("impeller-status", "children"),
        Output("impeller-error", "children"),
        Output("impeller-yaml-download-button", "disabled"),
        Input("impeller-evaluation", "data"),
        State("impeller-revision", "data"),
    )

    @app.callback(
        Output("impeller-config-replacement", "data"),
        Input("impeller-yaml-upload", "contents"),
        Input("wall-commit", "data"),
        Input("wall-clear", "n_clicks"),
        State("draft-config", "data"),
        State("impeller-revision", "data"),
        prevent_initial_call=True,
    )
    def replace(contents, commit, clear, draft, revision):
        try:
            if ctx.triggered_id == "impeller-yaml-upload":
                if not contents:
                    raise PreventUpdate
                candidate = model.uploaded_config(contents)
                notice = (
                    "Validated YAML loaded. Hub and shroud controls are synchronized."
                )
            else:
                candidate = walls.apply_commit(
                    draft, commit, clear=ctx.triggered_id == "wall-clear"
                )
                candidate = model.validate_config(candidate).config
                notice = (
                    "Wall refinements cleared."
                    if ctx.triggered_id == "wall-clear"
                    else "Wall edits applied to the complete impeller."
                )
            return {"base_revision": revision, "config": candidate, "notice": notice}
        except (ValueError, TypeError, KeyError, IndexError, OverflowError) as error:
            return {"base_revision": revision, "error": str(error)}

    @app.callback(
        Output("impeller-yaml-download", "data"),
        Input("impeller-yaml-download-button", "n_clicks"),
        State("impeller-accepted", "data"),
        State("impeller-revision", "data"),
        prevent_initial_call=True,
    )
    def download(clicks, accepted, revision):
        if not clicks:
            raise PreventUpdate
        try:
            text = model.downloadable_yaml(accepted, revision)
        except ValueError:
            raise PreventUpdate
        return dcc.send_string(text, "impeller_geometry.yaml")

    app.clientside_callback(
        ClientsideFunction(namespace="impeller", function_name="requestStl"),
        Output("impeller-stl-request", "data"),
        Input("impeller-stl-download-button", "n_clicks"),
        State("impeller-accepted", "data"),
        State("impeller-revision", "data"),
        prevent_initial_call=True,
    )

    @app.callback(
        Output("impeller-stl-result", "data"),
        Input("impeller-stl-request", "data"),
        prevent_initial_call=True,
    )
    def export_stl(request):
        if not request:
            raise PreventUpdate
        return model.stl_export_packet(request)

    app.clientside_callback(
        ClientsideFunction(namespace="impeller", function_name="completeStl"),
        Output("impeller-stl-download", "data"),
        Input("impeller-stl-result", "data"),
        State("impeller-accepted", "data"),
        State("impeller-revision", "data"),
        prevent_initial_call=True,
    )

    @app.server.get("/_impeller/docs")
    def documentation():
        return Response(
            """<!doctype html><html><head><meta charset="utf-8"><title>Impeller documentation</title>
<style>body{font:16px/1.6 'Segoe UI',sans-serif;max-width:760px;margin:60px auto;padding:0 24px;color:#233541}a{color:#176b89}li{margin:18px 0}</style></head><body>
<h1>Impeller geometry</h1><p>The app uses the same channel, blade and stacking construction as the script workflow.</p>
<ul><li><a href="/_impeller/docs/README.md">Workflow and controls (Markdown)</a></li>
<li><a href="/_impeller/docs/impeller_geometry_theory.md">Theory and equations (Markdown)</a></li></ul>
<p>Move sliders or type values to update the geometry. Use the native wall editor to drag interior hub or shroud points, then apply the refinements. Invalid drafts retain the last valid geometry and disable configuration export.</p></body></html>""",
            mimetype="text/html",
        )

    @app.server.get("/_impeller/docs/<filename>")
    def document(filename):
        from flask import abort

        if filename not in {"README.md", "impeller_geometry_theory.md"}:
            abort(404)
        path = model.DEFAULT_CASE.parents[1] / filename
        if not path.is_file():
            abort(404)
        return send_file(path, mimetype="text/plain; charset=utf-8")

    return app


def cli(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8052)
    parser.add_argument("--open-browser", action="store_true")
    args = parser.parse_args(argv)
    app = create_app(model.default_config(args.config))
    if args.open_browser:
        import os
        from turbodash.apps.shared.cli import run_app

        os.environ["HOST"], os.environ["PORT"] = args.host, str(args.port)
        return run_app(app, open_browser=True)
    return app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    cli()
