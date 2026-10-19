"""Serve an app version's own assets alongside the common Dash asset folder."""

from functools import partial
from pathlib import Path

from flask import send_file


def register_app_assets(app, directory, namespace):
    """Expose only concrete CSS/JS files from an explicitly selected directory."""
    directory = Path(directory).resolve()
    prefix = app.config.requests_pathname_prefix.rstrip("/")
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix not in (".css", ".js"):
            continue
        endpoint = f"app_asset_{namespace}_{path.name}"
        if endpoint in app.server.view_functions:
            continue
        route = f"/app-assets/{namespace}/{path.name}"
        app.server.add_url_rule(route, endpoint, partial(send_file, path))
        collection = app.config.external_stylesheets if path.suffix == ".css" else app.config.external_scripts
        collection.append(prefix + route)
