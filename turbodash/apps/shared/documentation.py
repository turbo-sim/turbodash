"""Serve a prebuilt Sphinx site alongside a Dash application."""

import os
from pathlib import Path

from flask import Response, abort, redirect, send_file


def register_documentation(app, build_directory=None):
    """Register documentation routes and return their browser-facing URL.

    Build the documentation before starting the application. Source checkouts
    use ``docs/_build/html`` by default; deployments can provide another build
    directory explicitly or through ``TURBODASH_DOCS_DIR``.

    Dash's server and browser prefixes can differ behind a reverse proxy, so
    routes use ``routes_pathname_prefix`` and links use ``get_relative_path``.
    """
    if build_directory is None:
        build_directory = os.environ.get("TURBODASH_DOCS_DIR") or (
            Path(__file__).resolve().parents[3] / "docs" / "_build" / "html"
        )
    directory = Path(build_directory).resolve()
    route = app.config.routes_pathname_prefix.rstrip("/") + "/docs"
    browser_url = app.get_relative_path("/docs/")

    def serve_documentation(filename="index.html"):
        # Backslashes and drive names are filesystem paths, never site URLs.
        # Resolving before checking containment also rejects external symlinks.
        if (
            "\\" in filename
            or ":" in filename
            or any(part.startswith(".") for part in filename.split("/"))
        ):
            abort(404)
        try:
            path = (directory / filename).resolve()
            if path.is_dir():
                path = (path / "index.html").resolve()
            if not path.is_relative_to(directory):
                abort(404)
        except (OSError, RuntimeError, ValueError):
            abort(404)

        if not (directory / "index.html").is_file():
            return Response(
                "<!doctype html><html lang=\"en\"><head>"
                "<meta charset=\"utf-8\">"
                "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
                "<title>Turbodash documentation</title></head><body>"
                "<main><h1>Documentation is not available yet</h1>"
                "<p>The documentation has not been built for this installation.</p>"
                "<p>Build it with <code>poetry run python docs/build_docs.py "
                "--no-autobuild</code> "
                "from the repository root, then reload this page.</p>"
                "</main></body></html>",
                status=503,
                mimetype="text/html",
            )
        if not path.is_file():
            abort(404)
        return send_file(path)

    endpoint = f"turbodash_documentation:{route}"
    app.server.add_url_rule(
        route,
        endpoint + ":redirect",
        lambda: redirect(browser_url, code=308),
    )
    app.server.add_url_rule(route + "/", endpoint, serve_documentation)
    app.server.add_url_rule(
        route + "/<path:filename>", endpoint + ":file", serve_documentation
    )
    return browser_url
