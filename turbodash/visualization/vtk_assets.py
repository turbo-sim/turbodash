"""Serve the installed TurboMoC VTK renderer without copying its implementation.

TurboMoC 1.2.0 owns the stable React/VTK actor lifecycle, cell-edge shaders,
quality coloring, camera controls and hidden-pane resizing. Serve its two
renderer scripts without loading its application-wide styles.
"""

from importlib import util
from pathlib import Path

from flask import Response, abort


_SCRIPTS = ("mesh_edges.js", "studio.js")
_PREFIX = "/_turbodash_vtk/"


def _asset_directory():
    specification = util.find_spec("turbo_moc")
    if specification is None or not specification.submodule_search_locations:
        return None
    return Path(next(iter(specification.submodule_search_locations))) / "app" / "assets"


def available():
    directory = _asset_directory()
    return bool(directory and util.find_spec("dash_vtk")
                and all((directory / name).is_file() for name in _SCRIPTS))


def register_vtk_assets(app):
    """Register optional VTK bundles before Dash creates its first index page."""
    if app.server.extensions.get("turbodash_vtk"):
        return True
    if not available():
        return False

    # Importing registers the dash-vtk component namespace and browser bundles.
    import dash_vtk  # noqa: F401

    @app.server.get(_PREFIX + "<path:filename>", endpoint="turbodash_vtk_asset")
    def renderer_asset(filename):
        if filename not in _SCRIPTS:
            abort(404)
        directory = _asset_directory()
        if directory is None:
            abort(404)
        try:
            contents = (directory / filename).read_text(encoding="utf-8")
        except OSError:
            abort(404)
        return Response(contents, mimetype="application/javascript")

    app.config.external_scripts.extend(_PREFIX + name for name in _SCRIPTS)
    app.server.extensions["turbodash_vtk"] = True
    return True
