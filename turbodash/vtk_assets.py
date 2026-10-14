"""Serve the installed TurboMoC VTK renderer without copying its implementation.

The pinned package owns the stable React/VTK actor lifecycle, cell-edge shaders,
quality coloring, camera controls and hidden-pane resizing. Only its orientation
widget CSS is included; its application-wide styles do not belong in TurboDash.
"""

from importlib import util
from pathlib import Path
import re

from flask import Response, abort


_SCRIPTS = ("mesh_edges.js", "orientation_cube.js", "studio.js")
_PREFIX = "/_turbodash_vtk/"


def _asset_directory():
    specification = util.find_spec("turbo_moc")
    if specification is None or not specification.submodule_search_locations:
        return None
    return Path(next(iter(specification.submodule_search_locations))) / "app" / "assets"


def available():
    directory = _asset_directory()
    return bool(directory and util.find_spec("dash_vtk")
                and all((directory / name).is_file() for name in (*_SCRIPTS, "studio.css")))


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
        if filename not in (*_SCRIPTS, "orientation.css"):
            abort(404)
        directory = _asset_directory()
        if directory is None:
            abort(404)
        try:
            if filename == "orientation.css":
                stylesheet = (directory / "studio.css").read_text(encoding="utf-8")
                # These rules belong entirely to the shared navigation widget.
                # Serve them from the installed version, keeping its markup and
                # styles in sync without loading unrelated page/layout rules.
                contents = "\n".join(re.findall(r"\.orientation-[^{}]*\{[^{}]*\}", stylesheet))
                mimetype = "text/css"
            else:
                contents = (directory / filename).read_text(encoding="utf-8")
                mimetype = "application/javascript"
        except OSError:
            abort(404)
        return Response(contents, mimetype=mimetype)

    app.config.external_scripts.extend(_PREFIX + name for name in _SCRIPTS)
    app.config.external_stylesheets.append(_PREFIX + "orientation.css")
    app.server.extensions["turbodash_vtk"] = True
    return True
