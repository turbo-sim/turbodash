"""Serve the installed TurboMoC VTK renderer without copying its implementation.

TurboMoC owns the stable React/VTK actor lifecycle, cell-edge shaders,
quality coloring, camera controls and hidden-pane resizing. Serve its renderer
scripts without loading its application-wide styles. Newer installations may
also supply an orientation widget; version 1.2.0 does not require one.
"""

from importlib import import_module, util
from pathlib import Path

from flask import Response, abort


_SCRIPTS = ("mesh_edges.js", "studio.js")
_ORIENTATION = "orientation_cube.js"
_PREFIX = "/_turbodash_vtk/"


def _asset_directory():
    specification = util.find_spec("turbo_moc")
    if specification is None or not specification.submodule_search_locations:
        return None
    return Path(next(iter(specification.submodule_search_locations))) / "app" / "assets"


def available():
    directory = _asset_directory()
    if not (directory and util.find_spec("dash_vtk")
            and all((directory / name).is_file() for name in _SCRIPTS)):
        return False
    # A partial newer installation must not mount a renderer whose required
    # navigation dependency is missing. The published 1.2.0 bundle has none.
    try:
        needs_orientation = "turbomocOrientation" in (directory / "studio.js").read_text(encoding="utf-8")
    except OSError:
        return False
    return not needs_orientation or (directory / _ORIENTATION).is_file()


def _scripts(directory):
    optional = (_ORIENTATION,) if directory and (directory / _ORIENTATION).is_file() else ()
    return (_SCRIPTS[0], *optional, _SCRIPTS[1])


def register_vtk_assets(app):
    """Register optional VTK bundles before Dash creates its first index page."""
    if app.server.extensions.get("turbodash_vtk"):
        return True
    if not available():
        return False

    # TurboMoC may lazily import pandas in the first 3D callback. Plotly checks
    # sys.modules without waiting for an in-progress optional import, so its
    # simultaneous 2D callback could observe pandas before Series is defined.
    # Complete the ordinary import before serving callbacks, when installed;
    # pandas remains optional for installations that do not provide it.
    if util.find_spec("pandas") is not None:
        import_module("pandas")

    # Importing registers the dash-vtk component namespace and browser bundles.
    import dash_vtk  # noqa: F401

    scripts = _scripts(_asset_directory())

    @app.server.get(_PREFIX + "<path:filename>", endpoint="turbodash_vtk_asset")
    def renderer_asset(filename):
        if filename not in scripts:
            abort(404)
        directory = _asset_directory()
        if directory is None:
            abort(404)
        try:
            contents = (directory / filename).read_text(encoding="utf-8")
        except OSError:
            abort(404)
        return Response(contents, mimetype="application/javascript")

    app.config.external_scripts.extend(_PREFIX + name for name in scripts)
    app.server.extensions["turbodash_vtk"] = True
    return True
