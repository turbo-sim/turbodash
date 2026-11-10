"""Bundle the existing repository viewers into an offline impeller report.

Three.js r180 and TrackballControls are read from nurbspy's pinned upstream
assets, without importing nurbspy or its geometry. The navigation cube remains
owned by TurboMoC, and its styling remains owned by Turbodash's shared viewer.
No network request is made while generating or opening the report.
"""

from __future__ import annotations

import base64
import hashlib
from importlib import util
import json
import os
from pathlib import Path


class ViewerAssetsUnavailable(RuntimeError):
    """A companion checkout lacks the browser assets required by the report."""


def _nurbspy_assets():
    override = os.environ.get("NURBSPY_WEB_ASSETS")
    candidates = [Path(override).expanduser()] if override else []
    if not override:
        specification = util.find_spec("nurbspy")
        if specification and specification.submodule_search_locations:
            candidates.extend(
                Path(item) / "web_assets"
                for item in specification.submodule_search_locations
            )
        candidates.extend(
            parent / "nurbspy" / "src" / "nurbspy" / "web_assets"
            for parent in Path(__file__).resolve().parents
        )
    required = (
        "three.core.js",
        "three.module.js",
        "TrackballControls.js",
        "LICENSE",
        "manifest.json",
    )
    for directory in candidates:
        vendor = directory / "vendor"
        if all((vendor / name).is_file() for name in required):
            manifest = json.loads(
                (vendor / "manifest.json").read_text(encoding="utf-8")
            )
            if manifest.get("version") != "0.180.0":
                raise ViewerAssetsUnavailable(
                    "The impeller report expects Three.js 0.180.0 assets; point NURBSPY_WEB_ASSETS to the compatible nurbspy web_assets directory."
                )
            for name in required[:-1]:
                expected = manifest.get("sha256", {}).get(name)
                if (
                    not expected
                    or hashlib.sha256((vendor / name).read_bytes()).hexdigest()
                    != expected
                ):
                    raise ViewerAssetsUnavailable(
                        f"The pinned Three.js asset {name} does not match nurbspy's manifest. Restore the companion assets before exporting HTML."
                    )
            return directory
    raise ViewerAssetsUnavailable(
        "Offline impeller HTML needs the pinned Three.js assets from the companion nurbspy checkout. Install that checkout, place it beside turbodash, or set NURBSPY_WEB_ASSETS to its src/nurbspy/web_assets directory. Geometry exports can still use write_html=False or --no-html."
    )


def _orientation_source():
    from turbodash.geometry.blade_parametrization.turbomoc_adapter import (
        TurboMOCUnavailable,
        renderer_asset_directory,
    )

    candidates = []
    try:
        candidates.append(renderer_asset_directory())
    except TurboMOCUnavailable:
        if os.environ.get("TURBOMOC_SOURCE"):
            raise
    # Published TurboMoC 1.2.0 predates the cube. Prefer the installed renderer
    # when capable, otherwise reuse the newer companion source without imports.
    if not os.environ.get("TURBOMOC_SOURCE"):
        candidates.extend(
            parent
            / "method_of_characteristics"
            / "src"
            / "turbo_moc"
            / "app"
            / "assets"
            for parent in Path(__file__).resolve().parents
        )
    for directory in candidates:
        path = directory / "orientation_cube.js"
        if path.is_file():
            source = path.read_text(encoding="utf-8")
            if "turbomocOrientation" not in source:
                raise ViewerAssetsUnavailable(
                    f"Unsupported orientation cube at {path}."
                )
            return path, source
    raise ViewerAssetsUnavailable(
        "The impeller report needs TurboMoC's orientation_cube.js. Set TURBOMOC_SOURCE to the companion method_of_characteristics checkout containing app/assets/orientation_cube.js; the published 1.2.0 assets predate the cube."
    )


def _script(source):
    return source.replace("</script", "<\\/script").replace("</SCRIPT", "<\\/SCRIPT")


def _cube_license(cube_path):
    """Read the owner repository/package license rather than maintaining a copy."""
    for parent in list(cube_path.parents)[:6]:
        for name in ("LICENSE.md", "LICENSE"):
            path = parent / name
            if path.is_file():
                return path.read_text(encoding="utf-8")
    for parent in cube_path.parents:
        if parent.name == "site-packages":
            for path in parent.glob("turbo_moc-*.dist-info/licenses/LICENSE*"):
                if path.is_file():
                    return path.read_text(encoding="utf-8")
            break
    raise ViewerAssetsUnavailable(
        "The TurboMoC navigation license is missing. Set TURBOMOC_SOURCE to the complete companion checkout, including LICENSE.md."
    )


def inline_viewer_assets():
    """Return inline styles, import map, original view cube and viewer module.

    Insert after ``window.impellerReport`` and the viewer's DOM. The resulting
    HTML is portable and opens directly from disk, with no sibling files.
    """
    assets = _nurbspy_assets()
    cube_path, cube_source = _orientation_source()
    vendor = assets / "vendor"
    modules = {
        "impeller/three-core": "three.core.js",
        "impeller/three": "three.module.js",
        "impeller/trackball": "TrackballControls.js",
    }
    imports = {}
    for name, filename in modules.items():
        source = (vendor / filename).read_text(encoding="utf-8")
        for relative, mapped in (
            ("./three.core.js", "impeller/three-core"),
            ("./three.module.js", "impeller/three"),
        ):
            for quote in ("'", '"'):
                source = source.replace(
                    quote + relative + quote, quote + mapped + quote
                )
        imports[name] = "data:text/javascript;base64," + base64.b64encode(
            source.encode()
        ).decode("ascii")
    local = Path(__file__).with_name("assets")
    shared_css = (
        Path(__file__).parents[1] / "apps" / "shared" / "assets" / "vtk_display.css"
    )
    css = (
        shared_css.read_text(encoding="utf-8")
        + "\n"
        + (local / "impeller_viewer.css").read_text(encoding="utf-8")
    )
    # The original cube has no license header; preserve its repository license
    # in the exported artifact without copying its source into this package.
    licenses = (
        (vendor / "LICENSE").read_text(encoding="utf-8")
        + "\n\n"
        + _cube_license(cube_path)
    )
    provenance = {
        "three_version": "0.180.0",
        "three_source": "nurbspy/web_assets/vendor",
        "navigation_source": "turbo_moc/app/assets/orientation_cube.js",
        "navigation_sha256": hashlib.sha256(cube_path.read_bytes()).hexdigest(),
    }
    viewer = (local / "impeller_viewer.js").read_text(encoding="utf-8")
    return (
        "<style>"
        + css
        + "</style>\n"
        + '<script type="application/json" id="impeller-viewer-provenance">'
        + json.dumps(provenance)
        + "</script>\n"
        + '<script type="application/json" id="impeller-viewer-licenses">'
        + json.dumps(licenses).replace("<", "\\u003c")
        + "</script>\n"
        + '<script type="importmap">'
        + json.dumps({"imports": imports})
        + "</script>\n"
        + "<script>"
        + _script(cube_source)
        + "</script>\n"
        + '<script type="module">'
        + _script(viewer)
        + "</script>\n"
    )
