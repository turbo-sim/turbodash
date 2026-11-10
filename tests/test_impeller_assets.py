"""Offline module closure and integrity of reused browser assets."""

import base64
import hashlib
from html.parser import HTMLParser
import json
import re

import pytest

from turbodash.visualization import impeller_assets


class Scripts(HTMLParser):
    def __init__(self):
        super().__init__()
        self.scripts = []
        self.current = None

    def handle_starttag(self, tag, attrs):
        if tag == "script":
            self.current = {"attributes": dict(attrs), "source": ""}
            self.scripts.append(self.current)

    def handle_data(self, data):
        if self.current is not None:
            self.current["source"] += data

    def handle_endtag(self, tag):
        if tag == "script":
            self.current = None


@pytest.fixture
def upstream(tmp_path, monkeypatch):
    assets = tmp_path / "web_assets"
    vendor = assets / "vendor"
    vendor.mkdir(parents=True)
    sources = {
        "three.core.js": "export const revision = 180;",
        "three.module.js": "export {revision} from './three.core.js';",
        "TrackballControls.js": "import {revision} from './three.module.js'; export class TrackballControls {}",
        "LICENSE": "Three.js upstream test license",
    }
    for name, source in sources.items():
        (vendor / name).write_text(source, encoding="utf-8")
    manifest = {
        "version": "0.180.0",
        "sha256": {
            name: hashlib.sha256((vendor / name).read_bytes()).hexdigest()
            for name in sources
        },
    }
    (vendor / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setenv("NURBSPY_WEB_ASSETS", str(assets))
    return assets


def test_export_module_graph_is_closed_and_preserves_original_navigation(
    upstream, tmp_path, monkeypatch
):
    cube_path = tmp_path / "orientation_cube.js"
    cube_source = "window.turbomocOrientation = {original: true};"
    cube_path.write_text(cube_source, encoding="utf-8")
    (tmp_path / "LICENSE.md").write_text(
        "Original navigation license", encoding="utf-8"
    )
    monkeypatch.setattr(
        impeller_assets, "_orientation_source", lambda: (cube_path, cube_source)
    )
    parser = Scripts()
    parser.feed(impeller_assets.inline_viewer_assets())
    assert all("src" not in script["attributes"] for script in parser.scripts)
    importmap = next(
        script
        for script in parser.scripts
        if script["attributes"].get("type") == "importmap"
    )
    modules = json.loads(importmap["source"])["imports"]
    sources = []
    for url in modules.values():
        prefix, encoded = url.split(",", 1)
        assert prefix == "data:text/javascript;base64"
        sources.append(base64.b64decode(encoded).decode("utf-8"))
    sources.extend(
        script["source"]
        for script in parser.scripts
        if script["attributes"].get("type") == "module"
    )
    for source in sources:
        for specifier in re.findall(r"\bfrom\s+['\"]([^'\"]+)['\"]", source):
            assert specifier in modules, f"Unbundled module dependency: {specifier}"
    assert any(script["source"] == cube_source for script in parser.scripts)
    licenses = next(
        script
        for script in parser.scripts
        if script["attributes"].get("id") == "impeller-viewer-licenses"
    )
    assert "Three.js upstream test license" in json.loads(licenses["source"])
    assert "Original navigation license" in json.loads(licenses["source"])


def test_modified_upstream_assets_are_rejected_before_export(upstream):
    assert impeller_assets._nurbspy_assets() == upstream
    (upstream / "vendor" / "three.module.js").write_text("changed", encoding="utf-8")
    with pytest.raises(impeller_assets.ViewerAssetsUnavailable, match="does not match"):
        impeller_assets._nurbspy_assets()


def test_explicit_missing_assets_report_an_actionable_error(tmp_path, monkeypatch):
    monkeypatch.setenv("NURBSPY_WEB_ASSETS", str(tmp_path / "missing"))
    with pytest.raises(
        impeller_assets.ViewerAssetsUnavailable, match="NURBSPY_WEB_ASSETS"
    ):
        impeller_assets._nurbspy_assets()
