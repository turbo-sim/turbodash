"""TurboMoC 1.2.0 supplies both renderer bundles through an allowlisted route."""

from dash import Dash, html
import pytest

from turbodash.visualization import vtk_assets


def test_meanline_app_needs_no_vtk_assets(monkeypatch):
    monkeypatch.setattr(vtk_assets, "available", lambda: False)
    app = Dash(__name__)
    app.layout = html.Div("Meanline")
    assert not vtk_assets.register_vtk_assets(app)
    assert not app.config.external_scripts
    assert app.server.test_client().get("/_dash-layout").status_code == 200


def test_serves_installed_renderer_unchanged_and_only_allowed_assets():
    pytest.importorskip("turbo_moc")
    pytest.importorskip("dash_vtk")
    app = Dash(__name__)
    app.layout = html.Div("Mesh")
    assert vtk_assets.register_vtk_assets(app)
    assert vtk_assets.register_vtk_assets(app)
    scripts = vtk_assets._scripts(vtk_assets._asset_directory())
    assert len(app.config.external_scripts) == len(scripts)
    client = app.server.test_client()
    for name in scripts:
        response = client.get(vtk_assets._PREFIX + name)
        assert response.status_code == 200
        assert response.get_data(as_text=True) == (vtk_assets._asset_directory() / name).read_text(encoding="utf-8")
    assert client.get(vtk_assets._PREFIX + "studio.css").status_code == 404
    assert client.get(vtk_assets._PREFIX + "application.py").status_code == 404
    assert client.get(vtk_assets._PREFIX + "../application.py").status_code == 404
    assert client.get(vtk_assets._PREFIX + "orientation_cube.js").status_code == (200 if vtk_assets._ORIENTATION in scripts else 404)
    assert client.get(vtk_assets._PREFIX + "orientation.css").status_code == 404
    assert not app.config.external_stylesheets
    index = client.get("/").get_data(as_text=True)
    assert "dash_vtk" in index
    assert vtk_assets._PREFIX + "studio.js" in index


def test_newer_optional_orientation_bundle_is_loaded_before_renderer_unchanged(tmp_path, monkeypatch):
    pytest.importorskip("dash_vtk")
    contents = {"mesh_edges.js": "window.turbomocMeshEdges = {};",
                "orientation_cube.js": "window.turbomocOrientation = {};",
                "studio.js": "window.turbomocOrientation.create();"}
    for name, source in contents.items():
        (tmp_path / name).write_text(source, encoding="utf-8")
    monkeypatch.setattr(vtk_assets, "_asset_directory", lambda: tmp_path)
    app = Dash(__name__)
    app.layout = html.Div("Preview")
    assert vtk_assets.register_vtk_assets(app)
    assert app.config.external_scripts == [vtk_assets._PREFIX + name for name in contents]
    client = app.server.test_client()
    for name, source in contents.items():
        assert client.get(vtk_assets._PREFIX + name).get_data(as_text=True) == source
    assert client.get(vtk_assets._PREFIX + "studio.css").status_code == 404


def test_incomplete_newer_bundle_explains_unavailable_viewer(tmp_path, monkeypatch):
    pytest.importorskip("dash_vtk")
    (tmp_path / "mesh_edges.js").write_text("", encoding="utf-8")
    (tmp_path / "studio.js").write_text("window.turbomocOrientation.create();", encoding="utf-8")
    monkeypatch.setattr(vtk_assets, "_asset_directory", lambda: tmp_path)
    assert not vtk_assets.available()


@pytest.mark.parametrize("installed", [True, False])
def test_optional_pandas_is_initialized_before_renderer_registration(monkeypatch, installed):
    pytest.importorskip("turbo_moc")
    pytest.importorskip("dash_vtk")
    app = Dash(__name__)
    app.layout = html.Div("Preview")
    find_spec = vtk_assets.util.find_spec
    monkeypatch.setattr(vtk_assets.util, "find_spec",
                        lambda name: object() if name == "pandas" and installed
                        else None if name == "pandas" else find_spec(name))
    imports = []

    def initialize(name):
        assert not app.config.external_scripts
        assert "turbodash_vtk_asset" not in app.server.view_functions
        imports.append(name)

    monkeypatch.setattr(vtk_assets, "import_module", initialize)
    assert vtk_assets.register_vtk_assets(app)
    assert imports == (["pandas"] if installed else [])
    assert app.config.external_scripts
