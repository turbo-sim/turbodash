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
    assert len(app.config.external_scripts) == 2
    client = app.server.test_client()
    for name in vtk_assets._SCRIPTS:
        response = client.get(vtk_assets._PREFIX + name)
        assert response.status_code == 200
        assert response.get_data(as_text=True) == (vtk_assets._asset_directory() / name).read_text(encoding="utf-8")
    assert client.get(vtk_assets._PREFIX + "studio.css").status_code == 404
    assert client.get(vtk_assets._PREFIX + "application.py").status_code == 404
    assert client.get(vtk_assets._PREFIX + "../application.py").status_code == 404
    assert client.get(vtk_assets._PREFIX + "orientation_cube.js").status_code == 404
    assert client.get(vtk_assets._PREFIX + "orientation.css").status_code == 404
    assert not app.config.external_stylesheets
    index = client.get("/").get_data(as_text=True)
    assert "dash_vtk" in index
    assert vtk_assets._PREFIX + "studio.js" in index
