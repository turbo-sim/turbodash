"""Indexed preview data and persistent-frame routing."""

from pathlib import Path

from dash import Dash
import numpy as np

from turbodash.apps.app_impeller import viewer
from turbodash.geometry.impeller import build_impeller, load_config
from turbodash.geometry.impeller.output import SurfaceMesh


def test_live_payload_contains_one_closed_blade_and_disk():
    config = load_config(Path(__file__).parents[1] / "development/impeller/examples/reversible_impeller_zero_rake.yaml")
    config["numerics"].update(streamwise_points=41, spanwise_points=3, integration_points=201)
    geometry = build_impeller(config)
    data = viewer.payload(geometry)
    assert data["blade_count"] == geometry.blade_count
    assert data["fingerprint"] == geometry.fingerprint
    for name in ("blade", "disk"):
        mesh = SurfaceMesh(np.asarray(data[name]["vertices"]), np.asarray(data[name]["faces"]))
        mesh.validate(closed=True, positive_volume=True)
    np.testing.assert_array_equal(data["edges"]["leading"], geometry.camber[:, 0])
    np.testing.assert_array_equal(data["edges"]["trailing"], geometry.camber[:, -1])
    disk = data["disk"]
    # Axis vertices are shared fan poles, and explicit regions preserve a
    # smooth nose while keeping the back/rim corners sharp in the renderer.
    assert len(disk["face_regions"]) == len(disk["faces"])
    vertices = np.asarray(disk["vertices"])
    assert np.count_nonzero(np.linalg.norm(vertices[:, :2], axis=1) == 0) == 2
    assert vertices[:, 2].min() < geometry.plus[0, :, 2].min() - .03


def test_frame_route_supports_app_prefix_without_embedding_geometry(monkeypatch):
    monkeypatch.setattr(viewer, "inline_viewer_assets", lambda: "<!-- shared renderer -->")
    app = Dash(__name__, url_base_pathname="/geometry/")
    viewer.register(app)
    app.layout = viewer.viewer_layout(app)
    frame = app.layout.children[1]
    assert frame.src == "/geometry/_impeller_preview"
    response = app.server.test_client().get(frame.src)
    assert response.status_code == 200
    assert "shared renderer" in response.text
    assert "window.impellerReport=null" in response.text
    assert '"vertices"' not in response.text
    assert app.server.test_client().get("/_impeller_preview").status_code == 404
