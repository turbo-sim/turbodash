"""Contour changes redraw accepted results independently of the meanline solve."""

from copy import deepcopy
import importlib
from pathlib import Path

from dash import dcc
import pytest
import yaml

from turbodash.apps.shared.startup import components


@pytest.fixture(params=["app_turbine", "app_turbine_experimental", "app_turbine_optimization"])
def application(request):
    return importlib.import_module(f"turbodash.apps.{request.param}.app")


def test_contour_selection_has_one_figure_writer_and_no_calculation_dependency(application):
    controls = list(components(application.app.layout))
    selector = next(c for c in controls
                    if getattr(c, "id", None) == "td-contour-degree")
    assert isinstance(selector, dcc.RadioItems)
    assert [option["value"] for option in selector.options] == [1, 3]
    assert [option["label"] for option in selector.options] == ["Linear", "Smooth"]
    assert selector.value == 1
    assert selector.persistence == "linear-smooth"
    assert selector.persistence_type == "session"
    setup_ids = [getattr(c, "id", None) for c in controls
                 if getattr(c, "id", None) in (
                     {"scope": "overall", "key": "turbine_type"},
                     "td-contour-degree", "stage_count_label")]
    assert setup_ids == [
        {"scope": "overall", "key": "turbine_type"},
        "td-contour-degree", "stage_count_label",
    ]
    writers = [key for key in application.app.callback_map if "meridional_plot.figure" in key]
    assert writers == ["meridional_plot.figure"]
    redraw = application.app.callback_map[writers[0]]
    assert redraw["inputs"] == [
        {"id": "result_store", "property": "data"},
        {"id": "td-contour-degree", "property": "value"},
    ]
    for key, callback in application.app.callback_map.items():
        if "result_store.data" in key:
            assert all(item["id"] != "td-contour-degree" for item in callback["inputs"])


def test_contour_callback_redraws_latest_result_without_solving(application, monkeypatch):
    def unexpected_solve(*args, **kwargs):
        pytest.fail("A display-only contour change invoked the turbine solver")

    monkeypatch.setattr(application.core, "compute_turbine_performance", unexpected_solve)
    path = Path(__file__).parents[1] / "demos/basic_demo/demo_axial_turbine.yaml"
    results = yaml.safe_load(path.read_text(encoding="utf-8"))
    original = deepcopy(results)
    client = application.server.test_client()
    assert client.get("/_dash-layout").status_code == 200

    def redraw(result, degree, changed="td-contour-degree.value"):
        response = client.post("/_dash-update-component", json={
            "output": "meridional_plot.figure",
            "outputs": {"id": "meridional_plot", "property": "figure"},
            "inputs": [
                {"id": "result_store", "property": "data", "value": result},
                {"id": "td-contour-degree", "property": "value", "value": degree},
            ],
            "state": [],
            "changedPropIds": [changed],
        })
        assert response.status_code == 200, response.get_data(as_text=True)
        data = response.get_json()["response"]["meridional_plot"]["figure"]["data"]
        return next(trace for trace in data if trace["name"] == "Hub")

    linear = redraw(results, 1)
    for degree in (2, 3):
        smooth = redraw(results, degree)
        assert len(smooth["x"]) > len(linear["x"])
        assert (smooth["x"][0], smooth["y"][0]) == pytest.approx((linear["x"][0], linear["y"][0]))
        assert (smooth["x"][-1], smooth["y"][-1]) == pytest.approx((linear["x"][-1], linear["y"][-1]))

    # A later solve publishes new stations, and the current selection still
    # governs its figure rather than a degree-1 figure from the solve callback.
    latest = deepcopy(results)
    for stage in latest["stages_performance"]:
        for station in stage["flow_stations"]:
            station["r"] += .01
    updated = redraw(latest, 3, changed="result_store.data")
    assert len(updated["x"]) > len(linear["x"])
    assert updated["y"][0] == pytest.approx(linear["y"][0] + 10)
    assert results == original
