"""Production callback graph: geometry is downstream of accepted meanline only."""
from copy import deepcopy
from unittest.mock import patch

from dash.exceptions import PreventUpdate
import pytest

from turbodash.apps.app_turbine_optimization import app as application
from turbodash.geometry.meanline import initialize_geometry, update_geometry_override


def callback(name):
    return next(entry for entry in application.app.callback_map.values()
                if entry.get("callback") and entry["callback"].__wrapped__.__name__ == name)


def test_solver_has_no_geometry_or_view_inputs_and_editor_has_no_optimizer_trials():
    inputs = callback("update_turbine")["inputs"]
    assert {item["id"] for item in inputs if not item["id"].startswith("{")} == {"opt-applied-result"}
    editor = callback("edit")["inputs"]
    assert any(item["id"] == "result_store" for item in editor)
    assert not any(item["id"].startswith("opt-") for item in editor)
    for name in ("manage_mesh", "download_geometry", "download_turbogrid"):
        entry = callback(name)
        assert any(item["id"] == "geometry_results_store" for item in entry["inputs"] + entry["state"])
        assert not any(item["id"] == "result_store" for item in entry["inputs"] + entry["state"])


def test_hidden_meanline_does_no_geometry_work_and_opening_uses_latest_state():
    render = callback("meanline_blades")["callback"].__wrapped__
    from turbodash.apps.shared import geometry_workflow as workflow
    with patch.object(workflow, "effective_results", return_value={"current": True}) as project, \
         patch.object(workflow, "plot_turbine_blades", return_value="figure") as plot:
        with pytest.raises(PreventUpdate):
            render({"revision": 8}, {"display": "none"}, {"solved": True})
        project.assert_not_called()
        plot.assert_not_called()
        assert render({"revision": 9}, {"display": "block"}, {"solved": True}) == "figure"
        project.assert_called_once_with({"solved": True}, {"revision": 9})


def test_projection_preserves_meanline_and_rejects_stale_baselines():
    configuration = application._assemble_cfg(application.OVERALL_DEFAULTS, [application.STAGE_DEFAULTS])
    results = application.core.compute_turbine_performance(configuration)
    original = deepcopy(results)
    state = initialize_geometry(results)
    state = update_geometry_override(state, state["rows"][0]["id"], "camberline.turning_concentration", 1.5)
    project = callback("downstream_geometry")["callback"].__wrapped__
    with patch.object(application.core, "compute_turbine_performance", side_effect=AssertionError("Unexpected solve")):
        projected = project(state, results)
    assert projected["stages_performance"][0]["geometry_parameters"]["stator"]["overrides"] == {
        "camberline.turning_concentration": 1.5,
    }
    for stage in projected["stages_performance"]:
        stage.pop("geometry_parameters")
    assert projected.pop("_geometry_projection") == {"epoch": None, "revision": 1}
    assert projected == original == results
    state["baseline_id"] = "superseded"
    with pytest.raises(PreventUpdate):
        project(state, results)
