"""Relocation parity and working YAML-driven applications through Dash HTTP."""
import contextlib
import importlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from turbodash.apps.shared.startup import components
from turbodash.core import stage, turbine
from turbodash.reporting import stage as reports

ROOT = Path(__file__).resolve().parents[1]
STAGE_REFERENCE = json.loads((ROOT / "tests/fixtures/stage_refactor_reference.json").read_text())
APPLICATIONS = ("app_turbine", "app_turbine_experimental", "app_stage", "app_stage_efficiency")


def compare(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key, value in expected.items():
            compare(actual[key], value)
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            compare(left, right)
    elif isinstance(expected, (float, int)):
        assert actual == pytest.approx(expected, rel=1e-12, abs=1e-12, nan_ok=True)
    else:
        assert actual == expected


@pytest.mark.parametrize("kind", ["axial", "radial"])
def test_original_stage_and_reports_match_pre_refactor_capture(kind):
    reference = STAGE_REFERENCE["cases"][kind]
    actual = stage.compute_stage_meanline(**reference["inputs"])
    compare(actual, reference["result"])
    compare([reports.stage_performance_table(actual), reports.geometry_table(actual),
             reports.flow_stations_table(actual)], reference["tables"])
    text = io.StringIO()
    with contextlib.redirect_stdout(text):
        reports.print_stage(actual)
    assert text.getvalue() == reference["console"]


def test_excel_adapter_retains_package_root_lookup_and_workbook_name():
    from turbodash.io import excel_turbine
    assert Path(excel_turbine._CONFIG_DIR) == ROOT / "turbodash"
    assert callable(excel_turbine.meanline_caculator_full)


def dispatch_calculation(module):
    """Use the endpoint payload produced by Dash for fixed and pattern inputs."""
    controls = list(components(module.app.layout))
    fixed = {c.id: c for c in controls if isinstance(getattr(c, "id", None), str)}
    if module.__name__.endswith(".app_stage_efficiency.app"):
        target = "efficiency_ts_plot"
    elif module.__name__.endswith(".app_stage.app"):
        target = "stage_result_store"
    else:
        target = "result_store"
    key, callback = next((k, value) for k, value in module.app.callback_map.items() if target in k)

    def value(item):
        identity, prop = item["id"], item["property"]
        if identity.startswith("{"):
            pattern = json.loads(identity)
            selected = []
            for control in controls:
                candidate = getattr(control, "id", None)
                if isinstance(candidate, dict) and all(
                    isinstance(expected, list) or candidate.get(k) == expected
                    for k, expected in pattern.items()
                ):
                    selected.append(candidate if prop == "id" else getattr(control, prop))
            return selected
        return getattr(fixed[identity], prop)

    outputs = callback["output"]
    if not isinstance(outputs, list):
        outputs = [outputs]
    response = module.server.test_client().post("/_dash-update-component", json={
        "output": key,
        "outputs": [{"id": out.component_id, "property": out.component_property} for out in outputs],
        "inputs": [{**item, "value": value(item)} for item in callback["inputs"]],
        "state": [{**item, "value": value(item)} for item in callback["state"]],
        "changedPropIds": [],
    })
    assert response.status_code == 200, response.get_data(as_text=True)
    return response.get_json()["response"]


@pytest.mark.parametrize("name", APPLICATIONS)
def test_four_yaml_cases_serve_and_calculate_via_dash(name, monkeypatch):
    module = importlib.import_module("turbodash.apps." + name + ".app")
    original = module.app.layout
    from copy import deepcopy
    monkeypatch.setattr(module.app, "layout", deepcopy(original))
    case = module.configure(ROOT / "demos/apps" / name.removeprefix("app_") / "case.yaml")
    client = module.server.test_client()
    for route in ("/", "/_dash-layout", "/_dash-dependencies"):
        assert client.get(route).status_code == 200
    assert Path(module.app.config.assets_folder) == ROOT / "turbodash/apps/shared/assets"
    response = dispatch_calculation(module)
    if name == "app_stage_efficiency":
        for identity in ("efficiency_ts_plot", "efficiency_tt_plot"):
            assert len(response[identity]["figure"]["data"]) == len(case["inputs"]["R_values"])
    elif name == "app_stage":
        result = response["stage_result_store"]["data"]
        assert np.isfinite(result["stage_performance"]["efficiency_ts"])
        assert result["inputs"]["degree_reaction"] == case["inputs"]["degree_reaction"]
    else:
        result = response["result_store"]["data"]
        assert result["inputs"] == case["inputs"]
        expected = turbine.compute_turbine_performance(case)
        compare(result["overall_performance"], expected["overall_performance"])
        if name == "app_turbine_experimental":
            from turbodash.geometry.meanline import initialize_geometry
            render = next(entry["callback"].__wrapped__ for entry in module.app.callback_map.values()
                          if entry.get("callback") and entry["callback"].__wrapped__.__name__ == "meanline_blades")
            assert render(initialize_geometry(result), {"display": "block"}, result).data
        else:
            assert response["blades_plot"]["figure"]["data"]


@pytest.mark.parametrize("name", APPLICATIONS)
def test_demo_runner_passes_its_adjacent_yaml_to_the_app(name, monkeypatch):
    import runpy
    module = importlib.import_module("turbodash.apps." + name + ".app")
    observed = []
    monkeypatch.setattr(module, "main", lambda path: observed.append(path))
    runner = ROOT / "demos/apps" / name.removeprefix("app_") / "run.py"
    runpy.run_path(str(runner), run_name="__main__")
    assert observed == [runner.with_name("case.yaml")]


def test_worker_publishes_failure_from_outside_checkout(tmp_path):
    import time
    request = tmp_path / "request.json"
    request.write_text(json.dumps(dict(results={}, stage_index=0, row="stator", options={}, created=time.time())))
    # Select the tested checkout even when an older release is installed.
    environment = {**os.environ, "PYTHONPATH": str(ROOT)}
    completed = subprocess.run([sys.executable, "-B", "-m", "turbodash.apps.shared.mesh_jobs", str(request)],
                               cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=30)
    assert completed.returncode == 0, completed.stderr
    assert json.loads((tmp_path / "status.json").read_text())["state"] == "failed"
