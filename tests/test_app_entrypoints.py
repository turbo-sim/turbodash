"""Keep the established meanline app separate from the experimental workspace."""

import importlib
from pathlib import Path
import subprocess
import sys
import textwrap
import threading
from unittest.mock import Mock
import webbrowser

import pytest

import turbodash


def _components(component):
    """Walk the component JSON returned to the browser."""
    if isinstance(component, list):
        for child in component:
            yield from _components(child)
    elif isinstance(component, dict) and "props" in component:
        yield component
        yield from _components(component["props"].get("children"))


@pytest.mark.parametrize(
    "module_name,labels,experimental",
    [
        ("apps.turbine", ["Turbodash", "Documentation"], False),
        ("apps.turbine_experimental", ["Meanline", "Turbine geometry", "Mesh"], True),
    ],
)
def test_each_app_serves_its_own_workspace(module_name, labels, experimental):
    module = importlib.import_module(f"turbodash.{module_name}")
    client = module.server.test_client()
    assert client.get("/").status_code == 200
    layout = client.get("/_dash-layout")
    dependencies = client.get("/_dash-dependencies")
    assert layout.status_code == dependencies.status_code == 200

    components = list(_components(layout.get_json()))
    assert [item["props"]["label"] for item in components if item["type"] == "Tab"] == labels
    ids = {item["props"]["id"] for item in components
           if isinstance(item["props"].get("id"), str)}
    assert {"result_store", "stage_data_store", "turbogrid_export_button"} <= ids
    experimental_ids = {"geometry_pane", "mesh_pane", "result_input_signature", "mesh_job"}
    if experimental:
        assert experimental_ids <= ids
    else:
        assert not experimental_ids & ids

    outputs = " ".join(callback["output"] for callback in dependencies.get_json())
    assert ("geometry_pane.style" in outputs) is experimental


def test_apps_have_independent_servers_and_callbacks():
    from turbodash.apps import turbine as app_turbine, turbine_experimental as app_experimental

    assert app_turbine.app is not app_experimental.app
    assert app_turbine.server is not app_experimental.server
    assert app_turbine.app.callback_map is not app_experimental.app.callback_map


@pytest.mark.parametrize("module_name,launcher", [
    ("apps.turbine", "launch_app_turbine"),
    ("apps.turbine_experimental", "launch_app_experimental"),
])
@pytest.mark.parametrize("reloader_child", [False, True])
def test_launchers_run_the_selected_app_and_open_one_browser(
    monkeypatch, module_name, launcher, reloader_child,
):
    module = importlib.import_module(f"turbodash.{module_name}")
    run = Mock()
    open_browser = Mock()
    timers = []

    def timer(delay, callback):
        scheduled = Mock()
        scheduled.start.side_effect = callback
        timers.append((delay, scheduled))
        return scheduled

    monkeypatch.setattr(module.app, "run", run)
    monkeypatch.setattr(webbrowser, "open", open_browser)
    monkeypatch.setattr(threading, "Timer", timer)
    monkeypatch.setenv("WERKZEUG_RUN_MAIN", "true" if reloader_child else "false")

    getattr(turbodash, launcher)()

    run.assert_called_once_with(debug=True)
    if reloader_child:
        assert len(timers) == 1
        assert timers[0][0] == 1
        timers[0][1].start.assert_called_once_with()
        open_browser.assert_called_once_with("http://127.0.0.1:8050/")
    else:
        assert not timers
        open_browser.assert_not_called()


def test_stable_app_starts_without_importing_experimental_dependencies():
    code = textwrap.dedent("""
        import sys

        blocked = (
            "turbodash.apps.turbine_experimental", "turbodash.apps.components.meshing",
            "turbodash.meshing", "turbodash.apps.mesh_jobs",
            "turbodash.visualization.plotly_mesh", "turbodash.visualization.vtk",
            "turbodash.visualization.vtk_assets", "turbo_moc", "dash_vtk",
        )

        class BlockExperimentalImports:
            def find_spec(self, fullname, path=None, target=None):
                if any(fullname == name or fullname.startswith(name + ".") for name in blocked):
                    raise ImportError("Stable app imported " + fullname)

        sys.meta_path.insert(0, BlockExperimentalImports())
        from turbodash.apps.turbine import server
        assert server.test_client().get("/_dash-layout").status_code == 200
    """)
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
