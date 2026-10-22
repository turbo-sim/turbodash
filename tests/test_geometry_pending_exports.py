"""Downstream actions cannot use a projection behind the browser edit journal."""
from copy import deepcopy
from types import SimpleNamespace

from dash import Dash, no_update
import pytest

from turbodash.apps.shared.components import meshing, turbogrid
from turbodash.apps.shared.geometry_workflow import pending_geometry_reason


@pytest.fixture
def guarded(monkeypatch):
    app = Dash(__name__, suppress_callback_exceptions=True)
    meshing.register_meshing_callbacks(app, results_store="geometry_results_store", modeller=True)
    turbogrid.register_turbogrid_callbacks(app, results_store="geometry_results_store", modeller=True)
    monkeypatch.setattr(meshing.meshing, "dependency_status", lambda: (True, ""))
    monkeypatch.setattr(meshing, "vtk_assets_available", lambda: True)
    callbacks = {entry["callback"].__wrapped__.__name__: entry["callback"].__wrapped__
                 for entry in app.callback_map.values() if "callback" in entry}
    return app, callbacks


@pytest.fixture
def state():
    results = {"inputs": {"turbine_type": "axial"}, "stages_performance": [{}],
               "_geometry_projection": {"epoch": "current", "revision": 5}}
    live = ([], [], [], [], [{}])
    return SimpleNamespace(results=results, live=live, signature=meshing.input_signature(*live),
                           event={"epoch": "current", "sequence": 6},
                           options=[item[2] for item in meshing._SETTINGS] + [[]])


@pytest.mark.parametrize("event,blocked", [
    ({"epoch": "current", "sequence": 6}, True),
    ({"epoch": "current", "sequence": 5}, False),
    ({"epoch": "current", "sequence": 4}, False),
    ({"epoch": "replaced", "sequence": 900}, False),
    (None, False),
])
def test_projection_guard_compares_only_same_baseline_epoch(state, event, blocked):
    assert bool(pending_geometry_reason(state.results, event)) is blocked


def test_availability_disables_actions_until_projection_catches_up(guarded, state):
    _, callbacks = guarded
    assert callbacks["geometry_export_available"](state.results, state.signature, *state.live, state.event)
    status = callbacks["display_status"]({"state": "complete"}, state.results,
                                          state.signature, *state.live, state.event)
    assert status[0] and status[-1] and "Wait for geometry edits" in status[1]
    disabled, message, _ = callbacks["update_export_availability"](
        state.results, "axial", "independent", "", state.event)
    assert disabled and "Wait for geometry edits" in message
    current = deepcopy(state.results)
    current["_geometry_projection"]["revision"] = 6
    assert not callbacks["geometry_export_available"](current, state.signature, *state.live, state.event)
    assert not callbacks["update_export_availability"](current, "axial", "independent", "", state.event)[0]
    ready = callbacks["display_status"]({"state": "idle"}, current, state.signature, *state.live, state.event)
    assert not ready[0] and ready[1] == ""


def test_pending_clicks_never_export_or_submit_a_mesh(guarded, state, monkeypatch):
    _, callbacks = guarded
    def forbidden(*args, **kwargs):
        pytest.fail("A pending edit reached an export or mesh submission.")
    monkeypatch.setattr(meshing.meshing, "geometry_zip", forbidden)
    monkeypatch.setattr(turbogrid, "export_turbogrid_zip", forbidden)
    monkeypatch.setattr(turbogrid, "export_turbogrid_assembly_zip", forbidden)
    monkeypatch.setattr(meshing.mesh_jobs, "start_job", forbidden)
    monkeypatch.setattr(meshing, "ctx", SimpleNamespace(triggered_id="mesh_generate"))
    data, message = callbacks["download_geometry"](
        1, state.results, state.signature, 0, "stator", *state.live, state.event)
    assert data is no_update and "Wait for geometry edits" in message
    for mode in ("independent", "assembly"):
        data, message, opened = callbacks["download_turbogrid"](
            1, state.results, "axial", mode, "1", state.event)
        assert data is no_update and opened and "Wait for geometry edits" in message
    handle, status, stopped = callbacks["manage_mesh"](
        1, 0, state.results, state.signature, 0, "stator",
        *state.options, *state.live, state.event, {"fingerprint": "old"})
    assert handle is None and stopped and "Wait for geometry edits" in status["message"]


def test_matching_old_mesh_cannot_download_during_pending_edit(guarded, state, monkeypatch):
    _, callbacks = guarded
    monkeypatch.setattr(meshing.mesh_jobs, "fingerprint", lambda *args: "still-matches-old-projection")
    monkeypatch.setattr(meshing.mesh_jobs, "mesh_archive", lambda *args: pytest.fail("Stale mesh downloaded."))
    data, message = callbacks["download_mesh"](
        1, {"fingerprint": "still-matches-old-projection"}, state.results,
        state.signature, 0, "stator", *state.options, *state.live, state.event)
    assert data is no_update and "Wait for geometry edits" in message
    # After projection catches up, the existing fingerprint guard still rejects
    # an artifact for the previous effective geometry.
    current = deepcopy(state.results)
    current["_geometry_projection"]["revision"] = 6
    monkeypatch.setattr(meshing.mesh_jobs, "fingerprint", lambda *args: "new-geometry")
    data, message = callbacks["download_mesh"](
        1, {"fingerprint": "still-matches-old-projection"}, current,
        state.signature, 0, "stator", *state.options, *state.live, state.event)
    assert data is no_update and "does not match" in message


def test_existing_signature_and_dependency_guards_remain_active(guarded, state, monkeypatch):
    _, callbacks = guarded
    _, message = callbacks["download_geometry"](
        1, state.results, "stale-inputs", 0, "stator", *state.live, state.event)
    assert "inputs have changed" in message
    monkeypatch.setattr(meshing.meshing, "dependency_status", lambda: (False, "Missing meshing package"))
    _, message = callbacks["download_geometry"](
        1, state.results, state.signature, 0, "stator", *state.live, None)
    assert message == "Missing meshing package"


def test_optional_registration_adds_journal_dependencies_without_affecting_stable_app(guarded):
    app, _ = guarded
    names = {"download_geometry", "download_turbogrid", "download_mesh", "manage_mesh",
             "display_status", "geometry_export_available", "update_export_availability"}
    for entry in app.callback_map.values():
        callback = entry.get("callback")
        if callback and callback.__wrapped__.__name__ in names:
            assert any(item["id"] == "blade_modeller_events" for item in entry["inputs"] + entry["state"])
    stable = Dash(__name__ + "_stable", suppress_callback_exceptions=True)
    meshing.register_meshing_callbacks(stable)
    turbogrid.register_turbogrid_callbacks(stable)
    assert not any(item["id"] == "blade_modeller_events" for entry in stable.callback_map.values()
                   for item in entry["inputs"] + entry["state"])


def test_modeller_mesh_controls_have_one_owner_for_domain_and_blade_sampling():
    def input_keys(nodes):
        found = set()
        for node in nodes:
            identity = getattr(node, "id", None)
            if isinstance(identity, dict) and identity.get("scope") == "mesh-row":
                found.add(identity["key"])
            children = getattr(node, "children", None)
            if children is not None:
                found.update(input_keys(children if isinstance(children, list) else [children]))
        return found
    shared_keys = {"inlet_chords", "outlet_chords", "profile_points"}
    assert not shared_keys & input_keys(meshing.mesh_row_panels([{}], modeller=True))
    assert shared_keys <= input_keys(meshing.mesh_row_panels([{}]))
