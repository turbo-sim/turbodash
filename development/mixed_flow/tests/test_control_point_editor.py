"""The native editor accepts only reproducible, valid meridional wall changes."""

from copy import deepcopy
import base64
import json

import numpy as np
from numpy.testing import assert_allclose
import pytest

from development.mixed_flow.explorer import model
from development.mixed_flow.explorer import control_point_editor as editor
from development.mixed_flow.meridional import wall_scale


@pytest.fixture(scope="module")
def baseline():
    config = model.default_config()
    return config, editor.make_scene(config)


@pytest.fixture(scope="module")
def app():
    from development.mixed_flow.explorer.app import create_app

    return create_app()


def parameters(state, name="hub"):
    return state["objects"][name]["geometry"]["parameters"]


def points(state, name="hub"):
    return parameters(state, name)["control_points"]["values"]


def packet(scene, name="hub"):
    native = scene["scenes"][name]["model"]
    return {
        "revision": 1,
        "action": "edit",
        "quality": "full",
        "control_points": deepcopy(native["control_points"]),
        "weights": deepcopy(native["weights"]),
    }


def browser_json(value):
    """Emulate JSON.stringify's number spelling and string object keys."""
    if isinstance(value, dict):
        return {str(key): browser_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [browser_json(item) for item in value]
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def test_identity_survives_browser_number_and_displacement_key_roundtrips():
    config = model.default_config()
    config["control_point_displacements"] = {"hub": {3: [0.002, 0.0]}}
    encoded = json.loads(json.dumps(browser_json(config)))
    assert isinstance(encoded["blade"]["leading_hub"], int)
    assert "3" in encoded["control_point_displacements"]["hub"]
    assert editor.identity(encoded) == editor.identity(config)


def test_identity_treats_signed_zero_as_the_same_browser_value():
    config = model.default_config()
    config["dimensions"]["inlet_flow_angle_deg"] = -0.0
    assert editor.identity(browser_json(config)) == editor.identity(config)


def test_native_scene_preserves_exact_baseline_and_noop_apply(baseline):
    config, scene = baseline
    assert scene["host_config"] == config
    assert scene["identity"] == editor.identity(config)
    assert set(scene["state"]["objects"]) == {"hub", "shroud"}
    assert editor.config_from_state(config, scene["state"]) == config
    result = editor.evaluate_edit(config, scene["state"], "hub", packet(scene))
    assert result["quality"] == "full"
    assert result["revision"] == 1
    assert result["vertices"]


def test_each_wall_saves_total_displacements_and_yaml_reproduces_geometry(baseline):
    config, scene = baseline
    state = deepcopy(scene["state"])
    points(state, "hub")[0][3] += 1.25
    points(state, "hub")[1][3] -= 0.75
    points(state, "shroud")[0][4] -= 0.6
    updated = editor.config_from_state(config, state)
    _, original, _, _ = model.build_geometry(config)
    offsets = updated["control_point_displacements"]
    assert set(offsets) == {"hub", "shroud"}
    assert set(offsets["hub"]) == {3}
    assert set(offsets["shroud"]) == {4}
    assert_allclose(
        offsets["hub"][3], np.array([1.25, -0.75]) / (1000 * wall_scale(original.hub))
    )
    assert_allclose(
        offsets["shroud"][4], np.array([-0.6, 0]) / (1000 * wall_scale(original.shroud))
    )
    restored = model.parse_yaml(model.dump_yaml(updated))
    assert restored == updated
    _, regenerated, _, _ = model.build_geometry(restored)
    assert_allclose(
        regenerated.hub.as_bspline().c.T * 1000, points(state, "hub"), atol=1e-14
    )
    assert_allclose(
        regenerated.shroud.as_bspline().c.T * 1000, points(state, "shroud"), atol=1e-14
    )


def test_noop_reapply_preserves_saved_vectors_without_roundoff_drift():
    config = model.default_config()
    saved = {
        "hub": {2: [0.00382719847493821, -0.00182734618276349]},
        "shroud": {4: [-0.002718281828459, 0.001414213562373]},
    }
    config["control_point_displacements"] = deepcopy(saved)
    for _ in range(3):
        scene = editor.make_scene(config)
        config = editor.config_from_state(config, scene["state"])
        assert config["control_point_displacements"] == saved


def test_editing_one_point_preserves_other_saved_vectors():
    config = model.default_config()
    previous = [0.00382719847493821, -0.00182734618276349]
    config["control_point_displacements"] = {"hub": {2: previous}}
    state = editor.make_scene(config)["state"]
    points(state)[1][3] += 0.25
    updated = editor.config_from_state(config, state)
    assert updated["control_point_displacements"]["hub"][2] == previous
    assert set(updated["control_point_displacements"]["hub"]) == {2, 3}


@pytest.mark.parametrize("index", [0, 1, 5, 6])
def test_station_positions_and_tangents_are_protected(baseline, index):
    config, scene = baseline
    state = deepcopy(scene["state"])
    points(state)[0][index] += 0.001
    with pytest.raises(ValueError, match="fixed"):
        editor.config_from_state(config, state)
    attempted = packet(scene)
    attempted["control_points"][0][index] += 0.001
    with pytest.raises(ValueError, match="fixed"):
        editor.evaluate_edit(config, scene["state"], "hub", attempted)


@pytest.mark.parametrize("change", ["weights", "knots", "plane", "objects"])
def test_non_wall_edits_are_rejected(baseline, change):
    config, scene = baseline
    state = deepcopy(scene["state"])
    if change == "weights":
        parameters(state)["weights"]["values"][3] = 1.1
    elif change == "knots":
        parameters(state)["knots"]["values"][7:] = [2.0] * 7
    elif change == "plane":
        points(state).append([0.0] * 7)
        points(state)[2][3] = 0.1
    else:
        state["objects"]["other"] = deepcopy(state["objects"]["shroud"])
    with pytest.raises(ValueError):
        editor.config_from_state(config, state)


def test_crossed_channel_is_rejected_before_native_acceptance(baseline):
    config, scene = baseline
    state = deepcopy(scene["state"])
    points(state)[1][3] += 200
    with pytest.raises(ValueError, match="folds|radius|tangent"):
        editor.config_from_state(config, state)
    attempted = packet(scene)
    attempted["control_points"][1][3] += 200
    with pytest.raises(ValueError, match="folds|radius|tangent"):
        editor.evaluate_edit(config, scene["state"], "hub", attempted)


def test_transport_serves_native_graphics_and_rejects_invalid_edits(app, baseline):
    config, scene = baseline
    client = app.server.test_client()
    for filename in ("figure.js", "explorer.js", "vendor/three.module.js"):
        assert client.get("/_mixed_flow_walls/assets/" + filename).status_code == 200
    request = {
        "config": config,
        "state": scene["state"],
        "name": "hub",
        "packet": packet(scene),
    }
    response = client.post("/_mixed_flow_walls/evaluate", json=request)
    assert response.status_code == 200
    assert response.get_json()["vertices"]
    assert response.headers["Cache-Control"] == "no-store"
    request["packet"]["control_points"][0][0] += 1
    response = client.post("/_mixed_flow_walls/evaluate", json=request)
    assert response.status_code == 422
    assert "fixed" in response.get_json()["error"]


def test_native_inspection_uses_current_dragged_coordinates(app, baseline):
    config, scene = baseline
    native = scene["scenes"]["hub"]["model"]
    request = {
        "config": config,
        "name": "hub",
        "packet": {
            "revision": 7,
            "quality": "full",
            "field": "curvature",
            "control_points": deepcopy(native["control_points"]),
            "weights": native["weights"],
        },
    }
    client = app.server.test_client()
    before = client.post("/_mixed_flow_walls/inspect", json=request)
    assert before.status_code == 200
    request["packet"]["control_points"][1][3] += 1.5
    response = client.post("/_mixed_flow_walls/inspect", json=request)
    assert response.status_code == 200
    result = response.get_json()
    assert result["revision"] == 7
    assert result["quality"] == "full"
    assert result["field"]["label"] == "Curvature"
    assert result["field"]["unit"] == "1/length"
    assert not np.allclose(
        result["field"]["values"], before.get_json()["field"]["values"]
    )
    curve_type, _, editor_type, _ = editor.nurbspy_api()
    curve = curve_type(
        control_points=np.asarray(native["control_points"]),
        weights=np.asarray(native["weights"]),
        degree=native["degree"],
        knots=np.asarray(native["knots"]),
    )
    expected = editor_type(
        curve, id="test-inspection", quality="fine", preview_quality="full"
    ).inspect(request["packet"])
    assert_allclose(result["field"]["values"], expected["field"]["values"])


@pytest.mark.parametrize("quality", ["fast", "auto", "fine"])
def test_native_quality_tessellation_retains_current_curve(app, baseline, quality):
    config, scene = baseline
    native = scene["scenes"]["shroud"]["model"]
    request = {
        "config": config,
        "name": "shroud",
        "packet": {
            "revision": 11,
            "quality": "full",
            "tessellate": True,
            "draw_quality": quality,
            "control_points": deepcopy(native["control_points"]),
            "weights": native["weights"],
        },
    }
    request["packet"]["control_points"][0][3] += 0.9
    response = app.server.test_client().post("/_mixed_flow_walls/inspect", json=request)
    assert response.status_code == 200
    result = response.get_json()
    assert result["revision"] == 11
    assert result["quality"] == "full"
    assert result["draw_quality"] == quality
    assert result["sampling"]
    assert result["topologies"]
    positions = np.frombuffer(
        base64.b64decode(result["vertices"]), dtype="<f4"
    ).reshape(-1, 3)
    assert np.isfinite(positions).all()
    assert len(positions) >= 8
    assert np.all(positions[:, 2] == 0)


def post_apply(app, draft, commit, trigger="wall-commit.data"):
    key, callback = next(
        (key, callback)
        for key, callback in app.callback_map.items()
        if "wall-notice.children" in key
    )
    outputs = [
        {"id": output.component_id, "property": output.component_property}
        for output in callback["output"]
    ]
    response = app.server.test_client().post(
        "/_dash-update-component",
        json={
            "output": key,
            "outputs": outputs,
            "inputs": [
                {**spec, "value": value}
                for spec, value in zip(callback["inputs"], [commit, 1], strict=True)
            ],
            "state": [{**callback["state"][0], "value": draft}],
            "changedPropIds": [trigger],
        },
    )
    assert response.status_code == 200, response.get_data(as_text=True)
    return response.get_json()["response"]


def test_stale_commit_keeps_current_configuration(app, baseline):
    original, scene = baseline
    changed = model.set_display_value(original, "dimensions.axial_length", 65)
    commit = {"identity": scene["identity"], "state": scene["state"]}
    result = post_apply(app, changed, commit)
    assert "draft-config" not in result
    assert "Engineering inputs changed" in result["wall-notice"]["children"]


def test_browser_number_roundtrip_does_not_reject_a_current_commit(app, baseline):
    original, scene = baseline
    commit = {"identity": scene["identity"], "state": scene["state"]}
    result = post_apply(app, browser_json(original), commit)
    assert result["draft-config"]["data"] == original
    assert "applied" in result["wall-notice"]["children"]


def test_apply_and_clear_offsets_share_the_configuration_flow(app, baseline):
    original, scene = baseline
    state = deepcopy(scene["state"])
    points(state)[0][3] += 1
    commit = {"identity": scene["identity"], "state": state}
    response = post_apply(app, original, commit)
    updated = response["draft-config"]["data"]
    assert updated["control_point_displacements"]["hub"]["3"]
    assert "applied" in response["wall-notice"]["children"]
    response = post_apply(app, updated, commit, trigger="wall-clear.n_clicks")
    assert response["draft-config"]["data"] == original
