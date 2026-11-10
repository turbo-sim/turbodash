"""Native wall editing retains engineering constraints and exact saved offsets."""

from copy import deepcopy
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose
import pytest

from turbodash.apps.app_impeller import walls
from turbodash.geometry.impeller import build_impeller, load_config, normalize_config


@pytest.fixture(scope="module")
def baseline():
    config = load_config(
        Path(__file__).parents[1]
        / "development/impeller/examples/reversible_impeller_zero_rake.yaml"
    )
    return config, walls.make_scene(config)


def points(state, name="hub"):
    return state["objects"][name]["geometry"]["parameters"]["control_points"]["values"]


def test_noop_roundtrip_retains_arbitrary_impeller_controls(baseline):
    config, scene = baseline
    result = walls.apply_commit(
        config, {"identity": scene["identity"], "state": scene["state"]}
    )
    assert result == config
    assert result["rake"]["mode"] == "trailing_edge"
    assert result["sections"] == config["sections"]


def test_wall_offsets_reproduce_actual_float64_points_and_preserve_saved_vectors(
    baseline,
):
    config = deepcopy(baseline[0])
    original = [0.002718281828459, -0.001414213562373]
    config["control_point_displacements"] = {"hub": {"2": original}}
    config = normalize_config(config)
    scene = walls.make_scene(config)
    state = deepcopy(scene["state"])
    points(state)[0][3] += 1.25
    points(state, "shroud")[1][4] -= 0.75
    result = walls.config_from_state(config, state)
    assert result["control_point_displacements"]["hub"]["2"] == original
    assert result["sections"] == config["sections"]
    _, channel, _, _ = walls._ChannelModel.build_geometry(result)
    for wall in ("hub", "shroud"):
        assert_allclose(
            getattr(channel, wall).as_bspline().c.T * 1000,
            points(state, wall),
            atol=2e-13,
        )
    for _ in range(3):
        assert (
            walls.config_from_state(result, walls.make_scene(result)["state"]) == result
        )


@pytest.mark.parametrize("index", [0, 1, 5, 6])
def test_station_and_tangent_controls_cannot_be_dragged(baseline, index):
    config, scene = baseline
    state = deepcopy(scene["state"])
    points(state)[0][index] += 0.01
    with pytest.raises(ValueError, match="fixed"):
        walls.config_from_state(config, state)


@pytest.mark.parametrize("mutation", ["weights", "knots", "extra_object", "nonplanar"])
def test_non_wall_changes_are_rejected(baseline, mutation):
    config, scene = baseline
    state = deepcopy(scene["state"])
    parameters = state["objects"]["hub"]["geometry"]["parameters"]
    if mutation == "weights":
        parameters["weights"]["values"][3] = 1.1
    elif mutation == "knots":
        parameters["knots"]["values"][7:] = [2.0] * 7
    elif mutation == "nonplanar":
        points(state).append([0.0] * 7)
        points(state)[2][3] = 0.1
    else:
        state["objects"]["unrelated"] = deepcopy(state["objects"]["hub"])
    with pytest.raises(ValueError):
        walls.config_from_state(config, state)


def test_stale_commit_does_not_replace_newer_engineering_inputs(baseline):
    config, scene = baseline
    updated = deepcopy(config)
    updated["sections"][0]["camberline"]["turning_location"] += 0.05
    with pytest.raises(ValueError, match="Engineering inputs changed"):
        walls.apply_commit(
            updated, {"identity": scene["identity"], "state": scene["state"]}
        )
    assert updated["control_point_displacements"] == {}


def test_dragged_channel_enters_same_geometry_pipeline_and_clear_restores_it(baseline):
    config, scene = baseline
    state = deepcopy(scene["state"])
    points(state)[0][3] += 1.0
    result = walls.apply_commit(config, {"identity": scene["identity"], "state": state})
    assert result["control_point_displacements"]
    result["numerics"].update(
        streamwise_points=61, spanwise_points=5, integration_points=501
    )
    geometry = build_impeller(result)
    assert geometry.diagnostics["rake"]["max_abs_rake_error_deg"] < 1e-8
    assert np.isfinite(geometry.plus).all()
    restored = walls.apply_commit(result, clear=True)
    assert restored["control_point_displacements"] == {}
    assert restored["sections"] == config["sections"]


def test_channel_crossing_is_rejected_before_native_editor_acceptance(baseline):
    config, scene = baseline
    native = scene["scenes"]["hub"]["model"]
    packet = {
        "revision": 1,
        "action": "edit",
        "quality": "full",
        "control_points": deepcopy(native["control_points"]),
        "weights": native["weights"],
    }
    packet["control_points"][1][3] += 1000
    with pytest.raises(ValueError, match="folds|radius|tangent"):
        walls.evaluate_edit(config, scene["state"], "hub", packet)


def test_revision_refresh_discards_unapplied_edits_even_for_unchanged_config(baseline):
    from dash import Dash

    config, scene = baseline
    app = Dash(__name__)
    walls.register(app, config)
    callback = app.callback_map["wall-editor-scene.data"]
    assert callback["inputs"] == [{"id": "impeller-revision", "property": "data"}]
    assert all("draft-config.data" not in output for output in app.callback_map)
    edited = deepcopy(scene["state"])
    points(edited)[0][3] += 1.0
    refreshed = callback["callback"].__wrapped__(1, config, edited)
    assert refreshed["host_revision"] == 1
    assert_allclose(points(refreshed["state"]), points(scene["state"]), rtol=0, atol=0)
    # Out-of-order responses carry their own requested revision, rather than
    # whichever scene happened to be constructed last on the server.
    newer = callback["callback"].__wrapped__(3, config, edited)
    older = callback["callback"].__wrapped__(2, config, edited)
    assert (newer["host_revision"], older["host_revision"]) == (3, 2)
