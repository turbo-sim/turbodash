"""Persistence, display-unit, and invalid-draft contracts for the local explorer."""

from copy import deepcopy
import base64

import numpy as np
import pytest
import yaml

from development.mixed_flow.explorer import model


@pytest.mark.parametrize("name", ["baseline", "inclined", "romei"])
def test_presets_roundtrip_yaml_and_generate_figures(name):
    config = model.preset_config(name)
    loaded = model.parse_yaml(model.dump_yaml(config))
    assert loaded == config
    result = model.evaluate_config(loaded)
    assert result["valid"], result.get("error")
    assert set(result["figures"]) == {"channel", "area", "curvature", "width"}
    assert len(result["metrics"]) == 4
    assert result["figures"]["channel"].layout.yaxis.scaleanchor == "x"


def test_display_millimetres_are_saved_in_metres_without_mutation():
    original = model.default_config()
    edited = model.set_display_value(original, "dimensions.outlet_width", 12.5)
    assert original["dimensions"]["outlet_width"] == 0.008
    assert edited["dimensions"]["outlet_width"] == 0.0125
    assert model.display_value(edited, "dimensions.outlet_width") == 12.5
    saved = yaml.safe_load(model.dump_yaml(edited))
    assert saved["dimensions"]["outlet_width"] == 0.0125


def test_linked_angles_follow_station_then_unlock_independently():
    config = model.default_config()
    assert model.is_linked(config, "shroud.inlet_angle_deg")
    changed = model.set_display_value(config, "dimensions.inlet_flow_angle_deg", 18)
    assert model.display_value(changed, "shroud.inlet_angle_deg") == 18
    changed = model.set_link(changed, "shroud.inlet_angle_deg", False)
    changed = model.set_display_value(changed, "dimensions.inlet_flow_angle_deg", 10)
    assert model.display_value(changed, "shroud.inlet_angle_deg") == 18
    assert model.display_value(changed, "hub.inlet_angle_deg") == 10
    changed = model.set_link(changed, "shroud.inlet_angle_deg", True)
    assert model.display_value(changed, "shroud.inlet_angle_deg") == 10
    assert (
        model.parse_yaml(model.dump_yaml(changed))["shroud"]["inlet_angle_deg"] is None
    )


@pytest.mark.parametrize(
    "path,value",
    [
        ("dimensions.outlet_width", -1),
        ("dimensions.inlet_hub_radius", 80),
        ("dimensions.inlet_shroud_radius", None),
        ("shroud.bow", -0.5),
        ("blade.leading_bow", -0.2),
        ("splitter_start", 1),
    ],
)
def test_invalid_draft_never_returns_successful_figures(path, value):
    config = model.set_display_value(model.default_config(), path, value)
    result = model.evaluate_config(config)
    assert result["valid"] is False
    assert result["error"]
    assert "figures" not in result and "metrics" not in result


@pytest.mark.parametrize(
    "text",
    [
        "[]",
        "dimensions: [1, 2]",
        "dimensions: {}",
        "dimensions: [",
        "!!python/object/apply:os.system ['echo forbidden']",
    ],
)
def test_invalid_yaml_is_rejected(text):
    with pytest.raises(ValueError):
        model.parse_yaml(text)


def test_unknown_fields_and_yaml_booleans_rejected():
    config = model.default_config()
    config["hub"]["control_points"] = [1, 2, 3]
    with pytest.raises(ValueError, match="Unknown"):
        model.normalize_config(config)
    config = model.default_config()
    config["dimensions"]["outlet_width"] = True
    with pytest.raises(ValueError, match="finite number"):
        model.normalize_config(config)


def test_omitted_length_uses_this_config_outlet_radius():
    config = model.default_config()
    config["dimensions"]["outlet_radius"] = 0.12
    del config["dimensions"]["axial_length"]
    assert model.normalize_config(config)["dimensions"][
        "axial_length"
    ] == pytest.approx(0.072)


def test_reference_does_not_change_geometry_or_metrics():
    config = model.preset_config("inclined")
    reference = model.preset_config("baseline")
    before = deepcopy(reference)
    plain = model.evaluate_config(config)
    overlay = model.evaluate_config(config, reference=reference)
    assert plain["valid"] and overlay["valid"]
    assert plain["config"] == overlay["config"]
    assert plain["metrics"] == overlay["metrics"]
    assert reference == before
    fig = overlay["figures"]["channel"]
    assert any(trace.name == "Reference" for trace in fig.data)
    plain_hub = next(t for t in plain["figures"]["channel"].data if t.name == "Hub")
    overlay_hub = next(t for t in fig.data if t.name == "Hub")
    np.testing.assert_array_equal(plain_hub.x, overlay_hub.x)
    np.testing.assert_array_equal(plain_hub.y, overlay_hub.y)


def test_app_serves_layout_without_starting_server():
    from development.mixed_flow.explorer.app import create_app

    app = create_app()
    client = app.server.test_client()
    for route in ("/", "/_dash-layout", "/_dash-dependencies", "/assets/style.css"):
        assert client.get(route).status_code == 200
    assert len(app.callback_map) == 4
    with pytest.raises(ValueError, match="dimensions"):
        create_app({})


def test_browser_upload_uses_same_validated_yaml_schema():
    from development.mixed_flow.explorer.app import uploaded_config

    config = model.preset_config("inclined")
    payload = base64.b64encode(model.dump_yaml(config).encode("utf-8")).decode("ascii")
    assert uploaded_config("data:application/yaml;base64," + payload) == config
    for invalid in ("", "not base64", "data:application/yaml;base64,@@@"):
        with pytest.raises(ValueError):
            uploaded_config(invalid)


def test_blank_explicit_wall_angle_remains_invalid_and_unlinked():
    from development.mixed_flow.explorer.app import (
        _control_view,
        ANGLE_PATHS,
        FIELD_PATHS,
    )

    config = model.set_link(model.default_config(), "shroud.inlet_angle_deg", False)
    config["shroud"]["inlet_angle_deg"] = ""  # app's explicit empty-input marker
    view = _control_view(config)
    assert view[0][FIELD_PATHS.index("shroud.inlet_angle_deg")] == ""
    assert view[-1][ANGLE_PATHS.index("shroud.inlet_angle_deg")] == []
    assert model.evaluate_config(config)["valid"] is False


def test_invalid_generation_callback_retains_figures_and_disables_save():
    from dash import no_update
    from development.mixed_flow.explorer.app import create_app

    app = create_app()
    callback = next(
        value["callback"].__wrapped__
        for key, value in app.callback_map.items()
        if "plot-channel.figure" in key
    )
    invalid = model.set_display_value(
        model.default_config(), "dimensions.outlet_width", -1
    )
    result = callback(invalid, model.default_config(), ["show"])
    assert all(item is no_update for item in result[:4])
    assert result[5] == {"valid": False}
    assert "Last valid geometry" in result[6]
    assert result[-2:] == (True, True)
