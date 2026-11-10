"""Workbench transactions retain coherent geometry and reproducible inputs."""

import base64
from copy import deepcopy

import pytest
import yaml

from turbodash.apps.app_impeller import model


@pytest.fixture(scope="module")
def config():
    config = model.default_config()
    config["numerics"].update(
        streamwise_points=61, spanwise_points=5, integration_points=501
    )
    return config


def test_yaml_roundtrip_preserves_full_precision_and_independent_sections(config):
    changed = deepcopy(config)
    changed["sections"][0]["thickness"]["maximum_thickness_m"] *= 1.1731
    text = yaml.safe_dump(changed)
    encoded = "data:application/yaml;base64," + base64.b64encode(text.encode()).decode()
    parsed = model.uploaded_config(encoded)
    assert parsed == changed
    saved = model.downloadable_yaml({"revision": 7, "config": parsed}, 7)
    assert yaml.safe_load(saved) == changed
    with pytest.raises(ValueError, match="valid current geometry"):
        model.downloadable_yaml({"revision": 7, "config": parsed}, 8)


def test_invalid_candidate_cannot_publish_geometry_or_change_previous_config(config):
    bad = deepcopy(config)
    bad["dimensions"]["inlet_hub_radius"] = bad["dimensions"]["inlet_shroud_radius"] * 2
    original = deepcopy(config)
    packet = model.evaluate(bad, 5)
    assert packet["revision"] == 5 and packet["valid"] is False
    assert packet["error"] and "viewer" not in packet and "figures" not in packet
    assert config == original


def test_changed_angle_is_used_by_both_plot_and_3d_packet(config):
    changed = deepcopy(config)
    changed["sections"][-1]["camberline"]["inlet_metal_angle_deg"] = -48.125
    packet = model.evaluate(changed, 11)
    assert packet["valid"]
    assert packet["figures"]["angles"].data[1].y[0] == pytest.approx(-48.125)
    assert packet["viewer"]["fingerprint"] == packet["fingerprint"]
    assert packet["config"] == changed
    assert packet["viewer"]["blade_count"] == changed["blade_count"]


@pytest.mark.parametrize(
    "contents",
    [
        "bad",
        "data:application/yaml;base64,!!!",
        "data:application/yaml;base64," + base64.b64encode(b"x: [").decode(),
    ],
)
def test_invalid_yaml_is_rejected_atomically(contents):
    with pytest.raises(ValueError):
        model.uploaded_config(contents)


def test_interactive_sampling_is_bounded_without_changing_kernel(config):
    changed = deepcopy(config)
    changed["numerics"]["spanwise_points"] = 1001
    with pytest.raises(ValueError, match="Interactive spanwise_points"):
        model.validate_config(changed)
