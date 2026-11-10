"""Portable disk controls preserve legacy identities and aerodynamic inputs."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
from numpy.testing import assert_array_equal
import pytest

from turbodash.geometry.impeller import (
    build_impeller,
    load_config,
    normalize_config,
    save_config,
)
from turbodash.geometry.impeller.config import config_fingerprint


@pytest.fixture
def config():
    return load_config(
        Path(__file__).parents[1]
        / "development/impeller/examples/reversible_impeller_zero_rake.yaml"
    )


def test_legacy_configuration_without_disk_retains_aerodynamic_identity(config):
    config.pop("disk")
    resolved = normalize_config(config)
    assert "disk" not in resolved
    assert (
        config_fingerprint(resolved)
        == "c7c54829660de75bc614cdbf17c32b391a7acbb19680ef89a27a40c42c1e4301"
    )


def test_disk_settings_round_trip_through_yaml_and_browser_json(config, tmp_path):
    expected = deepcopy(config)
    config["disk"]["bore_radius_m"] = 0  # Browser integer spelling of zero.
    normalized = normalize_config(json.loads(json.dumps(config)))
    destination = tmp_path / "with_nose.yaml"
    save_config(normalized, destination)
    loaded = load_config(destination)
    assert loaded["disk"] == expected["disk"]
    assert config_fingerprint(loaded) == config_fingerprint(expected)


def test_disk_defaults_are_explicit_only_when_group_is_supplied(config):
    config["disk"] = {}
    assert normalize_config(config)["disk"] == {
        "bore_radius_m": 0.0,
        "nose_length_m": 0.0,
        "back_thickness_m": None,
    }


@pytest.mark.parametrize(
    "name, value",
    [
        ("bore_radius_m", -0.001),
        ("nose_length_m", -0.001),
        ("back_thickness_m", 0),
        ("back_thickness_m", -0.001),
        ("nose_length_m", np.inf),
        ("bore_radius_m", np.nan),
        ("nose_length_m", True),
        ("bore_radius_m", False),
        ("back_thickness_m", "0.03"),
        ("nose_length_m", None),
    ],
)
def test_nonphysical_disk_inputs_are_rejected(config, name, value):
    config["disk"][name] = value
    with pytest.raises(ValueError, match="disk"):
        normalize_config(config)


def test_rounded_nose_rejects_a_conflicting_shaft_bore(config):
    config["disk"].update(nose_length_m=0.035, bore_radius_m=0.01)
    with pytest.raises(ValueError, match="rounded nose|solid centre"):
        normalize_config(config)


def test_unknown_disk_control_is_not_silently_ignored(config):
    config["disk"]["nose_length_mm"] = 35
    with pytest.raises(ValueError, match="Unknown disk"):
        normalize_config(config)


def test_nose_changes_assembly_identity_without_changing_blade_geometry(config):
    config["numerics"].update(
        streamwise_points=41, spanwise_points=3, integration_points=401
    )
    original = build_impeller(config)
    changed = deepcopy(config)
    changed["disk"].update(nose_length_m=0.05, back_thickness_m=0.035)
    modified = build_impeller(changed)
    assert original.fingerprint != modified.fingerprint
    for name in ("u", "v", "plus", "minus", "camber"):
        assert_array_equal(getattr(original, name), getattr(modified, name))
    for first, second in zip(original.sections, modified.sections):
        assert_array_equal(first.beta, second.beta)
        assert_array_equal(first.target_thickness_m, second.target_thickness_m)
