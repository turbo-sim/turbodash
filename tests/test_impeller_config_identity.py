"""Physically identical browser number spellings keep geometry identity."""

import json
from pathlib import Path

import pytest

from turbodash.geometry.impeller import load_config, normalize_config
from turbodash.geometry.impeller.config import config_fingerprint


def browser_roundtrip(value):
    if isinstance(value, dict):
        return {str(key): browser_roundtrip(item) for key, item in value.items()}
    if isinstance(value, list):
        return [browser_roundtrip(item) for item in value]
    if type(value) is float and value.is_integer():
        return int(value)
    return value


@pytest.fixture
def config():
    return load_config(
        Path(__file__).parents[1]
        / "development/impeller/examples/reversible_impeller_zero_rake.yaml"
    )


def test_browser_roundtrip_retains_recorded_default_geometry_fingerprint(config):
    assert (
        config_fingerprint(config)
        == "ff3f2291ad9de5e42d59ecdfedbd0a4aa8aafd0c02485a7eaab481f52bc35c5c"
    )
    # Adding the rounded solid disk deliberately changes assembly identity;
    # historical files without disk controls retain the aerodynamic identity.
    assert config_fingerprint({k: v for k, v in config.items() if k != "disk"}) == (
        "c7c54829660de75bc614cdbf17c32b391a7acbb19680ef89a27a40c42c1e4301"
    )
    received = normalize_config(json.loads(json.dumps(browser_roundtrip(config))))
    assert config_fingerprint(received) == config_fingerprint(config)
    # Source metadata is preserved verbatim and excluded from geometry identity.
    assert received["source"] == config["source"]
    assert json.dumps(
        {k: v for k, v in received.items() if k != "source"}, sort_keys=True
    ) == json.dumps({k: v for k, v in config.items() if k != "source"}, sort_keys=True)


def test_integer_camber_controls_nullable_wall_angles_and_negative_zero_roundtrip(
    config,
):
    config["hub"]["inlet_angle_deg"] = None
    config["shroud"]["outlet_angle_deg"] = None
    config["dimensions"]["inlet_flow_angle_deg"] = -0.0
    config["sections"][0]["camberline"].update(
        inlet_metal_angle_deg=-30.0,
        outlet_metal_angle_deg=-20.0,
        turning_concentration=1.0,
    )
    resolved = normalize_config(config)
    received = normalize_config(browser_roundtrip(config))
    assert config_fingerprint(received) == config_fingerprint(resolved)
    assert received["hub"]["inlet_angle_deg"] is None
    assert received["shroud"]["outlet_angle_deg"] is None
    assert isinstance(
        received["sections"][0]["camberline"]["turning_concentration"], float
    )
    assert received["construction"] == {"shrouded": True, "splitters": False}


@pytest.mark.parametrize(
    "group,field",
    [
        ("dimensions", "inlet_flow_angle_deg"),
        ("hub", "bow"),
        ("blade_edges", "leading_hub"),
    ],
)
def test_physical_scalar_coercion_does_not_accept_booleans(config, group, field):
    config[group][field] = False
    with pytest.raises(ValueError, match="finite real"):
        normalize_config(config)


@pytest.mark.parametrize("field", ["schema_version", "blade_count", "numerics"])
def test_integer_only_contracts_remain_integer_only(config, field):
    if field == "numerics":
        config[field]["spanwise_points"] = 17.0
    else:
        config[field] = float(config[field])
    with pytest.raises(ValueError):
        normalize_config(config)
