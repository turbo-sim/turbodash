"""YAML presentation must preserve precision and numerical field types."""

from copy import deepcopy
import math

import pytest
import yaml

from turbodash.io.yaml import dump_turbine_yaml


@pytest.mark.parametrize("value, text", [
    (600.0, "600.00"),
    (1.1, "1.10"),
    (-0.0, "-0.00"),
    (0.3123456789123456, "0.3123456789123456"),
    (1e-9, "1.00e-09"),
    (1.2345678912345678e20, "1.2345678912345678e+20"),
])
def test_float_formatting_preserves_exact_value(value, text):
    exported = dump_turbine_yaml({"quantity": value})
    assert exported == f"quantity: {text}\n"
    restored = yaml.safe_load(exported)["quantity"]
    assert type(restored) is float
    assert restored == value
    assert math.copysign(1, restored) == math.copysign(1, value)


def test_export_restores_browser_float_types_without_changing_counts_or_source():
    document = {
        "inputs": {"inlet_property_2": 600, "stages": [
            {"radius_ratio_12": 1, "name": "stage_1"},
        ]},
        "stages_performance": [{
            "geometry_meanline": {"rotor": {
                "blade_count": 30, "blade_count_integer": None, "aspect_ratio": 2,
            }},
            "geometry_discrete": {"rotor": {
                "blade_count": 30, "blade_count_integer": 30, "aspect_ratio": 2,
            }},
        }],
        "optimization": {
            "constraints": [{"variable": "$angle", "value": 10, "normalize": 10}],
            "solver_options": {"max_iterations": 150, "plot_convergence": False},
        },
    }
    original = deepcopy(document)
    restored = yaml.safe_load(dump_turbine_yaml(document))
    assert document == original
    assert type(document["inputs"]["inlet_property_2"]) is int
    assert type(restored["inputs"]["inlet_property_2"]) is float
    assert type(restored["inputs"]["stages"][0]["radius_ratio_12"]) is float
    stage = restored["stages_performance"][0]
    assert type(stage["geometry_meanline"]["rotor"]["blade_count"]) is float
    assert type(stage["geometry_discrete"]["rotor"]["blade_count"]) is int
    assert type(stage["geometry_discrete"]["rotor"]["blade_count_integer"]) is int
    assert type(stage["geometry_discrete"]["rotor"]["aspect_ratio"]) is float
    assert type(restored["optimization"]["constraints"][0]["value"]) is float
    options = restored["optimization"]["solver_options"]
    assert type(options["max_iterations"]) is int
    assert options["plot_convergence"] is False
    # The custom formatting does not affect unrelated PyYAML users.
    assert yaml.safe_dump({"quantity": 1.1}) == "quantity: 1.1\n"
