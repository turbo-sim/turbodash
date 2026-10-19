"""Inlet editing must preserve the thermodynamic state, ordering and precision."""

import math

import CoolProp.CoolProp as CP
import jaxprop as jxp
import pytest

from turbodash.apps.shared.fluid_inputs import (
    convert_pair,
    fluid_options,
    pair_fields,
    pair_properties,
    property_pair_options,
    validate_inlet_inputs,
    validate_pair_values,
)


def test_every_backend_pair_has_readable_ordered_fields():
    options = property_pair_options()
    assert {item["value"] for item in options} == set(jxp.INPUT_PAIRS)
    for option in options:
        pair = option["value"]
        fields = pair_fields(pair)
        assert len(fields) == 2
        assert "".join(field["property"] for field in fields) + "_INPUTS" == pair
        assert all(
            field["label"] in option["label"] and field["unit"] for field in fields
        )
    assert pair_properties("HmassP_INPUTS") == ("Hmass", "P")
    assert pair_fields("PQ_INPUTS")[1]["unit"] == "mol/mol"
    assert pair_fields("DmolarHmolar_INPUTS")[0]["unit"] == "mol/m³"
    assert pair_fields("PSmolar_INPUTS")[1]["unit"] == "J/(mol·K)"


def test_fluid_options_retain_loaded_alias():
    options = fluid_options("CO2")
    assert "CO2" in options
    assert "Water" in options
    assert len(options) == len(set(options))
    assert options == sorted(options, key=str.casefold)


@pytest.mark.parametrize("pair", [None, "PT", "massP_INPUTS", 9])
def test_invalid_pair_is_rejected(pair):
    with pytest.raises(ValueError, match="supported inlet property pair"):
        pair_fields(pair)


@pytest.mark.parametrize(
    "values",
    [
        None,
        [1e5],
        [1e5, 300, 1],
        [True, 300],
        [1e5, None],
        ["100000", 300],
        [math.nan, 300],
        [1e5, math.inf],
    ],
)
def test_invalid_numeric_drafts_are_rejected(values):
    with pytest.raises(ValueError):
        validate_pair_values("PT_INPUTS", values)


@pytest.mark.parametrize(
    "pair,values",
    [
        ("PT_INPUTS", [0, 300]),
        ("PT_INPUTS", [1e5, -1]),
        ("DmassT_INPUTS", [0, 300]),
        ("DmolarT_INPUTS", [-1, 300]),
        ("PQ_INPUTS", [1e5, -0.1]),
        ("QT_INPUTS", [1.1, 300]),
    ],
)
def test_physical_input_bounds_are_checked(pair, values):
    with pytest.raises(ValueError):
        validate_pair_values(pair, values)


def test_reference_dependent_properties_can_be_negative():
    assert validate_pair_values("HmassSmass_INPUTS", [-1, -2]) == (-1, -2)
    assert validate_pair_values("PUmass_INPUTS", [1e5, -3]) == (1e5, -3)


def test_existing_yaml_field_names_are_validated():
    assert validate_inlet_inputs(
        {
            "inlet_property_pair": "PT_INPUTS",
            "inlet_property_1": 1e5,
            "inlet_property_2": 300,
        }
    ) == (1e5, 300)
    with pytest.raises(ValueError, match="both inlet properties"):
        validate_inlet_inputs({"inlet_property_pair": "PT_INPUTS"})


def test_identical_pair_and_shared_values_preserve_entered_precision():
    values = (100000.123456789, 423.123456789)
    assert convert_pair("Water", "PT_INPUTS", values, "PT_INPUTS") == values
    enthalpy, pressure = convert_pair("Water", "PT_INPUTS", values, "HmassP_INPUTS")
    assert pressure == values[0]
    assert enthalpy == pytest.approx(
        CP.PropsSI("Hmass", "P", values[0], "T", values[1], "Water")
    )
    restored = convert_pair("Water", "HmassP_INPUTS", (enthalpy, pressure), "PT_INPUTS")
    assert restored[0] == pressure
    assert restored[1] == pytest.approx(values[1], rel=1e-8)


@pytest.mark.parametrize(
    "pair",
    [
        "DmassT_INPUTS",
        "DmolarT_INPUTS",
        "HmassP_INPUTS",
        "HmolarP_INPUTS",
        "PSmass_INPUTS",
        "PSmolar_INPUTS",
        "PUmass_INPUTS",
        "PUmolar_INPUTS",
    ],
)
def test_mass_and_molar_conversions_use_the_correct_backend_units(pair):
    values = (2e5, 430)
    actual = convert_pair("Water", "PT_INPUTS", values, pair)
    expected = [
        CP.PropsSI(prop, "P", values[0], "T", values[1], "Water")
        for prop in pair_properties(pair)
    ]
    assert actual == pytest.approx(expected, rel=1e-9)
    restored = convert_pair("Water", pair, actual, "PT_INPUTS")
    assert restored == pytest.approx(values, rel=1e-8)


@pytest.mark.parametrize("quality", [0.0, 0.123456789123, 1.0])
def test_quality_is_preserved_exactly_when_switching_saturation_pairs(quality):
    pressure = 200000.123456789
    q, temperature = convert_pair(
        "Water", "PQ_INPUTS", (pressure, quality), "QT_INPUTS"
    )
    assert q == quality
    p, restored_q = convert_pair("Water", "QT_INPUTS", (q, temperature), "PQ_INPUTS")
    assert restored_q == quality
    assert p == pytest.approx(pressure, rel=1e-8)


def test_saturated_mixture_quality_can_be_recovered_from_enthalpy_pressure():
    pressure, quality = 2e5, 0.37213456789
    hp = convert_pair("Water", "PQ_INPUTS", (pressure, quality), "HmassP_INPUTS")
    restored = convert_pair("Water", "HmassP_INPUTS", hp, "PQ_INPUTS")
    assert restored[0] == pressure
    assert restored[1] == pytest.approx(quality, abs=1e-10)


def test_single_phase_quality_cannot_be_silently_substituted():
    with pytest.raises(ValueError, match="Vapor quality is undefined"):
        convert_pair("Water", "PT_INPUTS", (1e5, 450), "PQ_INPUTS")


def test_saturated_pt_transition_rejects_loss_of_quality_information():
    with pytest.raises(ValueError, match="Enter both inlet properties"):
        convert_pair("Water", "PQ_INPUTS", (1e5, 0.5), "PT_INPUTS")


def test_invalid_source_state_and_unknown_fluid_fail_explicitly():
    with pytest.raises(ValueError, match="Cannot resolve the current Water inlet"):
        convert_pair("Water", "PT_INPUTS", (1e5, 1), "HmassP_INPUTS")
    with pytest.raises(
        ValueError, match="Cannot resolve the current unknown-fluid inlet"
    ):
        convert_pair("unknown-fluid", "PT_INPUTS", (1e5, 300), "HmassP_INPUTS")
