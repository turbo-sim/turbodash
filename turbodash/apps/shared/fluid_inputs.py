"""Readable inlet controls and safe transitions between thermodynamic pairs.

The saved identifiers and value ordering are the CoolProp/jaxprop conventions.
Conversion either preserves the same physical inlet or raises ``ValueError``;
callers must clear both new fields on failure rather than relabel old numbers.
"""

from functools import lru_cache
import math
from numbers import Real
import re

import CoolProp.CoolProp as CP
import jaxprop as jxp

INLET_FIELDS = ("inlet_property_pair", "inlet_property_1", "inlet_property_2")
PROPERTIES = {
    "P": ("Pressure", "Pa"),
    "T": ("Temperature", "K"),
    # CoolProp Q is molar quality. Mass and molar quality coincide for pure fluids.
    "Q": ("Vapor quality", "mol/mol"),
    "Dmass": ("Mass density", "kg/m³"),
    "Dmolar": ("Molar density", "mol/m³"),
    "Hmass": ("Specific enthalpy", "J/kg"),
    "Hmolar": ("Molar enthalpy", "J/mol"),
    "Smass": ("Specific entropy", "J/(kg·K)"),
    "Smolar": ("Molar entropy", "J/(mol·K)"),
    "Umass": ("Specific internal energy", "J/kg"),
    "Umolar": ("Molar internal energy", "J/mol"),
}
_PROPERTY_PATTERN = "(" + "|".join(PROPERTIES) + ")"
_POSITIVE_PROPERTIES = {"P", "T", "Dmass", "Dmolar"}


def pair_properties(pair):
    """Return the two property identifiers in the backend's required order."""
    if not isinstance(pair, str) or pair not in jxp.INPUT_PAIRS:
        raise ValueError("Select a supported inlet property pair.")
    match = re.fullmatch(_PROPERTY_PATTERN + _PROPERTY_PATTERN + "_INPUTS", pair)
    if match is None:
        raise ValueError(f"Unknown properties in inlet pair {pair!r}.")
    return match.groups()


def pair_fields(pair):
    """Return ``property``, readable ``label``, and SI ``unit`` for both fields."""
    return tuple(
        {"property": prop, "label": PROPERTIES[prop][0], "unit": PROPERTIES[prop][1]}
        for prop in pair_properties(pair)
    )


def property_pair_options():
    """Dash dropdown options covering every pair exposed by jaxprop."""
    return [
        {
            "label": f"{pair.removesuffix('_INPUTS')} — "
            + " / ".join(field["label"] for field in pair_fields(pair)),
            "value": pair,
        }
        for pair in jxp.INPUT_PAIRS
    ]


@lru_cache(maxsize=1)
def _supported_fluids():
    return tuple(CP.get_global_param_string("FluidsList").split(","))


def fluid_options(current=None):
    """Sorted fluid names, retaining a loaded case's valid alias or custom name."""
    names = set(_supported_fluids())
    if isinstance(current, str) and current.strip():
        names.add(current)
    return sorted(names, key=str.casefold)


def validate_pair_values(pair, values):
    """Validate a pair without an EOS call; return two finite Python floats.

    Enthalpy, entropy and internal energy may be negative depending on the
    reference state. Their validity is therefore decided by the EOS, not by an
    arbitrary positive UI bound. Booleans and numeric strings are not numbers.
    """
    properties = pair_properties(pair)
    try:
        values = tuple(values)
    except TypeError as exc:
        raise ValueError("Enter both inlet properties.") from exc
    if len(values) != 2:
        raise ValueError("Enter both inlet properties.")
    result = []
    for prop, value in zip(properties, values):
        label = PROPERTIES[prop][0]
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(value)
        ):
            raise ValueError(f"{label} must be a finite number.")
        if prop in _POSITIVE_PROPERTIES and value <= 0:
            raise ValueError(f"{label} must be greater than zero.")
        if prop == "Q" and not 0 <= value <= 1:
            raise ValueError("Vapor quality must be between 0 and 1.")
        result.append(float(value))
    return tuple(result)


def validate_inlet_inputs(parameters):
    """Validate the existing saved-case inlet fields, returning their values."""
    if not all(name in parameters for name in INLET_FIELDS):
        raise ValueError("Specify the inlet property pair and both inlet properties.")
    return validate_pair_values(
        parameters[INLET_FIELDS[0]], [parameters[name] for name in INLET_FIELDS[1:]]
    )


def _state(fluid, pair, values, backend):
    state = CP.AbstractState(backend, fluid)
    state.update(jxp.INPUT_PAIRS[pair], *values)
    if any(
        not math.isfinite(value) or value <= 0
        for value in (state.p(), state.T(), state.rhomass())
    ):
        raise ValueError(
            "The inlet must have finite, positive pressure, temperature and density."
        )
    return state


def convert_pair(fluid, old_pair, values, new_pair, *, backend="HEOS"):
    """Resolve ``values`` into a new pair for the same fluid and physical state.

    Shared properties are copied directly to preserve entered precision. The
    target flash is checked too: a property combination exposed by CoolProp
    may not support every fluid/phase, and PT does not determine vapor quality
    at saturation. Failed or ambiguous conversions raise ``ValueError`` and
    never return a partially converted pair.
    """
    original = validate_pair_values(old_pair, values)
    target_properties = pair_properties(new_pair)
    if not isinstance(fluid, str) or not fluid.strip():
        raise ValueError(
            "Select a working fluid before changing the inlet property pair."
        )
    try:
        source = _state(fluid, old_pair, original, backend)
    except (ValueError, RuntimeError) as exc:
        raise ValueError(f"Cannot resolve the current {fluid} inlet: {exc}") from exc
    if new_pair == old_pair:
        return original

    existing = dict(zip(pair_properties(old_pair), original))
    try:
        converted = tuple(
            (
                existing[prop]
                if prop in existing
                else source.keyed_output(CP.get_parameter_index(prop))
            )
            for prop in target_properties
        )
        if (
            "Q" in target_properties
            and not 0 <= converted[target_properties.index("Q")] <= 1
        ):
            raise ValueError(
                "Vapor quality is undefined for the current single-phase inlet."
            )
        converted = validate_pair_values(new_pair, converted)
        target = _state(fluid, new_pair, converted, backend)
        # Some saturation flashes have multiple roots. Do not silently move to
        # a different root while claiming that the inlet has been preserved.
        for first, second in zip(
            (source.p(), source.T(), source.rhomass()),
            (target.p(), target.T(), target.rhomass()),
        ):
            if not math.isclose(first, second, rel_tol=1e-6, abs_tol=1e-8):
                raise ValueError(
                    "The new pair does not uniquely preserve the current inlet state."
                )
    except (ValueError, RuntimeError) as exc:
        raise ValueError(
            f"Cannot preserve the {fluid} inlet with {new_pair.removesuffix('_INPUTS')}. "
            f"Enter both inlet properties for the new pair. {exc}"
        ) from exc
    return converted
