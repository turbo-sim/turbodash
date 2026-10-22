"""Independent V4 inputs, inherited defaults, and sparse geometry overrides.

Lengths use metres and explicitly named angle fields use degrees. A source is
an immutable snapshot of independent geometry inputs, not a TurboDash solver
result. No geometry edits are written back to the source. The resolved values
are disposable products; persist :meth:`ParameterState.to_dict` instead.

The optional source-only ``cascade.blade_count_continuous`` supplies
``ceil(N)`` when no integer count is supplied. An explicit source integer is
already a realization and takes precedence. A geometry override is applied
after this choice. Pitch and other dependent quantities belong to geometry
calculations and are never independently editable inputs.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Mapping

import yaml

from .thickness import fullness_coefficients


DEFAULTS = {
    "geometry": {
        "type": "axial",
        "mean_radius": 0.05,
        "blade_height": 0.01,
        "meridional_chord": 0.015,
        "inlet_radius": 0.02,
        "outlet_radius": 0.035,
    },
    "cascade": {"blade_count": 24},
    "camberline": {
        "inlet_metal_angle_deg": 0.0,
        "outlet_metal_angle_deg": 65.0,
        "interpolation": "angle",
        "turning_location": 0.5,
        "turning_concentration": 1.0,
    },
    "thickness": {
        "maximum_thickness_ratio": 0.25,
        "maximum_thickness_location": 0.3,
        "leading_edge_thickness_ratio": 0.08,
        "trailing_edge_thickness_ratio": 0.02,
        "leading_edge_fraction": 0.1,
        "trailing_edge_fraction": 0.025,
        "leading_edge_rounding_exponent": 3.0,
        "trailing_edge_rounding_exponent": 3.0,
        "shape_exponent": 2.0,
        "leading_edge_fullness": 0.0,
        "trailing_edge_fullness": 0.0,
    },
    "leading_edge": {"relief_strength": 0.15, "pressure_side": "auto"},
    "trailing_edge": {"relief_strength": 0.0, "pressure_side": "auto"},
    "correction": {
        "preserve_physical_thickness": False,
        "warning_tolerance": 0.01,
    },
    "numerics": {"number_of_points": 601},
    "flow_domain": {"inlet_chords": 0.5, "outlet_chords": 1.0},
}


def _spec(label, group, kind="float", units="-", **limits):
    return {"label": label, "group": group, "type": kind, "units": units, **limits}


# Bound exclusivity is explicit so the same metadata supports validation and
# UI help. Limits express mathematical domains, not aerodynamic guarantees.
FIELD_SPECS = {
    "geometry.type": _spec("Stream surface", "Meridional geometry", "enum", options=["axial", "radial"]),
    "geometry.mean_radius": _spec("Mean radius", "Meridional geometry", units="m", min=0.0, min_exclusive=True, step=0.001),
    "geometry.blade_height": _spec("Blade height", "Meridional geometry", units="m", min=0.0, min_exclusive=True, step=0.001),
    "geometry.meridional_chord": _spec("Axial meridional chord", "Meridional geometry", units="m", min=0.0, min_exclusive=True, step=0.001),
    "geometry.inlet_radius": _spec("Radial inlet radius", "Meridional geometry", units="m", min=0.0, min_exclusive=True, step=0.001),
    "geometry.outlet_radius": _spec("Radial outlet radius", "Meridional geometry", units="m", min=0.0, min_exclusive=True, step=0.001),
    "cascade.blade_count": _spec("Number of blades", "Blade count and pitch", "int", min=2, step=1),
    "camberline.inlet_metal_angle_deg": _spec("Inlet metal angle", "Camberline and turning", units="deg", min=-90.0, max=90.0, min_exclusive=True, max_exclusive=True, step=1.0),
    "camberline.outlet_metal_angle_deg": _spec("Outlet metal angle", "Camberline and turning", units="deg", min=-90.0, max=90.0, min_exclusive=True, max_exclusive=True, step=1.0),
    "camberline.interpolation": _spec("Turning interpolation", "Camberline and turning", "enum", options=["angle", "slope"]),
    "camberline.turning_location": _spec("Half-turning location", "Camberline and turning", min=0.0, max=1.0, min_exclusive=True, max_exclusive=True, step=0.01),
    "camberline.turning_concentration": _spec("Turning concentration", "Camberline and turning", min=1.0, step=0.1),
    "thickness.maximum_thickness_ratio": _spec("Nominal maximum thickness ratio", "Thickness distribution", min=0.0, min_exclusive=True, step=0.01),
    "thickness.maximum_thickness_location": _spec("Nominal maximum location", "Thickness distribution", min=0.0, max=1.0, min_exclusive=True, max_exclusive=True, step=0.01),
    "thickness.leading_edge_thickness_ratio": _spec("Unrounded leading thickness ratio", "Thickness distribution", min=0.0, step=0.01),
    "thickness.trailing_edge_thickness_ratio": _spec("Unrounded trailing thickness ratio", "Thickness distribution", min=0.0, step=0.005),
    "thickness.leading_edge_fraction": _spec("Leading rounding fraction", "Thickness distribution", min=0.0, max=1.0, step=0.01),
    "thickness.trailing_edge_fraction": _spec("Trailing rounding fraction", "Thickness distribution", min=0.0, max=1.0, step=0.005),
    "thickness.leading_edge_fullness": _spec("Leading shoulder fullness", "Thickness distribution", min=-0.5, max=0.5, step=0.01),
    "thickness.trailing_edge_fullness": _spec("Trailing shoulder fullness", "Thickness distribution", min=-0.5, max=0.5, step=0.01),
    "thickness.shape_exponent": _spec("Body shape exponent", "Thickness distribution", min=1.0, step=0.1),
    "thickness.leading_edge_rounding_exponent": _spec("Leading rounding exponent", "Thickness distribution", min=2.0, step=0.1),
    "thickness.trailing_edge_rounding_exponent": _spec("Trailing rounding exponent", "Thickness distribution", min=2.0, step=0.1),
    "leading_edge.relief_strength": _spec("Pressure-side relief", "Leading-edge relief", min=0.0, max=1.0, max_exclusive=True, step=0.05),
    "leading_edge.pressure_side": _spec("Pressure-side convention", "Leading-edge relief", "enum", options=["auto", "plus", "minus"]),
    "trailing_edge.relief_strength": _spec("Pressure-side relief", "Trailing-edge relief", min=0.0, max=1.0, max_exclusive=True, step=0.05),
    "trailing_edge.pressure_side": _spec("Pressure-side convention", "Trailing-edge relief", "enum", options=["auto", "plus", "minus"]),
    "correction.preserve_physical_thickness": _spec("Preserve physical thickness", "Thickness correction", "bool"),
    "correction.warning_tolerance": _spec("Relative thickness error tolerance", "Thickness correction", min=0.0, step=0.005),
    "numerics.number_of_points": _spec("Base sampling points", "Resolution", "int", min=21, step=20),
    "flow_domain.inlet_chords": _spec("Inlet extension / chord", "Flow domain", min=0.0, min_exclusive=True, step=0.1),
    "flow_domain.outlet_chords": _spec("Outlet extension / chord", "Flow domain", min=0.0, min_exclusive=True, step=0.1),
}

_CONTINUOUS_COUNT = "cascade.blade_count_continuous"
_SCHEMA_VERSION = 2


def _flatten(source: Mapping[str, Any], *, allow_continuous=False):
    if not isinstance(source, Mapping):
        raise ValueError("Parameter source must be a nested mapping.")
    flat = {}
    for group, members in source.items():
        if group not in DEFAULTS:
            raise ValueError(f"Unknown parameter group: {group!r}.")
        if not isinstance(members, Mapping):
            raise ValueError(f"Parameter group {group!r} must be a mapping.")
        for field, value in members.items():
            path = f"{group}.{field}"
            if path not in FIELD_SPECS and not (allow_continuous and path == _CONTINUOUS_COUNT):
                raise ValueError(f"Unknown parameter: {path}.")
            flat[path] = value
    return flat


def _assign(values, path, value):
    group, field = path.split(".", 1)
    values[group][field] = value


def _validate_value(path, value):
    spec = FIELD_SPECS[path]
    kind = spec["type"]
    if kind == "bool":
        if type(value) is not bool:
            raise ValueError(f"{path} must be a boolean (true or false).")
        return value
    if kind == "enum":
        if not isinstance(value, str) or value not in spec["options"]:
            raise ValueError(f"{path} must be one of {', '.join(spec['options'])}.")
        return value
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{path} must be a finite {'integer' if kind == 'int' else 'number'}.")
    if kind == "int" and not isinstance(value, Integral):
        raise ValueError(f"{path} must be an integer; fractional and floating-point inputs are not accepted.")
    if not math.isfinite(value):
        raise ValueError(f"{path} must be finite.")
    for bound in ("min", "max"):
        if bound not in spec:
            continue
        limit = spec[bound]
        exclusive = spec.get(f"{bound}_exclusive", False)
        invalid = (value <= limit if exclusive else value < limit) if bound == "min" else (value >= limit if exclusive else value > limit)
        if invalid:
            relation = ("greater than" if exclusive else "at least") if bound == "min" else ("less than" if exclusive else "at most")
            raise ValueError(f"{path} must be {relation} {limit}.")
    return int(value) if kind == "int" else float(value)


def _fingerprint(values):
    payload = json.dumps(values, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _plain_scalars(value):
    """Keep numerical-library scalar inputs portable in saved state."""
    if isinstance(value, Mapping):
        return {key: _plain_scalars(item) for key, item in value.items()}
    if isinstance(value, bool):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        return float(value)
    return value


@dataclass(frozen=True, init=False)
class ParameterState:
    """A source snapshot and explicit overrides, with functional updates.

    Public mapping properties return defensive copies. Editing controls must
    call ``with_override``; source refresh must call ``with_source``. This
    prevents mutable aliases from turning inherited values into a second
    editable geometry record.
    """

    _source: dict
    _overrides: dict
    label: str

    def __init__(self, source=None, overrides=None, label="Standalone geometry"):
        if not isinstance(label, str):
            raise ValueError("Parameter source label must be a string.")
        source = {} if source is None else source
        overrides = {} if overrides is None else overrides
        if not isinstance(source, Mapping):
            raise ValueError("Parameter source must be a nested mapping.")
        if not isinstance(overrides, Mapping):
            raise ValueError("Parameter overrides must be a mapping of dotted field names to values.")
        object.__setattr__(self, "_source", deepcopy(dict(source)))
        object.__setattr__(self, "_overrides", deepcopy(dict(overrides)))
        object.__setattr__(self, "label", label)
        resolve_parameters(self)

    @property
    def source(self):
        return deepcopy(self._source)

    @property
    def overrides(self):
        return deepcopy(self._overrides)

    def to_dict(self):
        """Return JSON/YAML-serializable state, preserving override intent."""
        return {"schema_version": _SCHEMA_VERSION, "label": self.label,
                "source": _plain_scalars(self._source),
                "overrides": _plain_scalars(self._overrides)}

    def with_override(self, path, value):
        if path not in FIELD_SPECS:
            raise ValueError(f"Unknown override parameter: {path!r}.")
        overrides = self.overrides
        overrides[path] = value
        return ParameterState(self._source, overrides, self.label)

    def reset(self, path):
        """Reset one field to its current source/default, not its old value."""
        if path not in FIELD_SPECS:
            raise ValueError(f"Unknown override parameter: {path!r}.")
        overrides = self.overrides
        overrides.pop(path, None)
        return ParameterState(self._source, overrides, self.label)

    def with_source(self, source, label=None):
        """Replace the inherited baseline while retaining explicit overrides.

        Incompatible combinations raise ValueError; the original state remains
        valid and unchanged. Callers can retain its preview and display the
        source-update error rather than silently dropping the user's edits.
        """
        return ParameterState(source, self._overrides, self.label if label is None else label)


@dataclass(frozen=True)
class ResolvedParameters:
    """Validated effective inputs; derived geometry is computed downstream."""

    values: dict
    provenance: dict[str, str]
    fingerprint: str


def resolve_parameters(state: ParameterState) -> ResolvedParameters:
    """Resolve defaults → source realization → sparse overrides, then validate.

    The fingerprint identifies effective input values, so metadata-only source
    changes do not invalidate a geometry cache. Provenance remains separate.
    Radius, height, chord, and count are independent inputs; changing height
    does not silently resize the chord or alter the selected blade count.
    """
    if not isinstance(state, ParameterState):
        raise ValueError("resolve_parameters requires a ParameterState.")
    source = _flatten(state._source, allow_continuous=True)
    for path, value in list(source.items()):
        if path != _CONTINUOUS_COUNT:
            source[path] = _validate_value(path, value)
    has_continuous = _CONTINUOUS_COUNT in source
    continuous = source.pop(_CONTINUOUS_COUNT, None)
    if has_continuous:
        if isinstance(continuous, bool) or not isinstance(continuous, Real) or not math.isfinite(continuous) or continuous <= 0:
            raise ValueError(f"{_CONTINUOUS_COUNT} must be finite and positive.")
        if "cascade.blade_count" not in source:
            source["cascade.blade_count"] = math.ceil(continuous)

    values = deepcopy(DEFAULTS)
    provenance = dict.fromkeys(FIELD_SPECS, "default")
    for path, value in source.items():
        _assign(values, path, value)
        provenance[path] = "inherited"
    for path, value in state._overrides.items():
        if path not in FIELD_SPECS:
            raise ValueError(f"Unknown override parameter: {path!r}.")
        _assign(values, path, _validate_value(path, value))
        provenance[path] = "override"
    for path, value in _flatten(values).items():
        _assign(values, path, _validate_value(path, value))
    validate_parameters(values)
    return ResolvedParameters(values, provenance, _fingerprint(values))


def validate_parameters(values: Mapping[str, Any]) -> None:
    """Validate a complete effective model without mutating it.

    Use :func:`state_from_dict` and :func:`resolve_parameters` to supply defaults
    for partial configurations. The geometry calculation entry point can call
    this function directly when given complete plain dictionaries.
    """
    flat = _flatten(values)
    missing = set(FIELD_SPECS) - set(flat)
    if missing:
        raise ValueError(f"Missing effective parameters: {', '.join(sorted(missing))}.")
    for path, value in flat.items():
        _validate_value(path, value)
    fullness_coefficients(values["thickness"])
    geometry = values["geometry"]
    if geometry["type"] == "axial" and geometry["mean_radius"] <= 0.5 * geometry["blade_height"]:
        raise ValueError("geometry.mean_radius must exceed geometry.blade_height / 2 so the axial hub radius is positive.")
    if geometry["type"] == "radial" and geometry["inlet_radius"] == geometry["outlet_radius"]:
        raise ValueError("geometry.inlet_radius and geometry.outlet_radius must differ for a radial blade row.")


def state_from_dict(data) -> ParameterState:
    """Load saved state or a plain nested geometry-input mapping.

    Plain inputs become an inherited standalone baseline; they are not copied
    into the overrides. Unknown fields are rejected to catch misspelled YAML.
    """
    if not isinstance(data, Mapping):
        raise ValueError("The geometry configuration must be a mapping.")
    envelope_keys = {"schema_version", "label", "source", "overrides"}
    if any(key in data for key in envelope_keys):
        unknown = set(data) - envelope_keys
        if unknown:
            raise ValueError(f"Unknown configuration fields: {', '.join(sorted(map(str, unknown)))}.")
        version = data.get("schema_version", _SCHEMA_VERSION)
        if type(version) is not int or version not in (1, _SCHEMA_VERSION):
            raise ValueError(f"Unsupported parameter schema_version: {version!r}; expected {_SCHEMA_VERSION}.")
        source = deepcopy(data.get("source", {}))
        if version == 1 and isinstance(source, Mapping):
            # Preserve the original V4 shape when loading an older saved design.
            thickness = source.setdefault("thickness", {})
            if isinstance(thickness, Mapping):
                thickness.setdefault("leading_edge_rounding_exponent", 3.0)
                thickness.setdefault("trailing_edge_rounding_exponent", 2.0)
        return ParameterState(source, data.get("overrides", {}), data.get("label", "Standalone geometry"))
    return ParameterState(data)


def load_case(path: str | Path) -> ParameterState:
    """Load a YAML example or saved parameter state without importing the UI."""
    path = Path(path)
    try:
        with path.open(encoding="utf-8") as stream:
            data = yaml.safe_load(stream)
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid YAML in {path.name}: {error}") from error
    return state_from_dict(data)
