"""Portable, SI-valued inputs for the explicit shrouded impeller model."""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from numbers import Integral, Real
from pathlib import Path

import numpy as np
import yaml

from ..blade_parametrization.parameters import (
    DEFAULTS,
    ParameterState,
    resolve_parameters,
)
from .meridional import (
    BladeEdges,
    MeanlineDimensions,
    WallShape,
    normalize_displacements,
)

_LENGTH_KEYS = (
    "maximum_thickness",
    "leading_edge_thickness",
    "trailing_edge_thickness",
)
_THICKNESS_DEFAULTS = {
    key: value
    for key, value in DEFAULTS["thickness"].items()
    if not key.endswith("_ratio")
}
_THICKNESS_DEFAULTS.update(
    leading_edge_fraction=0.08,
    trailing_edge_fraction=0.08,
    trailing_edge_rounding_exponent=3.0,
)
_NUMERICS = dict(streamwise_points=161, spanwise_points=17, integration_points=2001)


def _mapping(value, name, allowed):
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping.")
    unknown = set(value) - set(allowed)
    if unknown:
        raise ValueError(f"Unknown {name} fields: {sorted(map(str, unknown))}.")
    return deepcopy(dict(value))


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite real number.")
    return 0.0 if value == 0 else float(value)


def _physical_fields(values, group):
    """Canonical float scalars survive JSON's integer and signed-zero spelling."""
    return {
        key: None if value is None else _number(value, f"{group}.{key}")
        for key, value in values.items()
    }


def section_parameters(section, length_m=1.0):
    """Adapt physical thickness controls to the maintained section scalar laws.

    The law's ratios are multiplied by length_m by the caller. Only its scalar
    camber/thickness kernels are used; its axial/radial mapping is not reused.
    """
    thickness = deepcopy(section["thickness"])
    for name in _LENGTH_KEYS:
        thickness[name + "_ratio"] = thickness.pop(name + "_m") / length_m
    return resolve_parameters(
        ParameterState({"camberline": section["camberline"], "thickness": thickness})
    ).values


def normalize_config(data):
    """Resolve defaults and reject unknown or unsupported controls without mutation."""
    allowed = {
        "schema_version",
        "name",
        "source",
        "dimensions",
        "hub",
        "shroud",
        "control_point_displacements",
        "blade_edges",
        "blade_count",
        "sections",
        "inlet_interpolation",
        "construction",
        "numerics",
        "rake",
        "disk",
    }
    data = _mapping(data, "configuration", allowed)
    if (
        type(data.get("schema_version", 1)) is not int
        or data.get("schema_version", 1) != 1
    ):
        raise ValueError("Impeller schema_version must be 1.")
    result = {
        "schema_version": 1,
        "name": data.get("name", "Shrouded impeller"),
        "source": deepcopy(data.get("source", {})),
    }
    if not isinstance(result["name"], str) or not isinstance(result["source"], Mapping):
        raise ValueError("name must be text and source must be a provenance mapping.")
    try:
        result["dimensions"] = _physical_fields(
            asdict(MeanlineDimensions(**data["dimensions"])), "dimensions"
        )
        for name in ("hub", "shroud"):
            result[name] = _physical_fields(
                asdict(WallShape(**data.get(name, {}))), name
            )
        result["blade_edges"] = _physical_fields(
            asdict(BladeEdges(**data.get("blade_edges", {}))), "blade_edges"
        )
    except (KeyError, TypeError) as error:
        raise ValueError(f"Invalid channel inputs: {error}") from error
    result["control_point_displacements"] = normalize_displacements(
        data.get("control_point_displacements")
    )
    count = data.get("blade_count")
    if isinstance(count, bool) or not isinstance(count, Integral) or count < 2:
        raise ValueError("blade_count must be an integer >= 2.")
    result["blade_count"] = int(count)
    construction = _mapping(
        data.get("construction", {}), "construction", {"shrouded", "splitters"}
    )
    result["construction"] = {"shrouded": True, "splitters": False} | construction
    if (
        result["construction"]["shrouded"] is not True
        or result["construction"]["splitters"] is not False
    ):
        raise ValueError("This implementation supports shrouded main blades only.")
    result["inlet_interpolation"] = data.get("inlet_interpolation", "tangent_radius")
    if result["inlet_interpolation"] not in ("tangent_radius", "angle_span"):
        raise ValueError("inlet_interpolation must be tangent_radius or angle_span.")
    rake = {"mode": "free", "angle_deg": 0.0, "reference_azimuth_deg": None}
    rake.update(_mapping(data.get("rake", {}), "rake", rake))
    if rake["mode"] not in ("free", "preserve_le", "trailing_edge"):
        raise ValueError("rake.mode must be free, preserve_le or trailing_edge.")
    rake["angle_deg"] = _number(rake["angle_deg"], "rake.angle_deg")
    if not -90 < rake["angle_deg"] < 90:
        raise ValueError("rake.angle_deg must lie strictly between -90 and 90 degrees.")
    if rake["reference_azimuth_deg"] is not None:
        rake["reference_azimuth_deg"] = _number(
            rake["reference_azimuth_deg"], "rake.reference_azimuth_deg"
        )
    if rake["mode"] == "free" and (
        rake["angle_deg"] != 0 or rake["reference_azimuth_deg"] is not None
    ):
        raise ValueError(
            "An explicit rake angle/reference requires preserve_le or trailing_edge mode."
        )
    result["rake"] = rake
    # Disk controls are optional so historical aerodynamic-only files keep
    # their identity. When supplied, they travel with YAML and app exports.
    if "disk" in data:
        disk = {"bore_radius_m": 0.0, "nose_length_m": 0.0, "back_thickness_m": None}
        disk.update(_mapping(data["disk"], "disk", disk))
        for key, value in disk.items():
            if value is None and key == "back_thickness_m":
                continue
            disk[key] = _number(value, f"disk.{key}")
            if disk[key] < 0 or (key == "back_thickness_m" and disk[key] == 0):
                raise ValueError(f"disk.{key} must be nonnegative (back thickness strictly positive).")
        if disk["nose_length_m"] > 0 and disk["bore_radius_m"] > 0:
            raise ValueError("A rounded nose requires disk.bore_radius_m=0 (solid centre).")
        result["disk"] = disk
    result["numerics"] = _NUMERICS | _mapping(
        data.get("numerics", {}), "numerics", _NUMERICS
    )
    for name, minimum in (
        ("streamwise_points", 41),
        ("spanwise_points", 3),
        ("integration_points", 201),
    ):
        value = result["numerics"][name]
        if (
            isinstance(value, bool)
            or not isinstance(value, Integral)
            or value < minimum
        ):
            raise ValueError(f"numerics.{name} must be an integer >= {minimum}.")
        result["numerics"][name] = int(value)
    sections = data.get("sections")
    if not isinstance(sections, (list, tuple)) or len(sections) < 2:
        raise ValueError(
            "sections requires at least independent hub and shroud controls."
        )
    result["sections"] = []
    for index, section in enumerate(sections):
        section = _mapping(
            section,
            f"sections[{index}]",
            {"v", "camberline", "thickness", "stacking_offset_deg"},
        )
        v = _number(section.get("v"), "section.v")
        camber = _mapping(
            section.get("camberline", {}), "camberline", DEFAULTS["camberline"]
        )
        if not {"inlet_metal_angle_deg", "outlet_metal_angle_deg"} <= set(camber):
            raise ValueError(
                "Each section needs independent inlet and outlet metal angles."
            )
        camber = DEFAULTS["camberline"] | camber
        thickness = _THICKNESS_DEFAULTS | _mapping(
            section.get("thickness", {}),
            "thickness",
            set(_THICKNESS_DEFAULTS) | {k + "_m" for k in _LENGTH_KEYS},
        )
        thickness = {
            key: _number(value, f"thickness.{key}") for key, value in thickness.items()
        }
        for name in _LENGTH_KEYS:
            if name + "_m" not in thickness:
                raise ValueError(f"Each section needs {name}_m.")
            thickness[name + "_m"] = _number(thickness[name + "_m"], name + "_m")
        for end in ("leading", "trailing"):
            if (
                thickness[end + "_edge_thickness_m"] <= 0
                or thickness[end + "_edge_fraction"] <= 0
            ):
                raise ValueError(
                    "Rounded ends require positive shoulder thickness and rounding fraction."
                )
        normalized = {
            "v": v,
            "camberline": camber,
            "thickness": thickness,
            "stacking_offset_deg": _number(
                section.get("stacking_offset_deg", 0), "stacking_offset_deg"
            ),
        }
        section_parameters(
            normalized
        )  # Reuse maintained scalar domains/coupled fullness checks.
        normalized["camberline"] = {
            key: (
                value if key == "interpolation" else _number(value, f"camberline.{key}")
            )
            for key, value in camber.items()
        }
        result["sections"].append(normalized)
    spans = [section["v"] for section in result["sections"]]
    if spans[0] != 0 or spans[-1] != 1 or np.any(np.diff(spans) <= 0):
        raise ValueError(
            "Section v values must strictly increase from hub=0 to shroud=1."
        )
    if len({s["camberline"]["interpolation"] for s in result["sections"]}) != 1:
        raise ValueError(
            "All control sections must use the same camberline interpolation mode."
        )
    # Enforce portable finite metadata too; return defensive plain Python values.
    try:
        return json.loads(json.dumps(result, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Configuration and provenance must be finite JSON/YAML-compatible values."
        ) from error


def config_fingerprint(config):
    """Geometry identity, excluding descriptive name and source metadata."""
    inputs = {
        key: value for key, value in config.items() if key not in ("name", "source")
    }
    return hashlib.sha256(
        json.dumps(inputs, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def load_config(path):
    with Path(path).open(encoding="utf-8") as stream:
        return normalize_config(yaml.safe_load(stream))


def save_config(config, path):
    Path(path).write_text(
        yaml.safe_dump(normalize_config(config), sort_keys=False), encoding="utf-8"
    )
