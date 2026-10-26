"""Validated YAML configuration and engineering controls for the local explorer.

All persistent inputs use the study's YAML schema. Displayed lengths are mm;
the geometry kernel and saved configurations use metres. No Dash dependency.
"""

from copy import deepcopy
from dataclasses import asdict
from numbers import Real
from pathlib import Path

import numpy as np
import yaml

from ..meridional import (
    BladeEdges,
    BladeMeridional,
    MeanlineDimensions,
    MeridionalChannel,
    WallShape,
)

STUDY = Path(__file__).resolve().parents[1]
GROUPS = (
    "Dimensions",
    "Station directions",
    "Hub wall",
    "Shroud wall",
    "Blade edges",
    "Splitter",
)


def _field(group, label, unit, low, high, step, help_text):
    return dict(
        group=group,
        label=label,
        unit=unit,
        scale=1000 if unit == "mm" else 1,
        min=low,
        max=high,
        step=step,
        help=help_text,
    )


FIELD_SPECS = {
    "dimensions.inlet_hub_radius": _field(
        "Dimensions",
        "Inlet hub radius",
        "mm",
        10,
        50,
        0.5,
        "Actual hub radius at the nominal inlet station.",
    ),
    "dimensions.inlet_shroud_radius": _field(
        "Dimensions",
        "Inlet shroud radius",
        "mm",
        40,
        85,
        0.5,
        "Actual casing radius at the nominal inlet station; must exceed the hub radius.",
    ),
    "dimensions.outlet_radius": _field(
        "Dimensions",
        "Outlet mean radius",
        "mm",
        85,
        150,
        0.5,
        "Midpoint radius of the outlet station. For a radial outlet both wall radii equal this.",
    ),
    "dimensions.outlet_width": _field(
        "Dimensions",
        "Outlet width",
        "mm",
        3,
        18,
        0.25,
        "Width measured normal to the nominal outlet direction.",
    ),
    "dimensions.axial_length": _field(
        "Dimensions",
        "Axial length",
        "mm",
        35,
        95,
        0.5,
        "Axial distance between the inlet and outlet station midpoints.",
    ),
    "dimensions.inlet_flow_angle_deg": _field(
        "Station directions",
        "Inlet inclination",
        "°",
        -30,
        35,
        0.5,
        "Measured from +z towards +r. Zero is axial; the inlet connector rotates with this direction.",
    ),
    "dimensions.outlet_flow_angle_deg": _field(
        "Station directions",
        "Outlet inclination",
        "°",
        40,
        90,
        0.5,
        "Measured from +z towards +r. Ninety degrees is radial.",
    ),
}

for wall, group in (("hub", "Hub wall"), ("shroud", "Shroud wall")):
    FIELD_SPECS.update(
        {
            f"{wall}.inlet_angle_deg": _field(
                group,
                "Inlet wall angle",
                "°",
                -20,
                35,
                0.5,
                "Wall tangent from +z. Follow inlet links it to the nominal inlet direction.",
            ),
            f"{wall}.outlet_angle_deg": _field(
                group,
                "Outlet wall angle",
                "°",
                40,
                105,
                0.5,
                "Wall tangent from +z. Follow outlet links it to the nominal outlet direction.",
            ),
            f"{wall}.inlet_tangent_fraction": _field(
                group,
                "Inlet tangent reach",
                "–",
                0.1,
                0.85,
                0.01,
                "Fraction of projected wall chord influencing inlet straightness and turn distribution.",
            ),
            f"{wall}.outlet_tangent_fraction": _field(
                group,
                "Outlet tangent reach",
                "–",
                0.1,
                0.85,
                0.01,
                "Fraction of projected wall chord influencing the final turn and outlet straightness.",
            ),
            f"{wall}.bow": _field(
                group,
                "Wall bow / chord",
                "–",
                -0.06,
                0.06,
                0.002,
                "Signed normal offset at polynomial midpoint, divided by wall chord. Positive is left of the chord.",
            ),
        }
    )

FIELD_SPECS.update(
    {
        "blade.leading_hub": _field(
            "Blade edges",
            "Hub leading edge",
            "s",
            0,
            0.35,
            0.005,
            "Normalized hub wall arclength; zero is the nominal channel inlet.",
        ),
        "blade.leading_shroud": _field(
            "Blade edges",
            "Shroud leading edge",
            "s",
            0,
            0.35,
            0.005,
            "Normalized shroud wall arclength. Independent from the hub leading edge.",
        ),
        "blade.trailing_hub": _field(
            "Blade edges",
            "Hub trailing edge",
            "s",
            0.7,
            1,
            0.005,
            "Normalized hub wall arclength; one is the nominal channel outlet.",
        ),
        "blade.trailing_shroud": _field(
            "Blade edges",
            "Shroud trailing edge",
            "s",
            0.7,
            1,
            0.005,
            "Normalized shroud wall arclength; reduce to trim the blade within the channel.",
        ),
        "blade.leading_bow": _field(
            "Blade edges",
            "Leading-edge bow",
            "s",
            -0.12,
            0.12,
            0.002,
            "Additional streamwise displacement at midspan. Edge must remain inside the channel.",
        ),
        "blade.trailing_bow": _field(
            "Blade edges",
            "Trailing-edge bow",
            "s",
            -0.12,
            0.12,
            0.002,
            "Additional streamwise displacement at midspan; negative trims a baseline trailing edge.",
        ),
        "splitter_start": _field(
            "Splitter",
            "Splitter start",
            "ξ",
            0.1,
            0.55,
            0.005,
            "Fraction of the main blade's local s interval. This is physical arclength fraction only on the walls.",
        ),
    }
)
ANGLE_PATHS = tuple(
    f"{wall}.{end}_angle_deg"
    for wall in ("hub", "shroud")
    for end in ("inlet", "outlet")
)


def default_config():
    return {
        "dimensions": asdict(MeanlineDimensions.from_ratios()),
        "hub": asdict(WallShape()),
        "shroud": asdict(WallShape()),
        "blade": asdict(BladeEdges()),
        "splitter_start": 0.25,
    }


def _lookup(config, path):
    if "." not in path:
        return config[path]
    group, name = path.split(".", 1)
    return config[group][name]


def _assign(config, path, value):
    if "." not in path:
        config[path] = value
    else:
        group, name = path.split(".", 1)
        config[group][name] = value


def normalize_config(data):
    """Validate structure/scalars and fill defaults, without hiding invalid shapes."""
    if not isinstance(data, dict):
        raise ValueError("Configuration must be a YAML mapping.")
    result = default_config()
    unknown = set(data) - set(result)
    if unknown:
        raise ValueError(
            f"Unknown configuration group: {', '.join(map(str, unknown))}."
        )
    if "dimensions" not in data or not isinstance(data["dimensions"], dict):
        raise ValueError("Configuration needs a dimensions mapping.")
    required = {
        "inlet_hub_radius",
        "inlet_shroud_radius",
        "outlet_radius",
        "outlet_width",
    }
    if missing := required - set(data["dimensions"]):
        raise ValueError(f"Missing dimensions: {', '.join(sorted(missing))}.")
    for group, value in data.items():
        if group == "splitter_start":
            result[group] = value
            continue
        if not isinstance(value, dict):
            raise ValueError(f"{group} must be a YAML mapping.")
        if unknown := set(value) - set(result[group]):
            raise ValueError(
                f"Unknown {group} parameter: {', '.join(map(str, unknown))}."
            )
        result[group].update(value)
    # An omitted length follows the kernel default for this configuration's r2.
    if (
        "axial_length" not in data["dimensions"]
        or data["dimensions"]["axial_length"] is None
    ):
        result["dimensions"]["axial_length"] = None
    for path in FIELD_SPECS:
        value = _lookup(result, path)
        if value is None and (path in ANGLE_PATHS or path == "dimensions.axial_length"):
            continue
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not np.isfinite(value)
        ):
            raise ValueError(f"{FIELD_SPECS[path]['label']} must be a finite number.")
    result["dimensions"] = asdict(MeanlineDimensions(**result["dimensions"]))
    result["hub"] = asdict(WallShape(**result["hub"]))
    result["shroud"] = asdict(WallShape(**result["shroud"]))
    result["blade"] = asdict(BladeEdges(**result["blade"]))
    if not 0 <= result["splitter_start"] < 1:
        raise ValueError("Splitter start must be in [0, 1).")
    return result


def parse_yaml(text):
    if not isinstance(text, str) or len(text.encode("utf-8")) > 1_000_000:
        raise ValueError("Choose a YAML configuration smaller than 1 MB.")
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as error:
        raise ValueError(f"Could not read YAML: {error}") from error
    return normalize_config(data)


def dump_yaml(config):
    return yaml.safe_dump(normalize_config(config), sort_keys=False, allow_unicode=True)


def preset_config(name="baseline"):
    filenames = {
        "baseline": "baseline.yaml",
        "inclined": "inclined_swept.yaml",
        "romei": "romei_inspired.yaml",
    }
    if name not in filenames:
        raise ValueError(f"Unknown preset: {name}")
    return parse_yaml(
        (STUDY / "examples" / filenames[name]).read_text(encoding="utf-8")
    )


def display_value(config, path):
    value = _lookup(config, path)
    if value is None and path in ANGLE_PATHS:
        end = "inlet" if "inlet_angle" in path else "outlet"
        value = config["dimensions"][f"{end}_flow_angle_deg"]
    if value is None:
        return None
    return value * FIELD_SPECS[path]["scale"]


def set_display_value(config, path, value):
    if path not in FIELD_SPECS:
        raise ValueError(f"Unknown parameter: {path}")
    result = deepcopy(config)
    # Preserve an empty draft so the app can display validation feedback.
    _assign(result, path, None if value is None else value / FIELD_SPECS[path]["scale"])
    return result


def is_linked(config, path):
    return path in ANGLE_PATHS and _lookup(config, path) is None


def set_link(config, path, linked):
    if path not in ANGLE_PATHS:
        raise ValueError("Only wall endpoint angles can follow a station direction.")
    result = deepcopy(config)
    _assign(result, path, None if linked else display_value(config, path))
    return result


def build_geometry(config):
    config = normalize_config(config)
    channel = MeridionalChannel(
        MeanlineDimensions(**config["dimensions"]),
        hub=WallShape(**config["hub"]),
        shroud=WallShape(**config["shroud"]),
    )
    blade = BladeMeridional(channel, BladeEdges(**config["blade"]))
    return config, channel, blade, blade.splitter(config["splitter_start"])


def evaluate_config(config, reference=None):
    """Atomically derive plots and metrics; failure carries no successful snapshot."""
    from .plots import build_figures

    try:
        canonical, channel, blade, splitter = build_geometry(config)
        reference_geometry = (
            None if reference is None else build_geometry(reference)[1:]
        )
        figures = build_figures(channel, blade, splitter, reference=reference_geometry)
        d = channel.diagnostics(201)
        metrics = [
            {
                "label": "Inlet area",
                "value": f"{channel.dimensions.inlet_area * 1e6:,.0f} mm²",
                "detail": "Nominal geometric station",
            },
            {
                "label": "Outlet area",
                "value": f"{channel.dimensions.outlet_area * 1e6:,.0f} mm²",
                "detail": "Normal to outlet direction",
            },
            {
                "label": "Minimum connector",
                "value": f"{d['gap'].min() * 1000:.2f} mm",
                "detail": "Sampled hub–shroud connector",
            },
            {
                "label": "Wall lengths · hub / shroud",
                "value": f"{channel.hub.length * 1000:.1f} / {channel.shroud.length * 1000:.1f} mm",
                "detail": "Physical meridional arclength",
            },
        ]
        message = "Geometry checks passed. Station areas exclude blade blockage."
        if any(
            canonical["blade"][key] != value
            for key, value in (
                ("leading_hub", 0),
                ("leading_shroud", 0),
                ("trailing_hub", 1),
                ("trailing_shroud", 1),
                ("leading_bow", 0),
                ("trailing_bow", 0),
            )
        ):
            message += " Blade edges are trimmed inside the nominal meanline stations."
        return {
            "valid": True,
            "config": canonical,
            "figures": figures,
            "metrics": metrics,
            "message": message,
        }
    except (ValueError, TypeError, OverflowError) as error:
        return {"valid": False, "error": str(error)}
