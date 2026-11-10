"""Engineering controls for the impeller's existing SI configuration.

Slider ranges are exploration aids; typed inputs keep their full precision and
are validated by the geometry kernel. This module owns display conversion only.
"""

from __future__ import annotations

import json
import math

from dash import dcc, html

FIELD_SPECS = {}


def _field(
    path, label, group, low, high, step, *, unit="", scale=1.0, kind="float", help=""
):
    FIELD_SPECS[path] = dict(
        label=label,
        group=group,
        min=low,
        max=high,
        step=step,
        unit=unit,
        scale=scale,
        kind=kind,
        help=help,
    )


for wall, index in (("Hub", 0), ("Shroud", -1)):
    prefix = f"sections.{index}"
    _field(
        f"{prefix}.camberline.inlet_metal_angle_deg",
        "Inlet metal angle",
        f"{wall} angles",
        -80,
        20,
        0.25,
        unit="°",
    )
    _field(
        f"{prefix}.camberline.outlet_metal_angle_deg",
        "Outlet metal angle",
        f"{wall} angles",
        -80,
        20,
        0.25,
        unit="°",
    )
    _field(
        f"{prefix}.camberline.turning_location",
        "Half-turning location",
        f"{wall} angles",
        0.05,
        0.95,
        0.01,
        help="Location along the normalized conformal coordinate u.",
    )
    _field(
        f"{prefix}.camberline.turning_concentration",
        "Turning concentration",
        f"{wall} angles",
        1,
        5,
        0.05,
    )
    for name, label, low, high, step, unit, scale in (
        ("maximum_thickness_m", "Maximum thickness", 0.1, 15, 0.05, "mm", 1000),
        ("maximum_thickness_location", "Maximum location", 0.05, 0.95, 0.01, "", 1),
        (
            "leading_edge_thickness_m",
            "LE shoulder thickness",
            0.05,
            10,
            0.05,
            "mm",
            1000,
        ),
        (
            "trailing_edge_thickness_m",
            "TE shoulder thickness",
            0.05,
            10,
            0.05,
            "mm",
            1000,
        ),
        ("leading_edge_fraction", "LE rounding fraction", 0.01, 0.3, 0.005, "", 1),
        ("trailing_edge_fraction", "TE rounding fraction", 0.01, 0.3, 0.005, "", 1),
        ("leading_edge_rounding_exponent", "LE rounding exponent", 2, 8, 0.1, "", 1),
        ("trailing_edge_rounding_exponent", "TE rounding exponent", 2, 8, 0.1, "", 1),
        ("leading_edge_fullness", "LE shoulder fullness", -1, 1, 0.05, "", 1),
        ("trailing_edge_fullness", "TE shoulder fullness", -1, 1, 0.05, "", 1),
        ("shape_exponent", "Body shape exponent", 1, 6, 0.1, "", 1),
    ):
        _field(
            f"{prefix}.thickness.{name}",
            label,
            f"{wall} thickness",
            low,
            high,
            step,
            unit=unit,
            scale=scale,
        )

for name, label, low, high in (
    ("inlet_hub_radius", "Inlet hub radius", 10, 250),
    ("inlet_shroud_radius", "Inlet shroud radius", 20, 400),
    ("outlet_radius", "Outlet radius", 30, 600),
    ("outlet_width", "Outlet passage width", 1, 100),
    ("axial_length", "Outlet axial centre", 10, 300),
):
    _field(
        f"dimensions.{name}",
        label,
        "Channel dimensions",
        low,
        high,
        0.25,
        unit="mm",
        scale=1000,
    )
_field(
    "dimensions.inlet_flow_angle_deg",
    "Inlet station direction",
    "Channel dimensions",
    -45,
    45,
    0.5,
    unit="°",
)
_field(
    "dimensions.outlet_flow_angle_deg",
    "Outlet station direction",
    "Channel dimensions",
    30,
    90,
    0.5,
    unit="°",
)

for wall in ("hub", "shroud"):
    group = f"{wall.title()} wall"
    _field(
        f"{wall}.inlet_angle_deg", "Inlet tangent angle", group, -45, 75, 0.5, unit="°"
    )
    _field(
        f"{wall}.outlet_angle_deg",
        "Outlet tangent angle",
        group,
        10,
        120,
        0.5,
        unit="°",
    )
    _field(
        f"{wall}.inlet_tangent_fraction", "Inlet tangent reach", group, 0.05, 1, 0.01
    )
    _field(
        f"{wall}.outlet_tangent_fraction", "Outlet tangent reach", group, 0.05, 1, 0.01
    )
    _field(f"{wall}.bow", "Interior bow", group, -0.25, 0.25, 0.005)

for name, label, low, high, help_text in (
    (
        "nose_length_m",
        "Rounded nose length",
        0,
        150,
        "Axial hub extension upstream of the blade roots. Zero uses a flat front; a rounded nose requires a solid centre.",
    ),
    (
        "back_thickness_m",
        "Disk back thickness",
        1,
        100,
        "Thickness behind the hub outlet. Included in the 3D view and the full STL.",
    ),
    (
        "bore_radius_m",
        "Shaft bore radius (0 = solid)",
        0,
        100,
        "Zero closes the centre. Set nose length to zero before adding a shaft bore.",
    ),
):
    _field(
        f"disk.{name}",
        label,
        "Hub nose and disk",
        low,
        high,
        0.25,
        unit="mm",
        scale=1000,
        help=help_text,
    )

for name, label, low, high in (
    ("leading_hub", "Hub leading-edge station", 0, 0.4),
    ("leading_shroud", "Shroud leading-edge station", 0, 0.4),
    ("trailing_hub", "Hub trailing-edge station", 0.6, 1),
    ("trailing_shroud", "Shroud trailing-edge station", 0.6, 1),
    ("leading_bow", "Leading-edge bow", -0.2, 0.2),
    ("trailing_bow", "Trailing-edge bow", -0.2, 0.2),
):
    _field(f"blade_edges.{name}", label, "Blade footprint", low, high, 0.005)
_field("blade_count", "Main blade count", "Stacking and row", 5, 40, 1, kind="int")
_field("rake.angle_deg", "Exit rake", "Stacking and row", -30, 30, 0.5, unit="°")
_field(
    "sections.0.stacking_offset_deg",
    "Hub LE azimuth",
    "Stacking and row",
    -60,
    60,
    0.5,
    unit="°",
    help="In trailing-edge mode this fixes the hub reference only; LE lean is an outcome.",
)
_field(
    "sections.-1.stacking_offset_deg",
    "Shroud LE azimuth",
    "Stacking and row",
    -60,
    60,
    0.5,
    unit="°",
    help="Used by free/preserved-LE modes. Trailing-edge stacking derives this position.",
)

# Dash pattern-matched outputs follow layout order. Match the sidebar's first
# stacking group, then the remaining groups, so batch synchronization is exact.
FIELD_SPECS = {
    **{
        path: spec
        for path, spec in FIELD_SPECS.items()
        if spec["group"] == "Stacking and row"
    },
    **{
        path: spec
        for path, spec in FIELD_SPECS.items()
        if spec["group"] != "Stacking and row"
    },
}
FIELD_PATHS = tuple(FIELD_SPECS)
GROUPS = tuple(dict.fromkeys(spec["group"] for spec in FIELD_SPECS.values()))


def get_value(config, path):
    value = config
    for key in path.split("."):
        value = value[int(key)] if isinstance(value, list) else value[key]
    if value is None and path in {
        "hub.inlet_angle_deg",
        "shroud.inlet_angle_deg",
        "hub.outlet_angle_deg",
        "shroud.outlet_angle_deg",
    }:
        end = "inlet" if ".inlet_" in path else "outlet"
        value = config["dimensions"][f"{end}_flow_angle_deg"]
    return value


def to_display(path, value):
    return None if value is None else value * FIELD_SPECS[path]["scale"]


def from_display(path, value):
    if path not in FIELD_SPECS:
        raise ValueError(f"Unknown impeller control: {path}")
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
    ):
        raise ValueError(f"{FIELD_SPECS[path]['label']} needs a finite number.")
    value /= FIELD_SPECS[path]["scale"]
    if FIELD_SPECS[path]["kind"] == "int":
        if not value.is_integer():
            raise ValueError(f"{FIELD_SPECS[path]['label']} must be an integer.")
        return int(value)
    return value


def field_values(config):
    return [to_display(path, get_value(config, path)) for path in FIELD_PATHS]


def control_view(config):
    values = field_values(config)
    mode = config["rake"]["mode"]
    disabled = []
    for path in FIELD_PATHS:
        disabled.append(
            (path == "rake.angle_deg" and mode == "free")
            or (path == "sections.-1.stacking_offset_deg" and mode == "trailing_edge")
            or ("camberline.turning_location" in path and mode == "preserve_le")
            or (path == "disk.bore_radius_m" and config["disk"]["nose_length_m"] > 0)
        )
    return dict(
        numbers=values,
        sliders=values,
        disabled=disabled,
        minimums=[min(FIELD_SPECS[p]["min"], v) for p, v in zip(FIELD_PATHS, values)],
        maximums=[max(FIELD_SPECS[p]["max"], v) for p, v in zip(FIELD_PATHS, values)],
    )


def field(path, config):
    spec = FIELD_SPECS[path]
    value = to_display(path, get_value(config, path))
    identifier = {"type": "impeller-number", "path": path}
    label = spec["label"] + (f" [{spec['unit']}]" if spec["unit"] else "")
    disabled = control_view(config)["disabled"][FIELD_PATHS.index(path)]
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        html.Label(
                            label,
                            htmlFor=json.dumps(
                                identifier, sort_keys=True, separators=(",", ":")
                            ),
                            title=spec["help"],
                        ),
                        className="td-blade-label",
                    ),
                    dcc.Input(
                        id=identifier,
                        type="number",
                        value=value,
                        step=1 if spec["kind"] == "int" else "any",
                        debounce=True,
                        disabled=disabled,
                        persistence=False,
                    ),
                ],
                className="td-blade-input-line",
            ),
            dcc.Slider(
                id={"type": "impeller-slider", "path": path},
                min=min(spec["min"], value),
                max=max(spec["max"], value),
                value=value,
                step=spec["step"],
                marks=None,
                updatemode="mouseup",
                disabled=disabled,
                className="td-blade-slider",
                tooltip={"placement": "bottom", "always_visible": False},
            ),
        ],
        className="td-blade-field",
        **{"data-parameter": path},
    )


def group(name, config, *, opened=False):
    return html.Details(
        [
            html.Summary(name),
            *[
                field(path, config)
                for path in FIELD_PATHS
                if FIELD_SPECS[path]["group"] == name
            ],
        ],
        open=opened,
        className="td-row-settings td-blade-group",
    )


def sidebar(config):
    return html.Aside(
        [
            html.Div(
                [
                    html.Div("GEOMETRY INPUTS", className="impeller-eyebrow"),
                    html.H2("Engineering controls"),
                    html.P(
                        "Drag a slider or type an exact value. Changes rebuild the geometry after release.",
                        className="td-blade-caption",
                    ),
                ],
                className="impeller-sidebar-heading",
            ),
            html.Div(
                [
                    dcc.Upload(
                        id="impeller-yaml-upload",
                        children=html.Button("Load YAML", type="button"),
                        multiple=False,
                        accept=".yaml,.yml",
                    ),
                    html.Button(
                        "Save YAML", id="impeller-yaml-download-button", type="button"
                    ),
                    html.Button("Reset example", id="impeller-reset", type="button"),
                ],
                className="td-blade-actions impeller-file-actions",
            ),
            html.Details(
                [
                    html.Summary("Stacking and row"),
                    html.Div(
                        [
                            html.Label(
                                "Rake construction", htmlFor="impeller-rake-mode"
                            ),
                            dcc.Dropdown(
                                id="impeller-rake-mode",
                                options=[
                                    {
                                        "label": "Set rake; allow LE lean",
                                        "value": "trailing_edge",
                                    },
                                    {
                                        "label": "Free rake; set LE stacking",
                                        "value": "free",
                                    },
                                    {
                                        "label": "Set rake; preserve LE",
                                        "value": "preserve_le",
                                    },
                                ],
                                value=config["rake"]["mode"],
                                clearable=False,
                                searchable=False,
                            ),
                            html.P(
                                "Zero rake aligns the trailing edges. Leading-edge lean follows the hub and shroud turning.",
                                id="impeller-stacking-help",
                                className="td-blade-caption",
                            ),
                        ],
                        className="impeller-select-field",
                    ),
                    *[
                        field(path, config)
                        for path in FIELD_PATHS
                        if FIELD_SPECS[path]["group"] == "Stacking and row"
                    ],
                ],
                open=True,
                className="td-row-settings td-blade-group",
            ),
            html.Div(
                [
                    html.Label(
                        "Turning interpolation", htmlFor="impeller-angle-interpolation"
                    ),
                    dcc.Dropdown(
                        id="impeller-angle-interpolation",
                        options=[
                            {"label": "Metal angle", "value": "angle"},
                            {"label": "Tangent of angle", "value": "slope"},
                        ],
                        value=config["sections"][0]["camberline"]["interpolation"],
                        clearable=False,
                        searchable=False,
                        disabled=config["rake"]["mode"] == "preserve_le",
                    ),
                ],
                className="impeller-select-field impeller-interpolation",
            ),
            *[
                group(name, config, opened=name in {"Hub angles", "Shroud angles"})
                for name in GROUPS
                if name != "Stacking and row"
            ],
            html.P(
                "All section lengths are displayed in mm. Saved geometry uses metres. The hub and shroud curves share the same blade model as the scripts.",
                className="td-blade-caption",
            ),
        ],
        className="td-sidebar td-blade-modeller impeller-sidebar",
    )
