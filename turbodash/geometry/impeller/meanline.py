"""Convert a compressor meanline geometry into an editable impeller configuration.

This adapter consumes a plain output mapping. It does not import the thesis
meanline library, run a solver, or claim that station dimensions determine the
shape between stations. Lengths are metres and angles are degrees.
"""

from collections.abc import Mapping
from copy import deepcopy
import json
import math


_REQUIRED = (
    "D_1_h", "D_1_s", "D_2", "b_2", "L_z", "beta_1b_h", "beta_1b_s",
    "beta_2b", "t_b_1", "t_b_2", "Z_r_full",
)


def from_meanline_geometry(geometry, *, provenance=None):
    """Return the approved shrouded, main-blade-only starting configuration.

    ``geometry`` is the compressor's completed geometry dictionary from the
    reversible meanline result. Noninteger full-blade counts and nonzero splitter
    counts are rejected, rather than silently rounded or dropped.

    The meanline's ``L_z`` locates the hub outlet. Here ``axial_length`` locates
    the outlet centre, so the conversion is ``L_z - b_2 / 2``. The inlet and
    outlet are taken to be axial and radial respectively.

    The meanline's two thicknesses seed the *unrounded body* edge scales. Both
    rounded tips have zero paired thickness. A maximum body thickness of twice
    the larger edge scale is an editable geometric assumption, as are the wall,
    turning and rounding controls. These are recorded in ``source``.

    The first implementation is explicitly shrouded, as requested for the
    geometry study, even when the source meanline is open. Its raw clearance
    and shrouding fields are retained in ``source``; performance is not rerun.
    """
    if not isinstance(geometry, Mapping):
        raise TypeError("geometry must be a meanline geometry mapping.")
    missing = [key for key in _REQUIRED if key not in geometry]
    if missing:
        raise ValueError("Missing meanline geometry fields: " + ", ".join(missing))
    values = {}
    for key in _REQUIRED:
        value = geometry[key]
        if isinstance(value, bool):
            raise ValueError(f"{key} must be a finite number, not a boolean.")
        try:
            values[key] = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{key} must be a finite number.") from error
        if not math.isfinite(values[key]):
            raise ValueError(f"{key} must be a finite number.")
    for key in ("D_1_h", "D_1_s", "D_2", "b_2", "L_z", "t_b_1", "t_b_2"):
        if values[key] <= 0:
            raise ValueError(f"{key} must be positive.")
    if not values["D_1_h"] < values["D_1_s"] < values["D_2"]:
        raise ValueError("Require D_1_h < D_1_s < D_2.")
    if values["L_z"] <= values["b_2"]:
        raise ValueError("L_z must exceed b_2 so the shroud outlet is downstream.")
    for key in ("beta_1b_h", "beta_1b_s", "beta_2b"):
        if not -90 < values[key] < 90:
            raise ValueError(f"{key} must lie strictly between -90 and 90 degrees.")
    count = values["Z_r_full"]
    if count < 2 or not count.is_integer():
        raise ValueError("Z_r_full must be an integer >= 2 for this impeller model.")
    if float(geometry.get("Z_r_split", 0)) != 0:
        raise ValueError("Splitters are outside this adapter's main-blade-only scope.")
    if provenance is not None and not isinstance(provenance, Mapping):
        raise TypeError("provenance must be a JSON-safe mapping.")
    source = deepcopy(dict(provenance or {}))
    json.dumps(source, allow_nan=False)
    original = dict(values)
    for key in ("beta_1b", "eps_a", "eps_r", "eps_b", "Z_r_split"):
        if key in geometry:
            value = float(geometry[key])
            if not math.isfinite(value):
                raise ValueError(f"{key} must be finite when supplied.")
            original[key] = value
    if "is_shrouded" in geometry:
        original["is_shrouded"] = bool(geometry["is_shrouded"])
    source["meanline_geometry"] = original
    source["geometry_overrides"] = {
        "shrouded": True,
        "modeled_tip_gap_m": 0.0,
        "performance_recomputed": False,
        "reason": "Approved geometry brief: shrouded impeller with no modeled tip gap.",
    }
    source["shape_assumptions"] = {
        "walls": "Independent polynomial walls; axial inlet, radial outlet; tangent fractions 0.45, zero bow.",
        "blade_footprint": "Both edges at the channel endpoints; zero edge bow.",
        "maximum_thickness": "2 * max(t_b_1, t_b_2); a geometric starting choice, not a meanline output.",
        "edge_thickness": "t_b_1 and t_b_2 seed unrounded body scales; the rounded tips have zero paired thickness.",
        "spanwise_thickness": "Initially equal hub/shroud inputs; independently editable.",
        "angle_distribution": "Production monotone turning law; location 0.5, concentration 1.2.",
        "inlet_interpolation": "Linear interpolation of tangent of inlet angle against actual leading-edge radius.",
        "stacking": "Aligned leading edges, zero circumferential offsets.",
    }
    wall = {
        "inlet_angle_deg": 0.0,
        "outlet_angle_deg": 90.0,
        "inlet_tangent_fraction": 0.45,
        "outlet_tangent_fraction": 0.45,
        "bow": 0.0,
    }
    thickness = {
        "maximum_thickness_m": 2 * max(values["t_b_1"], values["t_b_2"]),
        "leading_edge_thickness_m": values["t_b_1"],
        "trailing_edge_thickness_m": values["t_b_2"],
        "maximum_thickness_location": 0.3,
        "leading_edge_fraction": 0.08,
        "trailing_edge_fraction": 0.08,
        "leading_edge_rounding_exponent": 3.0,
        "trailing_edge_rounding_exponent": 3.0,
        "shape_exponent": 2.0,
        "leading_edge_fullness": 0.0,
        "trailing_edge_fullness": 0.0,
    }
    sections = []
    for v, inlet_angle in ((0.0, values["beta_1b_h"]), (1.0, values["beta_1b_s"])):
        sections.append({
            "v": v,
            "camberline": {
                "inlet_metal_angle_deg": inlet_angle,
                "outlet_metal_angle_deg": values["beta_2b"],
                "interpolation": "angle",
                "turning_location": 0.5,
                "turning_concentration": 1.2,
            },
            "thickness": dict(thickness),
            "stacking_offset_deg": 0.0,
        })
    return {
        "schema_version": 1,
        "name": "meanline_impeller",
        "source": source,
        "dimensions": {
            "inlet_hub_radius": values["D_1_h"] / 2,
            "inlet_shroud_radius": values["D_1_s"] / 2,
            "outlet_radius": values["D_2"] / 2,
            "outlet_width": values["b_2"],
            "axial_length": values["L_z"] - values["b_2"] / 2,
            "inlet_flow_angle_deg": 0.0,
            "outlet_flow_angle_deg": 90.0,
        },
        "hub": dict(wall),
        "shroud": dict(wall),
        "control_point_displacements": {},
        "blade_edges": {
            "leading_hub": 0.0, "leading_shroud": 0.0,
            "trailing_hub": 1.0, "trailing_shroud": 1.0,
            "leading_bow": 0.0, "trailing_bow": 0.0,
        },
        "blade_count": int(count),
        "sections": sections,
        "inlet_interpolation": "tangent_radius",
        "construction": {"shrouded": True, "splitters": False},
        "numerics": {"streamwise_points": 161, "spanwise_points": 17, "integration_points": 2001},
    }
