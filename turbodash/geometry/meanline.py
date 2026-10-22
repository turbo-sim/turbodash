"""Meanline baselines and one immutable, serializable blade-modeller state.

Geometry is a downstream product of an accepted meanline result. No function
in this module calls the solver or changes its thermodynamic result.
"""
from copy import deepcopy
from functools import lru_cache
import hashlib
import json

import numpy as np

from turbodash.core.turbine import get_blade_row_geometry
from .blade_parametrization.geometry import build_section
from .blade_parametrization.parameters import ParameterState, resolve_parameters, state_from_dict


# Rounded shape endpoints informed by the development reaction and symmetric
# impulse examples. Dimensions and angles are always supplied by meanline.
REACTION_PRESET = {
    "camberline.turning_concentration": 1.2,
    "camberline.turning_location": .52,
    "thickness.maximum_thickness_location": .3,
    "thickness.leading_edge_fullness": 0.,
    "thickness.trailing_edge_fullness": 0.,
    "thickness.leading_edge_fraction": .1,
    "thickness.trailing_edge_fraction": .025,
    "thickness.leading_edge_rounding_exponent": 3.,
    "thickness.trailing_edge_rounding_exponent": 3.,
    "thickness.shape_exponent": 2.,
    "leading_edge.relief_strength": .0,
    "trailing_edge.relief_strength": .0,
}
IMPULSE_PRESET = {
    **REACTION_PRESET,
    "camberline.turning_concentration": 1.8753,
    "camberline.turning_location": .5,
    "thickness.maximum_thickness_location": .5,
    "thickness.leading_edge_fraction": .005,
    "thickness.trailing_edge_fraction": .005,
}


def classify_blade(beta_in, beta_out, *, outlet_tolerance=1e-8):
    """Physical signed-degree z=-beta_in/beta_out and clipped preset weight.

    Loss models use this same ratio in their ordinary domain, but continue the
    denominator below 10 degrees for optimizer trial points. That continuation
    is not a physical blade classification. Undefined/nonfinite inputs or an
    outlet within 1e-8 degree of zero explicitly select the reaction preset.
    """
    incoming, outgoing = float(beta_in), float(beta_out)
    if not np.isfinite([incoming, outgoing]).all() or abs(outgoing) <= outlet_tolerance:
        return {"z": None, "weight": 0., "fallback": "Undefined outlet-angle ratio; reaction shape preset used."}
    ratio = -incoming / outgoing
    if not np.isfinite(ratio):
        return {"z": None, "weight": 0., "fallback": "Nonfinite angle ratio; reaction shape preset used."}
    return {"z": ratio, "weight": float(np.clip(ratio, 0., 1.)), "fallback": None}


def interpolated_presets(beta_in, beta_out):
    classification = classify_blade(beta_in, beta_out)
    weight = classification["weight"]
    source = {}
    for path, value in REACTION_PRESET.items():
        group, field = path.split(".")
        source.setdefault(group, {})[field] = (1-weight)*value + weight*IMPULSE_PRESET[path]
    return source, classification


def parameters_from_meanline(geometry, turbine_type="axial", *, label="Meanline blade row"):
    """Map SI meanline dimensions and signed row-frame metal angles.

    Thickness ratios use *meridional* chord (not the loss-model true chord).
    A meanline LE radius becomes a rounded-tip limiting-radius target; the
    meanline finite TE thickness supplies the unrounded shoulder thickness.
    The rounded family closes the TE and does not reproduce a finite blunt tip.
    """
    if turbine_type not in ("axial", "radial"):
        raise ValueError("Blade modelling supports axial cylinders and radial planes only.")
    incoming, outgoing = (float(geometry[key]) for key in ("metal_angle_in", "metal_angle_out"))
    source, classification = interpolated_presets(incoming, outgoing)
    rin, rout = (float(geometry[key]) for key in ("radius_in", "radius_out"))
    if not np.isfinite([rin, rout]).all() or min(rin, rout) <= 0:
        raise ValueError("Meanline inlet and outlet radii must be finite and positive.")
    if turbine_type == "radial" and rin == rout:
        raise ValueError("Radial inlet and outlet radii must differ.")
    if not np.isfinite([incoming, outgoing]).all() or max(abs(incoming), abs(outgoing)) >= 90:
        raise ValueError("Meanline metal angles must be finite and strictly between -90 and 90 degrees.")
    chord = float(geometry["chord_meridional"])
    if not np.isfinite(chord) or chord <= 0:
        raise ValueError("Meanline meridional chord must be finite and positive.")
    source["geometry"] = {"type": turbine_type, "mean_radius": .5*(rin+rout),
                          "blade_height": float(geometry["height"]), "meridional_chord": chord,
                          "inlet_radius": rin, "outlet_radius": rout}
    count = geometry.get("blade_count_integer")
    if count is None and "blade_count_continuous" not in geometry:
        count = geometry["blade_count"]
    if count is not None and (isinstance(count, (bool, np.bool_)) or not np.isfinite(count)
                              or float(count) != int(count) or count < 2):
        raise ValueError("Blade count must be an integer of at least two.")
    source["cascade"] = ({"blade_count": int(count)} if count is not None
                         else {"blade_count_continuous": float(geometry["blade_count_continuous"])})
    source["camberline"].update(inlet_metal_angle_deg=incoming, outlet_metal_angle_deg=outgoing,
                                interpolation="angle")
    thickness = source["thickness"]
    thickness["maximum_thickness_ratio"] = float(geometry["maximum_thickness"]) / chord
    if "maximum_thickness_location" in geometry:
        thickness["maximum_thickness_location"] = float(geometry["maximum_thickness_location"])
    thickness["trailing_edge_thickness_ratio"] = float(geometry["trailing_edge_thickness"]) / chord
    # Local dimensional scale for an uncorrected conformal offset. This is the
    # cylinder chord axially and r_in*|log(r_out/r_in)| for radial geometry.
    scale = chord if turbine_type == "axial" else rin * abs(np.log(rout/rin))
    fraction = thickness["leading_edge_fraction"]
    exponent = thickness["leading_edge_rounding_exponent"]
    radius = float(geometry.get("leading_edge_radius", 0.))
    if radius < 0 or not np.isfinite(radius):
        raise ValueError("Meanline leading-edge radius must be finite and nonnegative.")
    thickness["leading_edge_thickness_ratio"] = float(np.sqrt(
        8*radius*fraction/(scale*exponent*np.cos(np.deg2rad(incoming)))))
    source["correction"] = {"preserve_physical_thickness": False}
    source["numerics"] = {"number_of_points": 601}
    warnings = []
    if classification["fallback"]:
        warnings.append(classification["fallback"])
    if float(geometry["trailing_edge_thickness"]) > 0:
        warnings.append("The meanline trailing-edge thickness sets the unrounded shoulder scale; the rounded blade tip closes to zero width.")
    if turbine_type == "radial":
        warnings.append("Radial thickness ratios refer to conformal chord; physical thickness varies with radius unless correction is enabled. Leading radius is a local-scale target, not an exact mapped tip radius.")
    if abs(float(geometry.get("flaring_angle", 0.))) > 1e-8 or (turbine_type == "axial" and not np.isclose(rin, rout)):
        warnings.append("The blade model uses mean radius and mean height with parallel endwalls; meanline flaring and free-vortex span twist are not represented.")
    return ParameterState(source, label=label), classification, warnings


def state_for_stage_row(stage, row, turbine_type="axial"):
    """Read a projected effective state or construct the row's fresh baseline."""
    saved = stage.get("geometry_parameters", {}).get(row)
    if saved is not None:
        return state_from_dict(saved)
    geometry = get_blade_row_geometry(stage, row, turbine_type=turbine_type, discrete=True)
    return parameters_from_meanline(geometry, turbine_type, label=f"Meanline {row}")[0]


def baseline_id(results):
    # Thermodynamic metadata need not invalidate geometry; the accepted result
    # callback still explicitly initializes a fresh baseline on every success.
    content = {"type": results["inputs"]["turbine_type"], "rows": [
        [get_blade_row_geometry(stage, row, turbine_type=results["inputs"]["turbine_type"], discrete=True)
         for row in ("stator", "rotor")] for stage in results["stages_performance"]]}
    payload = json.dumps(content, sort_keys=True, default=lambda item: np.asarray(item).tolist(), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def initialize_geometry(results):
    """Create a fresh baseline with empty overrides for every accepted row."""
    if not results or not results.get("stages_performance"):
        raise ValueError("A successful meanline result is required to initialize geometry.")
    turbine_type = results["inputs"]["turbine_type"]
    rows = []
    for index, stage in enumerate(results["stages_performance"]):
        for row in ("stator", "rotor"):
            label = f"Stage {index+1} {row}"
            geometry = get_blade_row_geometry(stage, row, turbine_type=turbine_type, discrete=True)
            parameters, classification, warnings = parameters_from_meanline(geometry, turbine_type, label=label)
            rows.append({"id": f"stage_{index+1:02d}/{row}", "stage_index": index, "row": row,
                         "label": label, "parameters": parameters.to_dict(),
                         "classification": classification, "warnings": warnings})
    return {"schema_version": 1, "baseline_id": baseline_id(results), "revision": 0, "rows": rows}


def _row_record(state, row_id):
    for record in state["rows"]:
        if record["id"] == row_id:
            return record
    raise ValueError(f"Unknown geometry row: {row_id!r}.")


def row_parameter_state(state, row_id):
    return state_from_dict(_row_record(state, row_id)["parameters"])


def update_geometry_override(state, row_id, path, value):
    parameters = row_parameter_state(state, row_id).with_override(path, value)
    updated = deepcopy(state)
    _row_record(updated, row_id)["parameters"] = parameters.to_dict()
    updated["revision"] += 1
    return updated


def reset_geometry_overrides(state, row_id, path=None):
    parameters = row_parameter_state(state, row_id)
    parameters = parameters.reset(path) if path else ParameterState(parameters.source, label=parameters.label)
    updated = deepcopy(state)
    _row_record(updated, row_id)["parameters"] = parameters.to_dict()
    updated["revision"] += 1
    return updated


@lru_cache(maxsize=64)
def _cached_section(payload, check_intersections):
    return build_section(json.loads(payload), check_intersections=check_intersections)


def section_from_state(parameters, *, check_intersections=False):
    values = resolve_parameters(parameters).values
    return _cached_section(json.dumps(values, sort_keys=True), bool(check_intersections))


def get_row_section(state, row_id, *, check_intersections=False):
    return section_from_state(row_parameter_state(state, row_id), check_intersections=check_intersections)


def effective_results(results, geometry_state):
    """Project effective geometry onto a copy; reject a stale source/result pair.

    Meanline geometry and all performance fields remain untouched. Downstream
    consumers explicitly opt into ``geometry_parameters``; this does not feed
    changes back into losses or the solver.
    """
    if geometry_state is None:
        geometry_state = initialize_geometry(results)
    if geometry_state.get("baseline_id") != baseline_id(results):
        raise ValueError("Geometry belongs to a different meanline result; wait for its new baseline.")
    projected = deepcopy(results)
    for record in geometry_state["rows"]:
        state = state_from_dict(record["parameters"])
        projected["stages_performance"][record["stage_index"]].setdefault("geometry_parameters", {})[record["row"]] = state.to_dict()
    return projected


def physical_row_geometry(stage, row, turbine_type="axial"):
    """Independent geometric metadata for CAD/exports, retaining solver records."""
    original = get_blade_row_geometry(stage, row, turbine_type=turbine_type, discrete=True)
    values = resolve_parameters(state_for_stage_row(stage, row, turbine_type)).values
    geom, thickness = values["geometry"], values["thickness"]
    if geom["type"] != turbine_type:
        raise ValueError("Changing a row's stream-surface type is not supported by the turbine assembly/export; use the standalone row preview.")
    chord = geom["meridional_chord"] if turbine_type == "axial" else abs(geom["outlet_radius"]-geom["inlet_radius"])
    radius_in = geom["mean_radius"] if turbine_type == "axial" else geom["inlet_radius"]
    radius_out = geom["mean_radius"] if turbine_type == "axial" else geom["outlet_radius"]
    count = values["cascade"]["blade_count"]
    pitch = np.pi*(radius_in+radius_out)/count
    pitch_out = 2*np.pi*radius_out/count
    angles = values["camberline"]
    beta_in, beta_out = np.deg2rad([angles["inlet_metal_angle_deg"], angles["outlet_metal_angle_deg"]])
    opening = pitch_out*np.cos(abs(beta_out)-np.pi/count if turbine_type == "radial" else beta_out)
    section = section_from_state(ParameterState(values))
    sweep = section.camber[-1, 1]-section.camber[0, 1]
    true_chord = (np.hypot(chord, radius_in*sweep) if turbine_type == "axial" else
                  np.sqrt(radius_in**2+radius_out**2-2*radius_in*radius_out*np.cos(sweep)))
    scale = chord if turbine_type == "axial" else radius_in*abs(np.log(radius_out/radius_in))
    leading_fraction = thickness["leading_edge_fraction"]
    leading_radius = (scale*thickness["leading_edge_thickness_ratio"]**2
                      * thickness["leading_edge_rounding_exponent"]*np.cos(beta_in)/(8*leading_fraction)
                      if leading_fraction > 0 else 0.)
    area_out = 2*np.pi*radius_out*geom["blade_height"]
    return {**original, "blade_count": count, "blade_count_integer": count,
            "chord_meridional": chord, "meridional_chord": chord,
            "chord": float(true_chord), "aspect_ratio": geom["blade_height"]/chord,
            "height": geom["blade_height"], "radius_in": radius_in, "radius_out": radius_out,
            "spacing": pitch, "pitch": pitch, "opening": opening,
            "solidity": chord/pitch, "solidity_meridional": chord/pitch,
            "solidity_chord": true_chord/pitch, "pitch_to_chord": pitch/true_chord,
            "A_out": area_out, "A_throat": area_out*opening/pitch_out,
            "leading_edge_radius": float(leading_radius), "leading_edge_diameter": 2*float(leading_radius),
            "flaring_angle": 0.,
            "maximum_thickness": thickness["maximum_thickness_ratio"]*chord,
            "maximum_thickness_location": thickness["maximum_thickness_location"],
            "trailing_edge_thickness": thickness["trailing_edge_thickness_ratio"]*chord,
            "metal_angle_in": values["camberline"]["inlet_metal_angle_deg"],
            "metal_angle_out": values["camberline"]["outlet_metal_angle_deg"]}
