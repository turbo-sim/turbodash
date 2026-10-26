"""Continuous meanline geometry between adjacent blade rows.

Axial spacings are fixed input ratios: gap length / downstream meridional
chord. Radial gaps follow the actual station-radius difference. These
geometric diagnostics do not introduce duct losses or change velocity triangles.
"""

from copy import deepcopy
from numbers import Real

import numpy as np

from .errors import ModelDomainError, model_context, require_positive


AXIAL_SPACING_FIELDS = ("axial_spacing_before_stator", "axial_spacing_before_rotor")
DEFAULT_AXIAL_SPACING = 0.3


def _positive(value, quantity):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{quantity} must be a finite positive number.")
    require_positive(value, quantity)
    return float(value)


def _row_geometry(stage, row):
    return stage.get("geometry_meanline", stage.get("geometry", {}))[row]


def _endpoints(stages, stage_index, row):
    if row == "rotor":
        upstream_index, upstream_row, upstream_station = stage_index, "stator", 1
        downstream_station = 2
    elif row == "stator" and stage_index > 0:
        upstream_index, upstream_row, upstream_station = stage_index - 1, "rotor", 3
        downstream_station = 0
    else:
        raise ValueError("An interspace exists before every rotor and every stator except the first.")
    upstream = stages[upstream_index]
    downstream = stages[stage_index]
    return upstream_index, upstream_row, upstream_station, downstream_station, upstream, downstream


def _compute_interspace(results, stage_index, row):
    stages = results["stages_performance"]
    up_index, up_row, up_station, down_station, upstream, downstream = _endpoints(
        stages, stage_index, row
    )
    inputs = results.get("inputs", {})
    turbine_type = inputs.get("turbine_type", "axial")
    axial = turbine_type == "axial"
    stage_inputs = inputs.get("stages", [])
    specification = stage_inputs[stage_index] if stage_index < len(stage_inputs) else {}
    field = f"axial_spacing_before_{row}"
    if axial:
        chord = _positive(_row_geometry(downstream, row)["chord_meridional"], "meridional_chord")
        if field in specification:
            ratio = _positive(specification[field], field)
            length = ratio * chord
            source = "prescribed_ratio"
        else:
            # Compatibility for old YAML and unannotated plotting results.
            length = 1.5 * _positive(_row_geometry(upstream, up_row)["opening"], "opening")
            ratio = length / chord
            source = "legacy_opening"
        require_positive(length, "interspace_meridional_length")
    else:
        ratio, source = None, "station_radii"
        length = abs(float(downstream["flow_stations"][down_station]["r"])
                     - float(upstream["flow_stations"][up_station]["r"]))

    record = dict(upstream_stage_index=up_index, upstream_row=up_row,
                  downstream_stage_index=stage_index, downstream_row=row,
                  meridional_length=float(length), spacing_source=source)
    if axial:
        record.update(axial_length=float(length), spacing_ratio=float(ratio))
    # Old geometry-only plot fixtures need the same resolved gap, but do not
    # supply station thermodynamics or heights.
    if "flow_stations" not in upstream or "flow_stations" not in downstream:
        return record
    inlet = upstream["flow_stations"][up_station]
    outlet = downstream["flow_stations"][down_station]
    r_in, r_out = float(inlet["r"]), float(outlet["r"])
    h_in, h_out = float(inlet["H"]), float(outlet["H"])
    for quantity, value in (("interspace_radius_in", r_in), ("interspace_radius_out", r_out),
                            ("interspace_height_in", h_in), ("interspace_height_out", h_out)):
        require_positive(value, quantity)
    if not np.isfinite(length):
        raise ModelDomainError("Interspace length must be finite.", code="nonfinite_quantity",
                               quantity="interspace_meridional_length", value=length)
    half_height_change = 0.5 * (h_out - h_in)
    # For axial flow mean-radius movement bends both walls. For radial flow
    # radius is the streamwise coordinate; the walls are at +/- half height.
    center_change = r_out - r_in if axial else 0.0
    def angle(change):
        return float(np.degrees(np.arctan2(change, length)))

    record.update(
        radius_in=r_in, radius_out=r_out, height_in=h_in, height_out=h_out,
        flaring_angle=angle(half_height_change),
        hub_wall_angle=angle(center_change - half_height_change),
        shroud_wall_angle=angle(center_change + half_height_change),
        degenerate_interspace=bool(length == 0.0 and h_out != h_in),
    )
    return record


def get_interspace_geometry(results, stage_index, row):
    """Read a stored gap, or resolve a legacy result without model evaluations."""
    stored = results["stages_performance"][stage_index].get("geometry_interspaces", {})
    key = f"before_{row}"
    if key in stored:
        return stored[key]
    return _compute_interspace(results, stage_index, row)


def add_interspace_geometry(results):
    """Attach every actual row-to-row passage to newly calculated results."""
    for index, stage in enumerate(results["stages_performance"]):
        with model_context(stage_index=index, stage_name=stage.get("name")):
            stage["geometry_interspaces"] = {
                f"before_{row}": _compute_interspace(results, index, row)
                for row in (("stator", "rotor") if index else ("rotor",))
            }
    return results


def normalize_axial_spacing(config, results=None):
    """Copy a case and resolve legacy gaps once into explicit fixed ratios.

    Existing ratios need no model evaluation. Missing ratios preserve the
    baseline's old 1.5-opening layout. Pass already calculated ``results`` to
    avoid repeating the baseline solve. First-stator and radial fields are
    omitted because no corresponding axial row-to-row passage exists.
    """
    normalized = deepcopy(config)
    inputs = normalized.get("inputs", normalized)
    stages = inputs.get("stages", [])
    axial = inputs.get("turbine_type", "axial") == "axial"
    missing = []
    for index, stage in enumerate(stages):
        for row, field in zip(("stator", "rotor"), AXIAL_SPACING_FIELDS, strict=True):
            if not axial or (row == "stator" and index == 0):
                stage.pop(field, None)
            elif field in stage:
                _positive(stage[field], field)
            else:
                missing.append((index, row, field))
    if missing and results is None:
        from .turbine import compute_turbine_performance
        results = compute_turbine_performance(normalized)
    for index, row, field in missing:
        stages[index][field] = float(get_interspace_geometry(results, index, row)["spacing_ratio"])
    return normalized
