"""Portable geometry inputs for whole-design YAML files.

Only row parameter sources and overrides belong in the saved document. The
meanline baseline, edit revisions and browser epochs are rebuilt on load.
"""

from collections.abc import Mapping

from .blade_parametrization.geometry import build_section
from .blade_parametrization.parameters import resolve_parameters, state_from_dict
from .meanline import baseline_id, initialize_geometry


_ROW_NAMES = ("stator", "rotor")
_SCHEMA_VERSION = 1


def _result_context(results):
    if not isinstance(results, Mapping):
        raise ValueError("A successful meanline result is required to save or restore geometry.")
    inputs = results.get("inputs")
    turbine_type = inputs.get("turbine_type") if isinstance(inputs, Mapping) else None
    if turbine_type not in ("axial", "radial"):
        raise ValueError("Geometry requires an axial or radial turbine configuration.")
    stages = results.get("stages_performance")
    if not isinstance(stages, list) or not stages or not all(isinstance(stage, Mapping) for stage in stages):
        raise ValueError("A successful meanline result with at least one stage is required for geometry.")
    return len(stages), turbine_type


def _exact_fields(value, required, context):
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a mapping.")
    missing = required - set(value)
    unknown = set(value) - required
    if missing:
        raise ValueError(f"{context} is missing: {', '.join(sorted(missing))}.")
    if unknown:
        raise ValueError(f"{context} has unknown fields: {', '.join(sorted(map(str, unknown)))}.")


def _fresh_state(results):
    try:
        return initialize_geometry(results)
    except (ValueError, KeyError, TypeError, IndexError) as error:
        raise ValueError(f"Could not initialize geometry from the meanline result: {error}") from error


def validate_geometry(document, stage_count, turbine_type):
    """Validate all saved rows atomically and return canonical parameter state.

    Parameter-state envelopes preserve explicit overrides and inherited values.
    Plain row inputs are accepted by the parameter loader and become inherited
    values, as they do in the existing row YAML editor.
    """
    if type(stage_count) is not int or stage_count < 1:
        raise ValueError("Geometry requires a positive integer number of stages.")
    if turbine_type not in ("axial", "radial"):
        raise ValueError("Geometry requires an axial or radial turbine configuration.")
    _exact_fields(document, {"schema_version", "stages"}, "Geometry configuration")
    if type(document["schema_version"]) is not int or document["schema_version"] != _SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported geometry schema_version: {document['schema_version']!r}; "
            f"expected {_SCHEMA_VERSION}."
        )
    stages = document["stages"]
    if not isinstance(stages, list):
        raise ValueError("Geometry stages must be a list containing stator and rotor inputs for every stage.")
    if len(stages) != stage_count:
        raise ValueError(
            f"Geometry contains {len(stages)} stages, but the design contains {stage_count}; "
            "provide stator and rotor inputs for every design stage."
        )
    canonical = []
    for index, saved_stage in enumerate(stages):
        _exact_fields(saved_stage, set(_ROW_NAMES), f"Geometry stage {index + 1}")
        stage = {}
        for row in _ROW_NAMES:
            try:
                parameters = state_from_dict(saved_stage[row])
                values = resolve_parameters(parameters).values
                if values["geometry"]["type"] != turbine_type:
                    raise ValueError(
                        f"The row stream surface must match the {turbine_type} turbine; "
                        "change axial/radial configuration in Meanline first."
                    )
                if (values["leading_edge"]["pressure_side"] != "auto"
                        or values["trailing_edge"]["relief_strength"] != 0):
                    raise ValueError(
                        "The geometry editor requires leading_edge.pressure_side: auto "
                        "and trailing_edge.relief_strength: 0."
                    )
                build_section(values, check_intersections=False)
                stage[row] = parameters.to_dict()
            except (ValueError, KeyError, TypeError) as error:
                raise ValueError(f"Geometry stage {index + 1} {row}: {error}") from error
        canonical.append(stage)
    return {"schema_version": _SCHEMA_VERSION, "stages": canonical}


def export_geometry(results, state=None):
    """Save every row's accepted parameter intent for the current meanline result."""
    stage_count, turbine_type = _result_context(results)
    if state is None:
        state = _fresh_state(results)
    if not isinstance(state, Mapping):
        raise ValueError("Geometry state must be a mapping containing every blade row.")
    try:
        current_baseline = baseline_id(results)
    except (ValueError, KeyError, TypeError, IndexError) as error:
        raise ValueError(f"Could not identify geometry from the meanline result: {error}") from error
    if state.get("baseline_id") != current_baseline:
        raise ValueError("Geometry belongs to a different meanline result; wait for the current geometry before saving.")
    rows = state.get("rows")
    if not isinstance(rows, list):
        raise ValueError("Geometry state must contain every stage's stator and rotor rows.")
    stages = [{} for _ in range(stage_count)]
    for record in rows:
        if not isinstance(record, Mapping):
            raise ValueError("Every geometry row must be a mapping.")
        index, row = record.get("stage_index"), record.get("row")
        if type(index) is not int or not 0 <= index < stage_count or row not in _ROW_NAMES:
            raise ValueError("Geometry contains an unknown stage or blade row; recalculate the design before saving.")
        expected_id = f"stage_{index + 1:02d}/{row}"
        if record.get("id") != expected_id:
            raise ValueError(f"Geometry stage {index + 1} {row} must have row ID {expected_id!r}.")
        if row in stages[index]:
            raise ValueError(f"Geometry stage {index + 1} contains more than one {row} row.")
        if "parameters" not in record:
            raise ValueError(f"Geometry stage {index + 1} {row} is missing its parameter inputs.")
        stages[index][row] = record["parameters"]
    return validate_geometry(
        {"schema_version": _SCHEMA_VERSION, "stages": stages}, stage_count, turbine_type,
    )


def restore_geometry(results, document):
    """Install saved row inputs onto a fresh current-result editor baseline."""
    stage_count, turbine_type = _result_context(results)
    canonical = validate_geometry(document, stage_count, turbine_type)
    state = _fresh_state(results)
    for record in state["rows"]:
        record["parameters"] = canonical["stages"][record["stage_index"]][record["row"]]
    return state
