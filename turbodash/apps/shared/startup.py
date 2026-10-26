"""Load a YAML case into an application's initial controls before serving it."""
from pathlib import Path

import yaml
from dash.development.base_component import Component
from turbodash.core.diffuser import normalize_diffuser_inputs, diffuser_settings
from turbodash.core.interspaces import normalize_axial_spacing


def components(node):
    """Walk a Dash layout, including controls inside tabs and accordions."""
    if isinstance(node, (list, tuple)):
        for child in node:
            yield from components(child)
    elif isinstance(node, Component):
        yield node
        yield from components(getattr(node, "children", None))


def configure(module, filename, kind):
    """Apply a case to controls and stage stores, leaving callbacks unchanged.

    Call before the server handles requests. Dimensional apps accept their
    existing ``inputs`` YAML schema. The efficiency app accepts its displayed
    parameter names and a list ``R_values`` in the same top-level mapping.
    """
    with Path(filename).open(encoding="utf-8") as stream:
        case = yaml.safe_load(stream)
    if not isinstance(case, dict) or not isinstance(case.get("inputs"), dict):
        raise ValueError("An app case must contain an 'inputs' mapping.")
    if kind == "turbine":
        case = module.core.convert_turbine_config_to_solidity(case)
        case["inputs"] = normalize_diffuser_inputs(case["inputs"])
        if "axial_spacing_before_rotor" in module.STAGE_DEFAULTS:
            case = normalize_axial_spacing(case)
    elif kind == "stage":
        case = module.core.convert_stage_config_to_solidity(case)
    values = case["inputs"]
    controls = list(components(module.app.layout))
    fixed = {c.id: c for c in controls if isinstance(getattr(c, "id", None), str)}

    if kind == "turbine":
        if not values.get("stages"):
            raise ValueError("A turbine app case requires at least one stage.")
        overall = dict(module.OVERALL_DEFAULTS)
        overall.update(values)
        overall["diffuser_kinetic_energy_recovery_percent"] = (
            100 * diffuser_settings(values)["kinetic_energy_recovery_fraction"]
        )
        rotation = values.get("rotational_speed", {})
        if not isinstance(rotation, dict):
            rotation = {"type": values.get("rotational_speed_type", "specific"), "value": rotation}
        overall["rotational_speed_type"] = rotation.get("type", overall["rotational_speed_type"])
        overall["rotational_speed_value"] = rotation.get("value", overall["rotational_speed_value"])
        stages = []
        for stage in values["stages"]:
            flattened = module.stage_defaults_for(overall["turbine_type"])
            flattened["name"] = stage.get("name", f"stage_{len(stages) + 1}")
            flattened.update({key: stage[key] for key in flattened if key in stage})
            for row in ("stator", "rotor"):
                for key, value in stage.get("blade_geometry", {}).get(row, {}).items():
                    flattened[f"{row}.{key}"] = value
            stages.append(flattened)
        for control in controls:
            identity = getattr(control, "id", None)
            if isinstance(identity, dict) and identity.get("scope") == "overall":
                control.value = overall[identity["key"]]
        fixed["stage_data_store"].data = stages
        if "td-inlet-pair" in fixed:
            fixed["td-inlet-pair"].value = overall["inlet_property_pair"]
        if "td-speed-mode" in fixed:
            fixed["td-speed-mode"].value = overall["rotational_speed_type"]
        if "td-stage-model-type" in fixed:
            fixed["td-stage-model-type"].data = overall["turbine_type"]
        rendered = module.render_stage_accordion(stages, overall["turbine_type"])
        accordion, count = rendered[:2]
        fixed["stages_accordion_container"].children = accordion
        fixed["stage_count_label"].children = count
        # The experimental workspace has a second set of geometry controls.
        if "geometry_rows_container" in fixed:
            fixed["geometry_rows_container"].children = rendered[2]
    else:
        aliases = {
            "p_out": "exit_pressure", "mdot": "mass_flow_rate",
            "alpha1": "stator_inlet_angle", "alpha2": "stator_exit_angle",
            "nu": "blade_velocity_ratio", "R": "degree_reaction",
            "HR_inlet": "height_radius_ratio",
            **{f"m_{pair}": f"meridional_velocity_ratio_{pair}" for pair in ("12", "23", "34")},
            **{f"rr_{pair}": f"radius_ratio_{pair}" for pair in ("12", "23", "34")},
            **{f"sigma_{row}": f"solidity_chord_{row}" for row in ("stator", "rotor")},
            **{f"xi_{row}": f"loss_coeff_{row}" for row in ("stator", "rotor")},
        } if kind == "stage" else {}
        for identity, control in fixed.items():
            if "value" not in control._prop_names:
                continue
            key = identity.removesuffix("_slider").removesuffix("_input")
            key = aliases.get(key, key)
            if identity == "R_values_input" and kind == "efficiency":
                if "R_values" in values:
                    control.value = ", ".join(str(value) for value in values["R_values"])
            elif key in values:
                control.value = values[key]
    if "loaded_cfg_store" in fixed:
        fixed["loaded_cfg_store"].data = case
    return case
