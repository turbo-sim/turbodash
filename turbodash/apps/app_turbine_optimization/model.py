"""Model paths, editor defaults and validation shared by optimization clients."""

from copy import deepcopy
import math

from turbodash.optimization import read_path
from turbodash.optimization.slsqp import normalize_solver_options


# The same order is used for loading, launching and exporting the editor.
SLSQP_EXTRA_CONTROLS = (
    ("first_step_fraction", "opt-first-step"),
    ("max_step_fraction", "opt-max-step"),
    ("step_growth_factor", "opt-step-growth-factor"),
    ("domain_backtracking", "opt-domain-backtracking"),
    ("backtrack_factor", "opt-backtrack-factor"),
    ("max_backtracks", "opt-max-backtracks"),
    ("finite_diff_step", "opt-finite-diff-step"),
    ("max_fd_backtracks", "opt-max-fd-backtracks"),
    ("optimality_tol", "opt-optimality-tolerance"),
)


def solver_editor_options(settings=None):
    """Validate and migrate YAML options to the app's native SLSQP controls."""
    settings = {} if settings is None else settings
    if not isinstance(settings, dict):
        raise ValueError("Optimization settings must be a mapping.")
    options = normalize_solver_options(
        settings.get("solver_options", {}),
        feasibility_tolerance=settings.get("feasibility_tolerance", 1e-6),
    )
    options.pop("feasibility_tol")  # YAML has one authoritative feasibility limit.
    return options


def extra_solver_options(values):
    """Collect the shared control order without silently dropping an option."""
    if not values:  # Also permits clients that use the default controls.
        return {}
    if len(values) != len(SLSQP_EXTRA_CONTROLS):
        raise ValueError("Incomplete SLSQP controls. Reload the app and try again.")
    return {name: value for (name, _), value in zip(SLSQP_EXTRA_CONTROLS, values)}


PERFORMANCE = {
    "efficiency_ts": ("Total-to-static efficiency [-]", 1),
    "efficiency_tt": ("Total-to-total efficiency [-]", 1),
    "power_actual": ("Shaft power [W]", 1e5),
    "shaft_torque": ("Shaft torque [N m]", 100),
    "maximum_mach_number": ("Maximum Mach number [-]", 1),
    "exit_rotor_diameter": ("Exit rotor diameter [m]", .1),
    "specific_speed": ("Specific speed [-]", 1),
    "rotational_speed": ("Rotational speed [rpm]", 10000),
}
GEOMETRY = {
    "flaring_angle": ("Flaring angle [deg]", 10),
    "height": ("Blade height [m]", .01),
    "chord": ("True chord [m]", .01),
    "chord_meridional": ("Meridional chord [m]", .01),
    "radius_in": ("Inlet radius [m]", .1),
    "radius_out": ("Exit radius [m]", .1),
    "aspect_ratio": ("Aspect ratio [-]", 1),
    "solidity_chord": ("Solidity [-]", 1),
    "blade_count_continuous": ("Continuous blade count [-]", 50),
}


def flat_config(config):
    """Use one numeric speed path without mutating the app's nested schema."""
    result = deepcopy(config)
    inputs = result["inputs"]
    rotation = inputs.get("rotational_speed")
    if isinstance(rotation, dict):
        inputs["rotational_speed_type"] = rotation["type"]
        inputs["rotational_speed"] = rotation["value"]
    return result


def nested_config(config):
    result = flat_config(config)
    inputs = result["inputs"]
    inputs["rotational_speed"] = {
        "type": inputs.pop("rotational_speed_type", "specific"),
        "value": inputs["rotational_speed"],
    }
    return result


def _stage_title(index, stage):
    """Keep stage identity visible even when names are repeated."""
    title = f"Stage {index + 1}"
    name = str(stage.get("name", "")).strip()
    return f"{title} · {name}" if name else title


def _radius_bounds(value, blade_row):
    """Keep a radial blade row on its current side of zero radial chord."""
    if not blade_row:
        return min(.9, .8 * value), max(1.4, 1.2 * value)
    if value < 1:
        return min(.7, .8 * value), max(.99, (1 + value) / 2)
    if value == 1:
        return 1.01, 1.4
    return min(1.01, (1 + value) / 2), max(1.4, 1.2 * value)


def variable_catalog(config):
    """Offer bounded continuous inputs, with hierarchy and preset applicability.

    ``preset_active`` excludes inputs that cannot affect supported outputs for
    this model configuration. Individual choices retain existing geometry paths
    so imported optimization settings remain editable.
    """
    inputs = flat_config(config)["inputs"]
    catalog = {}

    def add(path, label, lower, upper, short_label, hierarchy, *, preset_active=True):
        try:
            current = finite(read_path(inputs, path), label)
        except (ValueError, KeyError, TypeError, IndexError):
            return
        catalog[path] = dict(label=label, short_label=short_label, hierarchy=hierarchy,
                             value=current, lower=lower, upper=upper,
                             preset_active=preset_active)

    add("blade_velocity_ratio", "Global · Blade velocity ratio [-]", .25, 1,
        "Blade velocity ratio [-]", ["Overall"])
    actual = inputs.get("rotational_speed_type") == "actual"
    speed = inputs.get("rotational_speed", .7)
    speed_label = "Rotational speed [rpm]" if actual else "Specific speed [-]"
    add("rotational_speed", f"Global · {speed_label}",
        max(1, speed * .5) if actual else .2, speed * 1.5 if actual else 1.5,
        speed_label, ["Overall"])
    radial = inputs.get("turbine_type") in ("radial", "radial_inflow", "radial_outflow")
    empirical = inputs.get("loss_model", "benner") in ("benner", "moustapha", "kacker_okapuu")
    for i, stage in enumerate(inputs.get("stages", [])):
        prefix = f"stages.{i}."
        title = f"Stage {i + 1}"
        hierarchy_title = _stage_title(i, stage)
        for key, label, lo, hi in (
            ("degree_reaction", "Reaction [-]", 0, .75),
            ("stator_exit_angle", "Stator exit flow angle [deg]", 50, 80),
            ("work_fraction_split", "Work fraction weight [-]", .1, 2),
            *((f"meridional_velocity_ratio_{pair}", f"Meridional velocity ratio {pair} [-]", .5, 1.5)
              for pair in ("12", "23", "34")),
        ):
            if key != "work_fraction_split" or len(inputs["stages"]) > 1:
                if key == "work_fraction_split":
                    weight = stage.get(key, 1)
                    if isinstance(weight, (int, float)) and weight > 0 and math.isfinite(weight):
                        lo, hi = .1 * weight, 2 * weight
                category = "Meridional velocity ratios" if key.startswith("meridional_") else "Stage inputs"
                add(prefix + key, f"{title} · {label}", lo, hi, label,
                    [hierarchy_title, category])
        # Axial row radii are fixed by that model; radial radius changes set chord.
        if radial:
            for pair in ("01", "12", "23", "34"):
                # The first stage is anchored at station 3; its station 0 radius
                # is unused. In later stages 01 connects to the preceding rotor.
                if i == 0 and pair == "01":
                    continue
                path = prefix + f"radius_ratio_{pair}"
                try:
                    ratio = finite(read_path(inputs, path), path)
                except (ValueError, KeyError, TypeError, IndexError):
                    continue
                lo, hi = _radius_bounds(ratio, pair in ("12", "34"))
                label = f"Radius ratio {pair} [-]"
                add(path, f"{title} · {label}", lo, hi, label,
                    [hierarchy_title, "Radius ratios"])
        for row in ("stator", "rotor"):
            row_title = f"{title} · {row.title()}"
            row_path = prefix + f"blade_geometry.{row}."
            hierarchy = [hierarchy_title, f"{row.title()} geometry"]
            add(row_path + "solidity_chord", f"{row_title} · Solidity [-]", .5, 2,
                "Solidity [-]", hierarchy)
            add(row_path + "trailing_edge_thickness_to_opening",
                f"{row_title} · Trailing thickness / opening [-]", .02, .1,
                "Trailing thickness / opening [-]", hierarchy, preset_active=empirical)
            # The profile thickness exponent is zero for a first stator with
            # nonnegative inlet flow: its entire preset exit-angle range is
            # positive. Later-stage inlet flow and rotor turning can vary.
            thickness_active = empirical and not (
                i == 0 and row == "stator" and inputs.get("inlet_flow_angle", 0) >= 0
            )
            add(row_path + "maximum_thickness_to_chord",
                f"{row_title} · Maximum thickness / meridional chord [-]", .15, .35,
                "Maximum thickness / meridional chord [-]", hierarchy,
                preset_active=thickness_active)
            if not radial:
                add(row_path + "aspect_ratio", f"{row_title} · Axial aspect ratio [-]", 1, 4,
                    "Axial aspect ratio [-]", hierarchy)
    return catalog


def variable_groups(config, catalog=None):
    """Return applicable presets without changing the caller's inputs/catalog.

    Work fractions are normalized by the solver. Presets vary N-1 weights and
    hold the final stage's weight fixed, removing the common scaling freedom.
    """
    catalog = variable_catalog(config) if catalog is None else catalog
    stages = config["inputs"].get("stages", [])
    groups = {}

    def add(key, label, description, paths, hierarchy):
        paths = list(dict.fromkeys(path for path in paths
                                   if path in catalog and catalog[path].get("preset_active", True)))
        if paths:
            groups[key] = dict(label=label, description=description, paths=paths, hierarchy=hierarchy)

    add("overall", "Shaft speed and blade velocity ratio",
        "Vary the selected shaft-speed definition and overall blade velocity ratio.",
        ["blade_velocity_ratio", "rotational_speed"], ["Overall"])
    stage_design = [f"stages.{i}.{key}" for i in range(len(stages))
                    for key in ("degree_reaction", "stator_exit_angle")]
    add("stage_design", "Reaction and stator angle · all stages",
        "Vary reaction and stator exit flow angle in every stage.", stage_design, ["All stages"])
    for i, stage in enumerate(stages):
        title = _stage_title(i, stage)
        add(f"stage_{i}_design", "Reaction and stator angle",
            f"Vary reaction and stator exit flow angle in {title}.",
            [f"stages.{i}.degree_reaction", f"stages.{i}.stator_exit_angle"], [title])
    add("work_split", "Stage work split",
        "Vary the first N−1 relative work weights; keep the final stage's weight fixed.",
        [f"stages.{i}.work_fraction_split" for i in range(len(stages) - 1)], ["All stages"])
    add("radial_radii", "Radial passage radii",
        "Vary effective radial radius ratios while preserving each blade row's radial direction.",
        [path for path in catalog if ".radius_ratio_" in path], ["All stages"])
    add("meridional_ratios", "Meridional velocity ratios",
        "Vary the inlet-to-exit meridional velocity ratios of each row and inter-row gap.",
        [path for path in catalog if ".meridional_velocity_ratio_" in path], ["All stages"])
    add("blade_geometry", "Blade geometry · all stages",
        "Vary applicable solidity, axial aspect ratio and loss-sensitive thickness ratios.",
        [path for path in catalog if ".blade_geometry." in path], ["All stages"])
    for i, stage in enumerate(stages):
        for row in ("stator", "rotor"):
            add(f"stage_{i}_{row}_geometry", f"{row.title()} geometry",
                "Vary applicable solidity, axial aspect ratio and loss-sensitive thickness ratios.",
                [path for path in catalog if path.startswith(f"stages.{i}.blade_geometry.{row}.")],
                [_stage_title(i, stage)])
    reference = f"stages.{len(stages) - 1}.work_fraction_split"
    add("all", "Optimize all applicable variables",
        "Combine all applicable presets; keep operating conditions, inactive geometry inputs "
        "and the final stage's reference work weight fixed.",
        [path for path in catalog if path != reference], ["Overall"])
    return groups


def _constraint_default(path, scale):
    """Useful editable starting targets, rather than zero for every quantity."""
    quantity = path.rsplit(".", 1)[-1]
    if quantity in ("maximum_mach_number", "Ma_abs", "Ma_rel"):
        return "<=", 1
    if quantity == "flaring_angle":
        return "<=", 10
    if quantity == "height":
        return ">=", .001
    if quantity in ("efficiency_ts", "efficiency_tt"):
        return ">=", .8
    return "=", scale


def output_catalog(config, results=None):
    """Scalar objectives and scalar/wildcard constraints with physical scales."""
    catalog = {}

    def add(path, label, scale, short_label, hierarchy, objective=False):
        if results:
            try:
                read_path(results, path)
            except (KeyError, ValueError, TypeError, IndexError):
                return
        comparison, target = _constraint_default(path, scale)
        catalog[path] = dict(label=label, short_label=short_label, hierarchy=hierarchy,
                             scale=scale, objective=objective, comparison=comparison, target=target)

    for key, (label, scale) in PERFORMANCE.items():
        add(f"$overall_performance.{key}", f"Overall · {label}", scale, label, ["Overall"], True)
    scopes = [("*", "All stages", "All stages")]
    scopes += [(str(i), f"Stage {i + 1}", _stage_title(i, stage))
               for i, stage in enumerate(config["inputs"]["stages"])]
    for index, title, hierarchy_title in scopes:
        for key, (label, scale) in PERFORMANCE.items():
            add(f"$stages_performance.{index}.stage_performance.{key}", f"{title} · {label}",
                scale, label, [hierarchy_title, "Performance"])
        for row in ("*", "stator", "rotor"):
            row_title = "Both rows" if row == "*" else row.title()
            for key, (label, scale) in GEOMETRY.items():
                add(f"$stages_performance.{index}.geometry_meanline.{row}.{key}",
                    f"{title} · {row_title} · {label}", scale, label,
                    [hierarchy_title, f"{row_title} geometry"])
        for station in ("1", "2", "3", "4"):
            for key, label, scale in (("Ma_abs", "Absolute Mach [-]", 1), ("Ma_rel", "Relative Mach [-]", 1),
                                      ("p", "Pressure [Pa]", 1e5), ("T", "Temperature [K]", 500)):
                add(f"$stages_performance.{index}.flow_stations.{int(station) - 1}.{key}",
                    f"{title} · Station {station} · {label}", scale, label,
                    [hierarchy_title, f"Station {station}"])
    return catalog


def constraint_groups(config, catalog=None):
    """Editable engineering limits; two-sided bounds use distinct row specs."""
    catalog = output_catalog(config) if catalog is None else catalog
    groups = {}

    def row(path, comparison=None, target=None):
        entry = catalog.get(path)
        if entry is None:
            return None
        return dict(path=path, comparison=comparison or entry["comparison"],
                    target=entry["target"] if target is None else target, scale=entry["scale"])

    def add(key, label, description, rows, hierarchy):
        rows = [spec for spec in rows if spec is not None]
        if rows:
            groups[key] = dict(label=label, description=description, rows=rows,
                               paths=list(dict.fromkeys(spec["path"] for spec in rows)),
                               hierarchy=hierarchy)

    def geometry_limits(scope):
        prefix = f"$stages_performance.{scope}.geometry_meanline.*."
        return [row(prefix + "flaring_angle", ">=", -10),
                row(prefix + "flaring_angle", "<=", 10),
                row(prefix + "height", ">=", .001)]

    add("subsonic", "Maximum Mach number",
        "Keep blade-row Mach numbers at or below 1: absolute in stators, relative in rotors.",
        [row("$overall_performance.maximum_mach_number")], ["Overall"])
    add("flaring_limits", "Flaring angle ±10°",
        "Bound both positive and negative flaring in every blade row; edit either limit as needed.",
        geometry_limits("*")[:2], ["All stages"])
    add("minimum_height", "Minimum blade height",
        "Require at least 1 mm mean blade height in every row; edit this allowance as needed.",
        geometry_limits("*")[2:], ["All stages"])
    for i, stage in enumerate(config["inputs"].get("stages", [])):
        add(f"stage_{i}_limits", "Mach, flaring and blade-height limits",
            "Use editable starting limits: maximum Mach 1, flaring ±10° and blade height at least 1 mm.",
            [row(f"$stages_performance.{i}.stage_performance.maximum_mach_number"),
             *geometry_limits(str(i))], [_stage_title(i, stage)])
    return groups


def finite(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number.")
    return float(value)


def build_settings(config, variables, constraints, objective, direction, tolerance, iterations, feasibility, results=None, *, require_variables=True, solver_options=None):
    """Validate UI values before launching; let the API clip initial guesses."""
    choices = variable_catalog(config)
    outputs = output_catalog(config, results)
    if objective not in outputs or not outputs[objective]["objective"]:
        raise ValueError("Select a scalar objective from the current design.")
    if direction not in ("maximize", "minimize"):
        raise ValueError("Select maximize or minimize.")
    selected = {}
    for row in variables:
        path = row.get("path")
        if path not in choices:
            raise ValueError("Select an available design variable in every row.")
        if path in selected:
            raise ValueError(f"Duplicate variable: {choices[path]['label']}.")
        lo, hi = finite(row.get("lower"), "Lower bound"), finite(row.get("upper"), "Upper bound")
        if lo >= hi or not math.isfinite(hi - lo):
            raise ValueError("Each variable needs finite bounds with lower < upper.")
        initial = row.get("initial")
        if initial is not None:
            initial = finite(initial, "Initial value")
        selected[path] = dict(lower_value=lo, upper_value=hi, initial_value=initial)
    if not selected and require_variables:
        raise ValueError("Add at least one design variable before starting.")
    limits = []
    for row in constraints:
        path = row.get("path")
        if path not in outputs:
            raise ValueError("Select an available calculated quantity in every constraint.")
        operator = row.get("comparison")
        if operator not in ("<=", "=", ">="):
            raise ValueError("Choose a constraint comparison.")
        scale = finite(row.get("scale"), "Constraint scale")
        if scale <= 0:
            raise ValueError("Constraint scale must be positive, including for a zero target.")
        limits.append(dict(variable=path, type=operator, value=finite(row.get("target"), "Target"), normalize=scale))
    tol, feas = finite(tolerance, "Solver tolerance"), finite(feasibility, "Feasibility tolerance")
    count = finite(iterations, "Iteration limit")
    if min(tol, feas) <= 0 or count < 1 or count > 2000 or not count.is_integer():
        raise ValueError("Tolerances must be positive; iteration limit must be an integer from 1 to 2000.")
    options = solver_editor_options(dict(
        solver_options=dict(solver_options or {}, ftol=tol, maxiter=int(count)),
        feasibility_tolerance=feas,
    ))
    return dict(design_variables=selected, constraints=limits,
                objective_function=dict(variable=objective, type=direction, normalize=outputs[objective]["scale"]),
                feasibility_tolerance=feas,
                solver_options=options)
