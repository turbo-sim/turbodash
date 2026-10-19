"""Model paths, editor defaults and validation shared by optimization clients."""

from copy import deepcopy
import math

from turbodash.optimization import read_path


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


def variable_catalog(config):
    """Offer only effective independent inputs for every current stage/row."""
    inputs = flat_config(config)["inputs"]
    catalog = {}

    def add(path, label, lower, upper):
        try:
            current = finite(read_path(inputs, path), label)
        except (ValueError, KeyError, TypeError):
            return
        catalog[path] = dict(label=label, value=current, lower=lower, upper=upper)

    add("blade_velocity_ratio", "Global · Blade velocity ratio [-]", .25, 1)
    actual = inputs.get("rotational_speed_type") == "actual"
    speed = inputs.get("rotational_speed", .7)
    add("rotational_speed", "Global · Rotational speed [rpm]" if actual else "Global · Specific speed [-]",
        max(1, speed * .5) if actual else .2, speed * 1.5 if actual else 1.5)
    radial = inputs.get("turbine_type") == "radial"
    for i, stage in enumerate(inputs.get("stages", [])):
        prefix = f"stages.{i}."
        title = f"Stage {i + 1}"
        for key, label, lo, hi in (
            ("degree_reaction", "Reaction [-]", 0, .75),
            ("stator_exit_angle", "Stator exit flow angle [deg]", 50, 80),
            ("work_fraction_split", "Work fraction weight [-]", .1, 2),
            *((f"meridional_velocity_ratio_{pair}", f"Meridional velocity ratio {pair} [-]", .5, 1.5)
              for pair in ("12", "23", "34")),
        ):
            if key != "work_fraction_split" or len(inputs["stages"]) > 1:
                add(prefix + key, f"{title} · {label}", lo, hi)
        # Axial row radii are fixed by that model; radial radius changes set chord.
        if radial:
            for pair in ("01", "12", "23", "34"):
                add(prefix + f"radius_ratio_{pair}", f"{title} · Radius ratio {pair} [-]",
                    1.01 if pair in ("12", "34") else .9, 1.4)
        for row in ("stator", "rotor"):
            row_title = f"{title} · {row.title()}"
            row_path = prefix + f"blade_geometry.{row}."
            add(row_path + "solidity_chord", f"{row_title} · Solidity [-]", .5, 2)
            add(row_path + "trailing_edge_thickness_to_opening", f"{row_title} · Trailing thickness / opening [-]", .02, .1)
            if not radial:
                add(row_path + "aspect_ratio", f"{row_title} · Axial aspect ratio [-]", 1, 4)
    return catalog


def output_catalog(config, results=None):
    """Scalar objectives and scalar/wildcard constraints with physical scales."""
    catalog = {}

    def add(path, label, scale, objective=False):
        if results:
            try:
                read_path(results, path)
            except (KeyError, ValueError, TypeError, IndexError):
                return
        catalog[path] = dict(label=label, scale=scale, objective=objective)

    for key, (label, scale) in PERFORMANCE.items():
        add(f"$overall_performance.{key}", f"Overall · {label}", scale, True)
    scopes = [("*", "All stages")]
    scopes += [(str(i), f"Stage {i + 1}") for i, _ in enumerate(config["inputs"]["stages"])]
    for index, title in scopes:
        for key, (label, scale) in PERFORMANCE.items():
            add(f"$stages_performance.{index}.stage_performance.{key}", f"{title} · {label}", scale)
        for row in ("*", "stator", "rotor"):
            for key, (label, scale) in GEOMETRY.items():
                add(f"$stages_performance.{index}.geometry_meanline.{row}.{key}",
                    f"{title} · {'Both rows' if row == '*' else row.title()} · {label}", scale)
        for station in ("1", "2", "3", "4"):
            for key, label, scale in (("Ma_abs", "Absolute Mach [-]", 1), ("Ma_rel", "Relative Mach [-]", 1),
                                      ("p", "Pressure [Pa]", 1e5), ("T", "Temperature [K]", 500)):
                add(f"$stages_performance.{index}.flow_stations.{int(station) - 1}.{key}",
                    f"{title} · Station {station} · {label}", scale)
    return catalog


def finite(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number.")
    return float(value)


def build_settings(config, variables, constraints, objective, direction, tolerance, iterations, feasibility, results=None, *, require_variables=True):
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
    return dict(design_variables=selected, constraints=limits,
                objective_function=dict(variable=objective, type=direction, normalize=outputs[objective]["scale"]),
                feasibility_tolerance=feas,
                solver_options=dict(library="scipy", method="slsqp", tolerance=tol, max_iterations=int(count),
                                    plot_convergence=False, print_convergence=False))
