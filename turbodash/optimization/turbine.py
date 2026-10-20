"""Turbine optimization, plotting, and YAML output using ThermOpt's solver pattern.

The numerical model stays independent of the app. PySolverView handles the
solver, variable scaling, convergence reporting, and convergence plots.
"""

from collections import OrderedDict
from collections.abc import Mapping, MutableMapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pysolver_view as psv
import yaml

from turbodash.io.yaml import dump_turbine_yaml


# =============================================================================
# Numeric result selectors and ThermOpt-style objectives / constraints
# =============================================================================

def _path_parts(path):
    if not isinstance(path, str):
        raise TypeError("A result path must be a string.")
    path = path.strip()
    if path.startswith("$"):
        path = path[1:]
    parts = path.split(".")
    if not path or any(not part or part != part.strip() for part in parts):
        raise ValueError(f"Invalid result path: {path!r}.")
    return parts


def _child(node, key, path):
    if isinstance(node, Mapping):
        if key not in node:
            raise KeyError(f"Path {path!r}: dictionary has no key {key!r}.")
        return node[key]
    if isinstance(node, (list, tuple, np.ndarray)):
        if not key.isdecimal():
            raise ValueError(f"Path {path!r}: sequence index {key!r} must be nonnegative.")
        try:
            return node[int(key)]
        except IndexError as exc:
            raise IndexError(f"Path {path!r}: sequence index {key!r} is out of range.") from exc
    raise TypeError(f"Path {path!r}: cannot traverse a {type(node).__name__}.")


def read_path(data, path):
    """Read an existing dotted path; ``*`` selects every mapping or sequence item."""
    parts = _path_parts(path)

    def walk(node, remaining):
        if not remaining:
            return node
        key, *tail = remaining
        if key == "*":
            if isinstance(node, Mapping):
                children = node.values()
            elif isinstance(node, (list, tuple)) or (
                isinstance(node, np.ndarray) and node.ndim > 0
            ):
                children = node
            else:
                raise TypeError(f"Path {path!r}: wildcard requires a mapping or sequence.")
            return [walk(child, tail) for child in children]
        return walk(_child(node, key, path), tail)

    return walk(data, parts)


def set_path(data, path, value):
    """Replace an existing mapping/list value using a path without wildcards."""
    parts = _path_parts(path)
    if "*" in parts:
        raise ValueError(f"Design-variable path {path!r} cannot contain wildcards.")
    parent = data
    for part in parts[:-1]:
        parent = _child(parent, part, path)
    key = parts[-1]
    _child(parent, key, path)  # Require the destination to exist.
    if isinstance(parent, MutableMapping):
        parent[key] = value
    elif isinstance(parent, (list, np.ndarray)):
        parent[int(key)] = value
    else:
        raise TypeError(f"Path {path!r}: destination is not mutable.")


def _numeric_array(value, label):
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must have a regular numeric shape.") from exc
    if array.dtype.kind not in "iuf":
        raise TypeError(f"{label} must contain real numeric values, excluding booleans.")
    if array.size == 0:
        raise ValueError(f"{label} cannot be empty.")
    with np.errstate(over="ignore", invalid="ignore"):
        array = array.astype(float, copy=False)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must contain only finite values.")
    return array


def _label(selector):
    if isinstance(selector, str):
        return selector
    return getattr(selector, "__name__", type(selector).__name__)


def resolve_value(selector, data):
    """Resolve a numeric dotted selector or ``callable(data)`` without evaluation."""
    if callable(selector):
        value = selector(data)
    elif isinstance(selector, str):
        value = read_path(data, selector)
    else:
        raise TypeError("A selector must be a dotted path or a callable.")
    array = _numeric_array(value, f"Selection {_label(selector)!r}")
    return array.item() if array.ndim == 0 else array


def evaluate_objective(results, spec):
    """Return the physical scalar objective; the driver applies sign and scaling."""
    if not isinstance(spec, Mapping):
        raise TypeError("objective_function must be a dictionary.")
    if spec.get("type") not in ("minimize", "maximize"):
        raise ValueError("Objective type must be 'minimize' or 'maximize'.")
    if "variable" not in spec:
        raise ValueError("objective_function requires a variable selector.")
    value = np.asarray(resolve_value(spec["variable"], results))
    if value.size != 1:
        raise ValueError(f"Objective {_label(spec['variable'])!r} must be scalar; got {value.shape}.")
    return float(value.reshape(-1)[0])


def evaluate_constraints(results, specs):
    """Return equality/``<= 0`` inequality residuals and component-wise reports."""
    equalities, inequalities, report = [], [], []
    if specs is None:
        specs = []
    if not isinstance(specs, (list, tuple)):
        raise TypeError("constraints must be a list of dictionaries.")
    for number, spec in enumerate(specs):
        if not isinstance(spec, Mapping):
            raise TypeError(f"Constraint {number} must be a dictionary.")
        operator = spec.get("type")
        if operator not in ("=", "==", "<", "<=", ">", ">="):
            raise ValueError(f"Constraint {number} has unsupported type {operator!r}.")
        if "variable" not in spec or "value" not in spec:
            raise ValueError(f"Constraint {number} requires variable and value.")
        selector = spec["variable"]
        label = _label(selector)
        current = np.asarray(resolve_value(selector, results))
        target_spec = spec["value"]
        target = (
            np.asarray(resolve_value(target_spec, results))
            if isinstance(target_spec, str) or callable(target_spec)
            else _numeric_array(target_spec, f"Target for constraint {label!r}")
        )
        normalize = spec.get("normalize", False)
        if isinstance(normalize, (bool, np.bool_)):
            scale = np.abs(target) if normalize else np.asarray(1.0)
        else:
            scale = _numeric_array(normalize, f"Normalization for constraint {label!r}")
        if np.any(scale <= 0):
            raise ValueError(
                f"Normalization for constraint {label!r} must be positive; "
                "use an explicit positive scale for a zero target."
            )
        try:
            current, target, scale = np.broadcast_arrays(current, target, scale)
        except ValueError as exc:
            raise ValueError(
                f"Constraint {label!r} value, target, and normalization have "
                f"incompatible shapes {current.shape}, {target.shape}, {np.shape(scale)}."
            ) from exc
        equality = operator in ("=", "==")
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            residual = target - current if operator in (">", ">=") else current - target
            normalized = residual / scale
        if not np.all(np.isfinite(residual)) or not np.all(np.isfinite(normalized)):
            raise ValueError(f"Constraint {label!r} produced nonfinite residuals.")
        destination = equalities if equality else inequalities
        destination.extend(normalized.reshape(-1).tolist())
        for index, (value, goal, physical, scaled) in enumerate(
            zip(current.flat, target.flat, residual.flat, normalized.flat)
        ):
            report.append({
                "variable": label,
                "index": index if current.ndim else None,
                "type": operator,
                "value": float(value),
                "target": float(goal),
                "residual": float(physical),
                "normalized_residual": float(scaled),
                "violation": float(abs(physical) if equality else max(0.0, physical)),
                "normalized_violation": float(abs(scaled) if equality else max(0.0, scaled)),
            })
    return np.asarray(equalities, dtype=float), np.asarray(inequalities, dtype=float), report


# =============================================================================
# Design-variable configuration
# =============================================================================


def load_case(filename):
    """Load a standalone YAML case containing inputs and optimization settings."""
    with Path(filename).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("The case YAML must contain a dictionary.")
    for section in ("inputs", "optimization"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"The case needs a '{section}' dictionary.")
    return config


def _scalar(value, label):
    array = np.asarray(value)
    if array.ndim != 0 or array.dtype.kind not in "iuf" or not np.isfinite(array):
        raise ValueError(f"{label} must be a finite numeric scalar.")
    return float(array)


def _read_design_variables(inputs, specifications, logger=None):
    """Read physical bounds and initial values in YAML insertion order."""
    if not isinstance(specifications, dict) or not specifications:
        raise ValueError("design_variables must be a nonempty dictionary.")

    names, paths, lower, initial, upper = [], [], [], [], []
    for name, spec in specifications.items():
        if not isinstance(spec, dict):
            raise ValueError(f"Design variable {name!r} must be a dictionary.")
        path = spec.get("path", name)
        if not isinstance(path, str) or "*" in path:
            raise ValueError(f"Design variable {name!r} needs one scalar input path.")
        path = path.strip().removeprefix("$")
        current = _scalar(read_path(inputs, path), f"Input at {path!r}")
        if path in paths:
            raise ValueError(f"Duplicate design-variable path: {path!r}.")
        lo = _scalar(spec["lower_value"], f"Lower bound of {name!r}")
        hi = _scalar(spec["upper_value"], f"Upper bound of {name!r}")
        start = spec.get("initial_value")
        start = current if start is None else _scalar(start, f"Initial value of {name!r}")
        if not lo < hi or not np.isfinite(hi - lo):
            raise ValueError(f"Bounds for {name!r} must be finite with lower_value < upper_value.")
        if not lo <= start <= hi:
            clipped = float(np.clip(start, lo, hi))
            message = (
                f"Initial value for {name!r} clipped from {start:g} to {clipped:g} "
                f"(bounds [{lo:g}, {hi:g}])."
            )
            if logger is None:
                print(message)
            else:
                logger.warning(message)
            start = clipped
        names.append(name)
        paths.append(path)
        lower.append(lo)
        initial.append(start)
        upper.append(hi)
    return names, paths, np.array(lower), np.array(initial), np.array(upper)


# =============================================================================
# PySolverView optimization problem
# =============================================================================


class TurbineOptimizationProblem(psv.OptimizationProblem):
    """Adapt the existing turbine calculation to PySolverView's physical API."""

    def __init__(self, config, optimization=None, evaluator=None, logger=None):
        self.source_config = deepcopy(config if "inputs" in config else {"inputs": config})
        settings = optimization if optimization is not None else self.source_config.get("optimization")
        if not isinstance(settings, dict):
            raise ValueError("Provide an optimization dictionary or a case with an optimization section.")
        self.optimization = deepcopy(settings)
        self.source_config["optimization"] = deepcopy(settings)
        self.config = deepcopy(self.source_config)
        for key in ("overall_performance", "stages_performance", "optimization_result"):
            self.config.pop(key, None)

        self.objective_spec = self.optimization.get("objective_function")
        if not isinstance(self.objective_spec, dict):
            raise ValueError("objective_function must be a dictionary.")
        objective_type = self.objective_spec.get("type")
        if objective_type not in ("minimize", "maximize"):
            raise ValueError("Objective type must be 'minimize' or 'maximize'.")
        self.objective_sign = -1.0 if objective_type == "maximize" else 1.0
        self.objective_scale = _scalar(self.objective_spec.get("normalize", 1.0), "Objective normalization")
        self.feasibility_tolerance = _scalar(
            self.optimization.get("feasibility_tolerance", 1e-6), "Feasibility tolerance"
        )
        if min(self.objective_scale, self.feasibility_tolerance) <= 0:
            raise ValueError("Objective normalization and feasibility tolerance must be positive.")
        self.constraint_specs = self.optimization.get("constraints") or []
        if not isinstance(self.constraint_specs, list):
            raise ValueError("constraints must be a list, empty, or omitted.")

        (
            self.variable_names, self.variable_paths,
            self.lower_bounds, self.x0, self.upper_bounds,
        ) = _read_design_variables(self.config["inputs"], self.optimization.get("design_variables"), logger)
        self.bound_range = self.upper_bounds - self.lower_bounds
        self.problem_scale = 1.0
        self.cache = OrderedDict()
        self.n_evaluations = 0
        self.constraint_sizes = None

        if evaluator is None:
            from turbodash.core.turbine import compute_turbine_performance
            evaluator = compute_turbine_performance
        self.evaluator = evaluator

        # Like ThermOpt, evaluate before constructing the solver: it queries
        # constraint counts in its constructor.
        self.fitness(self.x0)
        if self.get_nec() > len(self.variable_names):
            raise ValueError("There are more scalar equality constraints than design variables.")

    def evaluate(self, physical_values):
        """Return a complete evaluation at this exact physical design point."""
        values = np.asarray(physical_values, dtype=float)
        if values.shape != self.x0.shape or not np.all(np.isfinite(values)):
            raise ValueError("The design vector has an invalid shape or nonfinite values.")
        cache_key = values.tobytes()
        if cache_key in self.cache:
            self.cache.move_to_end(cache_key)
            return self.cache[cache_key]

        config = deepcopy(self.config)
        variables = dict(zip(self.variable_names, values.tolist(), strict=True))
        for path, value in zip(self.variable_paths, values, strict=True):
            set_path(config["inputs"], path, float(value))
        try:
            with np.errstate(invalid="raise", divide="raise", over="raise"):
                turbine_results = self.evaluator(config)
        except (ValueError, ArithmeticError) as exc:
            raise ValueError(
                f"Turbine evaluation failed at {variables}. "
                f"Choose bounds within the model domain. {exc}"
            ) from exc

        objective = evaluate_objective(turbine_results, self.objective_spec)
        equalities, inequalities, report = evaluate_constraints(turbine_results, self.constraint_specs)
        sizes = (equalities.size, inequalities.size)
        if self.constraint_sizes is None:
            self.constraint_sizes = sizes
        elif sizes != self.constraint_sizes:
            raise ValueError("The number of constraint components changed during optimization.")
        violation = max(
            float(np.max(np.abs(equalities), initial=0)),
            float(np.max(inequalities, initial=0)),
            float(np.max((self.lower_bounds - values) / self.bound_range, initial=0)),
            float(np.max((values - self.upper_bounds) / self.bound_range, initial=0)),
        )
        evaluation = {
            "config": config, "turbine_results": turbine_results,
            "design_variables": variables, "objective": objective,
            "equalities": equalities, "inequalities": inequalities,
            "constraint_report": report, "max_constraint_violation": violation,
        }
        self.n_evaluations += 1
        self.cache[cache_key] = evaluation
        if len(self.cache) > 128:
            self.cache.popitem(last=False)
        return evaluation

    def fitness(self, physical_values):
        evaluation = self.evaluate(physical_values)
        self.f = self.objective_sign * evaluation["objective"] / self.objective_scale
        self.c_eq = evaluation["equalities"]
        self.c_ineq = evaluation["inequalities"]
        self.constraint_data = evaluation["constraint_report"]
        return psv.combine_objective_and_constraints(self.f, self.c_eq, self.c_ineq)

    def get_bounds(self):
        return self.lower_bounds, self.upper_bounds

    def get_nec(self):
        return self.c_eq.size

    def get_nic(self):
        return self.c_ineq.size


@dataclass
class OptimizationResult:
    """Physical design, evaluated geometry, and the actual PySolverView solver."""

    config: dict
    turbine_results: dict
    geometry: list
    design_variables: dict
    objective: float
    initial_objective: float
    constraint_report: list
    success: bool
    solver_success: bool
    feasible: bool
    message: str
    max_constraint_violation: float
    history: list
    solver: psv.OptimizationSolver
    n_evaluations: int
    initial_turbine_results: dict
    initial_config: dict
    source_config: dict


def optimize_turbine(config, optimization=None, *, evaluator=None, callback=None, logger=None):
    """Optimize a case and return its physical design and convergence information.

    If optimization is omitted, read config['optimization']. Variables omitted
    from design_variables stay fixed at their current input values. Each active
    variable uses lower_value / upper_value and optionally initial_value.

    A callback receives independent copies of the progress data, including the
    full config and turbine_results. It runs once at the initial point and at
    distinct PySolverView progress points. The default update_on='gradient'
    avoids sending finite-difference evaluations to plotting or app code.
    An optional logger receives initial-value clipping and native PSV output;
    console output is disabled for that solve when a logger is supplied.
    """
    problem = TurbineOptimizationProblem(config, optimization, evaluator, logger=logger)
    options = {
        "library": "scipy", "method": "slsqp", "problem_scale": 1.0,
        "tolerance": 1e-9, "max_iterations": 150,
        "derivative_method": "2-point", "derivative_abs_step": 1e-6,
        "print_convergence": True, "plot_convergence": False,
        "plot_scale_constraints": "log", "update_on": "gradient",
        "tolerance_check_cache": 0.0,
    }
    configured_options = problem.optimization.get("solver_options") or {}
    if not isinstance(configured_options, dict):
        raise ValueError("solver_options must be a dictionary.")
    if "callback_functions" in configured_options:
        raise ValueError("Pass application callbacks through optimize_turbine(callback=...).")
    options.update(configured_options)
    if logger is not None:
        options.update(logger=logger, print_convergence=False)

    initial = problem.evaluate(problem.x0)
    history = []
    previous_values = None

    def report_progress(physical_values):
        nonlocal previous_values
        values = np.asarray(physical_values, dtype=float)
        # PSV may report the initial/final point twice. Filter roundoff-level
        # repeats while retaining genuine changes in the design.
        if previous_values is not None and np.all(
            np.abs(values - previous_values) <= 8 * np.finfo(float).eps * problem.bound_range
        ):
            return
        evaluation = problem.evaluate(values)
        entry = {
            "iteration": len(history),
            "objective": evaluation["objective"],
            "max_constraint_violation": evaluation["max_constraint_violation"],
            "design_variables": dict(evaluation["design_variables"]),
        }
        history.append(entry)
        previous_values = values.copy()
        if callback is not None:
            state = deepcopy(entry)
            state["config"] = deepcopy(evaluation["config"])
            state["turbine_results"] = deepcopy(evaluation["turbine_results"])
            callback(state)

    def solver_callback(normalized_values, update_number):
        # Finite differences can leave the problem at a perturbed point.
        # Retrieve this callback's exact design rather than that last probe.
        physical_values = problem.scale_normalized_to_physical(normalized_values)
        report_progress(physical_values)

    solver = psv.OptimizationSolver(problem, callback_functions=[solver_callback], **options)
    report_progress(problem.x0)
    physical_final_values = solver.solve(problem.x0)
    final = problem.evaluate(physical_final_values)
    report_progress(physical_final_values)

    # A saved optimized case should use its final inputs as the next baseline.
    # Preserve the original overrides in source_config and initial_config.
    final_config = deepcopy(final["config"])
    for variable in final_config["optimization"]["design_variables"].values():
        variable.pop("initial_value", None)

    feasible = final["max_constraint_violation"] <= problem.feasibility_tolerance
    solver_success = bool(solver.success)
    message = str(solver.message)
    if not feasible:
        message += (
            f" Maximum normalized constraint violation {final['max_constraint_violation']:.3g}"
            f" exceeds {problem.feasibility_tolerance:.3g}."
        )
    return OptimizationResult(
        config=final_config, turbine_results=final["turbine_results"],
        geometry=[stage["geometry_meanline"] for stage in final["turbine_results"]["stages_performance"]],
        design_variables=final["design_variables"], objective=final["objective"],
        initial_objective=initial["objective"], constraint_report=final["constraint_report"],
        success=solver_success and feasible, solver_success=solver_success, feasible=bool(feasible),
        message=message, max_constraint_violation=final["max_constraint_violation"],
        history=history, solver=solver, n_evaluations=problem.n_evaluations,
        initial_turbine_results=initial["turbine_results"], initial_config=initial["config"],
        source_config=problem.source_config,
    )


# =============================================================================
# Turbine and convergence plots
# =============================================================================


def plot_turbine(turbine_results, title=None, figure=None):
    """Draw meridional and blade views, optionally reusing an existing figure."""
    from turbodash.visualization import matplotlib_turbine as plotting

    if figure is None:
        figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    else:
        if len(figure.axes) != 2:
            raise ValueError("A turbine figure must have exactly two axes.")
        axes = figure.axes
        for axis in axes:
            axis.clear()
    plotting.plot_turbine_meridional_channel(turbine_results, ax=axes[0])
    plotting.plot_turbine_blades(turbine_results, ax=axes[1], N_points=101)
    axes[0].set_title("Meridional channel")
    axes[1].set_title("Blade geometry")
    efficiency = turbine_results["overall_performance"]["efficiency_ts"]
    figure.suptitle(f"{title or 'Turbine'} — total-to-static efficiency {efficiency:.2%}")
    figure.tight_layout(rect=(0, 0, 1, 0.91))
    return figure


def plot_convergence(result, show=False):
    """Use PySolverView's native objective and constraint convergence figure."""
    figure = result.solver.plot_convergence_history(showfig=show)
    figure.axes[0].set_ylabel("Minimized objective (signed and normalized)")
    return figure


def save_figure(figure, filename, formats=("png", "svg")):
    """Save the same figure in the requested formats, without changing it."""
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    for extension in formats:
        figure.savefig(filename.with_suffix(f".{extension}"), dpi=180, bbox_inches="tight")


class TurbinePlotCallback:
    """Save the initial turbine and optionally refresh a live progress figure."""

    def __init__(self, output_directory, *, show=True, plot_each_iteration=False):
        self.output_directory = Path(output_directory)
        self.show = show
        self.plot_each_iteration = plot_each_iteration
        self.initial_figure = None
        self.progress_figure = None

    def __call__(self, state):
        results = state["turbine_results"]
        if state["iteration"] == 0:
            self.initial_figure = plot_turbine(results, title="Initial turbine")
            save_figure(self.initial_figure, self.output_directory / "initial_turbine")
            if self.show:
                plt.show(block=False)
                self.initial_figure.canvas.draw_idle()
                self.initial_figure.canvas.flush_events()
                plt.pause(0.01)
            else:
                plt.close(self.initial_figure)
        if self.plot_each_iteration:
            self.progress_figure = plot_turbine(
                results, title=f"Turbine — optimization update {state['iteration']}",
                figure=self.progress_figure,
            )
            if self.show:
                self.progress_figure.canvas.draw_idle()
                plt.show(block=False)
                plt.pause(0.01)
            else:
                plt.close(self.progress_figure)


# =============================================================================
# Reproducible YAML inputs and outputs
# =============================================================================


def _plain(value):
    """Convert NumPy values to the ordinary Python types accepted by safe YAML."""
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if callable(value):
        raise TypeError("YAML export requires path selectors; Python callables are not serializable.")
    return value


def save_optimization_results(result, output_directory):
    """Save input, effective initial/final cases, and traceable results as YAML."""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    documents = {
        "input_case.yaml": result.source_config,
        "initial_case.yaml": result.initial_config,
        "optimized_case.yaml": result.config,
        "optimization_result.yaml": {
            "provenance": {
                "input_case": "input_case.yaml",
                "effective_initial_case": "initial_case.yaml",
                "optimized_case": "optimized_case.yaml",
                "solver": "pysolver_view.OptimizationSolver",
                "library": result.solver.library, "method": result.solver.method,
            },
            "success": result.success, "solver_success": result.solver_success,
            "feasible": result.feasible, "message": result.message,
            "initial_objective": result.initial_objective, "objective": result.objective,
            "design_variables": result.design_variables,
            "max_normalized_constraint_violation": result.max_constraint_violation,
            "constraint_report": result.constraint_report,
            "model_evaluations": result.n_evaluations,
            "solver_elapsed_seconds": result.solver.elapsed_time,
            "progress_history": result.history,
            "solver_convergence_history": result.solver.convergence_history,
            "turbine_results": result.turbine_results,
        },
    }
    paths = {}
    for name, document in documents.items():
        path = output_directory / name
        path.write_text(dump_turbine_yaml(_plain(document)), encoding="utf-8")
        paths[name] = path
    return paths
