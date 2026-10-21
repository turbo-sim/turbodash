"""Independent audit of production turbine SLSQP reports.

This module deliberately does not call ``assess_kkt``. It recomputes physical
constraint values, uses the exported multipliers to reconstruct all residuals,
and obtains a fresh final Jacobian from separate model evaluations. A half-step
finite-difference estimate measures sensitivity; it is informative, not another
pass/fail gate on the optimizer's configured derivative scheme.
"""

from __future__ import annotations

import numpy as np

from turbodash.core.errors import ModelDomainError
from turbodash.optimization.turbine import TurbineOptimizationProblem, read_path


def _maximum(values):
    return max(0.0, float(np.max(np.asarray(values, dtype=float), initial=0.0)))


def _norm(values):
    return _maximum(np.abs(values))


def _selected(results, selector):
    return np.asarray(
        selector(results) if callable(selector) else read_path(results, selector),
        dtype=float,
    )


def _sample(problem, x):
    """Independently assemble minimized objective and h=0 / c>=0 residuals."""
    physical = problem.lower_bounds + np.asarray(x) * problem.bound_range
    evaluated = problem.evaluate(physical)
    results = evaluated["turbine_results"]
    objective = float(_selected(results, problem.objective_spec["variable"]).item())
    objective *= problem.objective_sign / problem.objective_scale
    equalities, inequalities = [], []
    for spec in problem.constraint_specs:
        current = _selected(results, spec["variable"])
        target = spec["value"]
        target = (
            _selected(results, target)
            if isinstance(target, str) or callable(target)
            else np.asarray(target, dtype=float)
        )
        normalization = spec.get("normalize", False)
        scale = (
            (np.abs(target) if normalization else 1.0)
            if isinstance(normalization, (bool, np.bool_))
            else normalization
        )
        current, target, scale = np.broadcast_arrays(current, target, scale)
        operator = spec["type"]
        if operator in ("=", "=="):
            equalities.extend(((current - target) / scale).ravel())
        elif operator in ("<", "<="):
            inequalities.extend(((target - current) / scale).ravel())
        elif operator in (">", ">="):
            inequalities.extend(((current - target) / scale).ravel())
        else:
            raise ValueError(f"Unrecognized constraint operator {operator!r}.")
    return (
        np.r_[objective, equalities, inequalities],
        np.asarray(equalities),
        np.asarray(inequalities),
        evaluated,
    )


def _jacobian(problem, x, base, options, step_factor=1.0):
    """Fresh bounded one-sided differences without physical-domain retries.

    Choose each probe independently from the configured step and variable bounds.
    A failed physical probe is unavailable, matching the current solver contract;
    neither its direction nor its step is changed after evaluation.
    """
    requested = (
        float(options.get("finite_diff_step", options.get("derivative_abs_step", 1e-6)))
        * step_factor
    )
    matrix = np.empty((base.size, x.size))
    used_steps = np.empty(x.size)
    rejections = 0
    for column in range(x.size):
        forward, backward = 1.0 - x[column], x[column]
        direction = 1 if forward >= requested or forward >= backward else -1
        room = forward if direction == 1 else backward
        probe = x.copy()
        probe[column] = np.clip(
            x[column] + direction * min(requested, room), 0.0, 1.0
        )
        step = probe[column] - x[column]
        if abs(step) <= 8 * np.finfo(float).eps * max(1.0, abs(x[column])):
            raise RuntimeError(f"No representable audit derivative for variable {column}.")
        try:
            sampled, *_ = _sample(problem, probe)
        except ModelDomainError as error:
            raise RuntimeError(
                f"Audit derivative probe failed for variable {column}; probes are not retried: {error}"
            ) from error
        matrix[:, column] = (sampled - base) / step
        used_steps[column] = step
    return matrix, used_steps, rejections


def _residuals(x, h, c, multipliers, lagrangian_gradient):
    eq = np.asarray(multipliers["equality"], dtype=float)
    inequality = np.asarray(multipliers["inequality"], dtype=float)
    lower = np.asarray(multipliers["lower"], dtype=float)
    upper = np.asarray(multipliers["upper"], dtype=float)
    if eq.shape != h.shape or inequality.shape != c.shape:
        raise ValueError("Constraint multiplier dimensions do not match the case.")
    if lower.shape != x.shape or upper.shape != x.shape:
        raise ValueError("Bound multiplier dimensions do not match the design.")
    primal = max(_norm(h), _maximum(-c), _maximum(-x), _maximum(x - 1.0))
    stationarity = _norm(lagrangian_gradient)
    dual = _maximum(-np.r_[inequality, lower, upper])
    complementarity = _norm(np.r_[inequality * c, lower * x, upper * (1.0 - x)])
    optimality = max(stationarity, dual, complementarity)
    return dict(
        primal_feasibility=primal,
        stationarity=stationarity,
        dual_feasibility=dual,
        complementarity=complementarity,
        optimality=optimality,
        kkt_residual=max(primal, optimality),
    )


def _lagrangian(matrix, h, multipliers):
    return (
        matrix[0]
        - matrix[1 : 1 + h.size].T @ np.asarray(multipliers["equality"])
        - matrix[1 + h.size :].T @ np.asarray(multipliers["inequality"])
        - np.asarray(multipliers["lower"])
        + np.asarray(multipliers["upper"])
    )


def audit_result(result):
    """Return JSON-safe checks of final metrics, all history rows, and gates.

    Expected optimization failures are valid audit outcomes: ``passed`` means
    the numerical reporting and success/failure classification were consistent.
    This is a turbine-only checker and evaluates the production meanline model.
    """
    problem = TurbineOptimizationProblem(result.source_config)
    options = problem.optimization.get("solver_options", {}) or {}
    feasibility_tol = float(problem.optimization.get("feasibility_tolerance", 1e-6))
    optimality_tol = float(options.get("optimality_tol", 1e-4))
    solver = result.solver
    checks, errors = [], []

    def check(name, condition, detail=""):
        condition = bool(condition)
        checks.append(dict(name=name, passed=condition))
        if not condition:
            errors.append(f"{name}: {detail}" if detail else name)

    def close(name, actual, expected, *, atol=3e-11, rtol=3e-9):
        actual, expected = np.asarray(actual), np.asarray(expected)
        okay = actual.shape == expected.shape and np.allclose(
            actual, expected, atol=atol, rtol=rtol
        )
        check(name, okay, f"reported={actual}; reconstructed={expected}")

    def classified(metrics):
        metrics = dict(metrics)
        metrics["kkt_ratio"] = max(
            metrics["primal_feasibility"] / feasibility_tol,
            metrics["optimality"] / optimality_tol,
        )
        metrics["passed"] = metrics["kkt_ratio"] <= 1.0
        return metrics

    history = solver.history
    check("accepted history exists", bool(history))
    previous_x, previous_f = None, None
    sampled_history = []
    for number, entry in enumerate(history):
        prefix = f"history[{number}]"
        x = np.asarray(entry["x"], dtype=float)
        vector, h, c, physical = _sample(problem, x)
        sampled_history.append((x, vector, h, c, physical))
        close(f"{prefix} same-point objective", entry["objective"], vector[0])
        close(f"{prefix} independent equality signs/scales", physical["equalities"], h)
        close(
            f"{prefix} independent inequality signs/scales",
            -physical["inequalities"],
            c,
        )
        close(
            f"{prefix} actual step norm",
            entry["step_norm"],
            0.0 if previous_x is None else np.linalg.norm(x - previous_x),
        )
        close(
            f"{prefix} objective change",
            entry["objective_change"],
            0.0 if previous_f is None else vector[0] - previous_f,
        )
        previous_x, previous_f = x, vector[0]
        kkt = entry.get("kkt")
        if not entry.get("diagnostics_available"):
            check(
                f"{prefix} unavailable is not a pass",
                not kkt or not kkt.get("passed"),
            )
            continue
        metrics = classified(
            _residuals(x, h, c, kkt["multipliers"], kkt["lagrangian_gradient"])
        )
        for field, expected in metrics.items():
            if field == "passed":
                check(f"{prefix} pass gate", kkt[field] == expected)
            else:
                close(f"{prefix} {field}", kkt[field], expected)
                if field in entry:
                    close(f"{prefix} scalar {field}", entry[field], expected)
        close(
            f"{prefix} feasibility tolerance",
            kkt["feasibility_tol"],
            feasibility_tol,
            atol=0,
            rtol=0,
        )
        close(
            f"{prefix} optimality tolerance",
            kkt["optimality_tol"],
            optimality_tol,
            atol=0,
            rtol=0,
        )
        close(
            f"{prefix} reported violation",
            entry["max_constraint_violation"],
            metrics["primal_feasibility"],
        )
        close(
            f"{prefix} physical violation",
            physical["max_constraint_violation"],
            metrics["primal_feasibility"],
        )

    # The app/export histories carry physical objectives but solver-scaled KKT.
    check(
        "physical and solver histories have equal length",
        len(result.history) == len(history),
    )
    for number, entry in enumerate(result.history):
        values = np.asarray(
            [entry["design_variables"][name] for name in problem.variable_names]
        )
        x = (values - problem.lower_bounds) / problem.bound_range
        if number >= len(history):
            continue
        native, sample = history[number], sampled_history[number]
        check(
            f"physical history[{number}] iteration alignment",
            entry["iteration"] == native["iteration"],
        )
        check(
            f"physical history[{number}] matches accepted point",
            np.allclose(sample[0], x, atol=2e-14, rtol=0),
        )
        close(
            f"physical history[{number}] objective",
            entry["objective"],
            sample[-1]["objective"],
        )
        for field in (
            "step_norm",
            "objective_change",
            "optimality",
            "stationarity",
            "complementarity",
            "dual_feasibility",
            "kkt_residual",
            "kkt_ratio",
        ):
            if native.get(field) is not None:
                close(
                    f"physical history[{number}] {field}", entry[field], native[field]
                )

    x = np.asarray(solver.x, dtype=float)
    vector, h, c, evaluated = _sample(problem, x)
    primal = max(_norm(h), _maximum(-c), _maximum(-x), _maximum(x - 1.0))
    close("final same-point objective", solver.fun, vector[0])
    close("final physical objective", result.objective, evaluated["objective"])
    check(
        "final constraint report component count",
        len(result.constraint_report) == len(evaluated["constraint_report"]),
    )
    for index, (saved, fresh) in enumerate(
        zip(result.constraint_report, evaluated["constraint_report"])
    ):
        for field in (
            "value",
            "target",
            "residual",
            "normalized_residual",
            "violation",
            "normalized_violation",
        ):
            close(
                f"final constraint report[{index}] {field}", saved[field], fresh[field]
            )
    close("final normalized violation", solver.max_constraint_violation, primal)
    close(
        "final physical normalized violation", result.max_constraint_violation, primal
    )
    physical_report_violation = max(
        (item["normalized_violation"] for item in result.constraint_report),
        default=0.0,
    )
    close(
        "final constraint report violation",
        max(physical_report_violation, _maximum(-x), _maximum(x - 1.0)),
        primal,
    )
    check(
        "final solver feasibility gate", solver.feasible == (primal <= feasibility_tol)
    )
    check("final app feasibility gate", result.feasible == (primal <= feasibility_tol))
    final, sensitivity = {}, {}
    if solver.diagnostics_available:
        matrix, steps, rejected = _jacobian(problem, x, vector, options)
        multipliers = solver.kkt_multipliers
        lagrangian = _lagrangian(matrix, h, multipliers)
        final = classified(_residuals(x, h, c, multipliers, lagrangian))
        close(
            "final fresh objective gradient",
            solver.jac,
            matrix[0],
            atol=1e-9,
            rtol=2e-7,
        )
        close(
            "final fresh Lagrangian gradient",
            solver.kkt["lagrangian_gradient"],
            lagrangian,
            atol=2e-9,
            rtol=2e-7,
        )
        for field, expected in final.items():
            if field == "passed":
                check("final KKT pass gate", solver.kkt["passed"] == expected)
            else:
                (
                    close(
                        f"final fresh {field}",
                        solver[field],
                        expected,
                        atol=2e-9,
                        rtol=2e-7,
                    )
                    if field != "primal_feasibility"
                    else close(
                        "final fresh primal feasibility", solver.kkt[field], expected
                    )
                )
        stationary = final["optimality"] <= optimality_tol
        check("final stationarity gate", solver.stationary == stationary)
        final["finite_difference_steps"] = steps.tolist()
        final["derivative_domain_rejections"] = rejected
        try:
            alternate, alternate_steps, rejected = _jacobian(
                problem, x, vector, options, step_factor=0.5
            )
            alternate_lagrangian = _lagrangian(alternate, h, multipliers)
            half = _residuals(x, h, c, multipliers, alternate_lagrangian)
            # Richardson extrapolation is second order only where both schemes
            # sampled the same side at exactly half the physical displacement.
            proportional = np.isclose(
                alternate_steps, steps * 0.5, rtol=1e-6, atol=1e-16
            )
            richardson = np.where(
                proportional[None, :], 2 * alternate - matrix, alternate
            )
            richardson_lagrangian = _lagrangian(richardson, h, multipliers)
            sensitivity = dict(
                available=True,
                method="half-step one-sided differences; Richardson on matching half-steps",
                baseline_objective_gradient=matrix[0].tolist(),
                half_step_objective_gradient=alternate[0].tolist(),
                max_gradient_change=_norm(alternate[0] - matrix[0]),
                max_jacobian_change=_norm(alternate - matrix),
                half_step_stationarity=half["stationarity"],
                half_step_optimality=half["optimality"],
                half_step_kkt_ratio=max(
                    primal / feasibility_tol, half["optimality"] / optimality_tol
                ),
                richardson_stationarity=_norm(richardson_lagrangian),
                richardson_columns=int(np.count_nonzero(proportional)),
                finite_difference_steps=alternate_steps.tolist(),
                derivative_domain_rejections=rejected,
                note="Uses the reported multipliers without refitting; sensitivity is not a separate pass/fail test.",
            )
        except RuntimeError as error:
            sensitivity = dict(available=False, message=str(error))
    else:
        stationary = False
        check(
            "unavailable final diagnostics do not pass",
            not solver.stationary and not solver.success,
        )
    check(
        "native success matches status", solver.solver_success == (solver.status == 0)
    )
    expected_success = solver.status == 0 and primal <= feasibility_tol and stationary
    check("final solver success gate", solver.success == expected_success)
    check("final app success gate", result.success == expected_success)
    return dict(
        passed=not errors,
        checks=checks,
        errors=errors,
        history_records_checked=len(history),
        physical_history_records_checked=len(result.history),
        audit_model_evaluations=problem.n_evaluations,
        feasibility_tol=feasibility_tol,
        optimality_tol=optimality_tol,
        final=final,
        derivative_sensitivity=sensitivity,
    )
