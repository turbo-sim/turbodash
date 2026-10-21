"""Assess first-order constrained optimality at the point actually reported.

Native SLSQP termination combines several stopping tests. Its success flag or
a small step alone does not establish that each KKT residual is below a user's
chosen threshold, especially when trial steps are deliberately shortened. This
module separately checks primal feasibility, stationarity of the Lagrangian,
multiplier signs (dual feasibility), and complementarity, including variable
bounds. The driver uses the diagnostics for reporting and final acceptance.

``assess_kkt`` estimates multipliers at the supplied point with a small linear
program, using all inequalities and finite bounds rather than selecting an
active set through an arbitrary cutoff. It then recomputes the residuals from
those multipliers and the supplied values and derivatives. Separate feasibility
and optimality tolerances make acceptance explicit; native ``ftol`` is not
treated as a component-wise KKT tolerance.

For h(x)=0, c(x)>=0 and l<=x<=u the sign convention is
L = f - lambda*h - mu*c - z_lower*(x-l) - z_upper*(u-x).
Equality multipliers are free; all other multipliers are nonnegative.

All diagnostics use the caller's objective, constraint, and coordinate scaling.
Their quality depends on derivative accuracy and numerical conditioning. A
passing check is numerical evidence of first-order KKT conditions, not proof of
a local minimum, second-order sufficiency, or global optimality. The usual
constraint qualifications are needed when interpreting KKT as a necessary
condition at a constrained minimum. This module has no turbine or app dependency.
"""

import numpy as np
from scipy.optimize import linprog


def assess_kkt(
    x,
    gradient,
    equality_values,
    equality_jacobian,
    inequality_values,
    inequality_jacobian,
    lower,
    upper,
    *,
    feasibility_tol=1e-6,
    optimality_tol=1e-4,
):
    """Estimate multipliers and return independently recomputed KKT residuals.

    Minimize the infinity norm of stationarity and multiplier-times-slack
    residuals, using ALL inequalities and finite bounds. The resulting LP
    enforces multiplier signs and needs no arbitrary active-set tolerance.
    Columns are equilibrated to preserve positive constraint rescaling.

    Residuals use the supplied Jacobian at x, in the exact objective, variable,
    and constraint scaling of the optimization problem. ``ftol`` is deliberately
    not used: native SLSQP stopping tests do not guarantee a KKT residual bound.
    ``passed`` requires both the independent feasibility and optimality limits.
    """
    x = np.asarray(x, dtype=float)
    gradient = np.asarray(gradient, dtype=float)
    lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
    h, c = np.asarray(equality_values, dtype=float), np.asarray(
        inequality_values, dtype=float
    )
    jh, jc = np.asarray(equality_jacobian, dtype=float), np.asarray(
        inequality_jacobian, dtype=float
    )
    if (
        x.ndim != 1
        or not x.size
        or gradient.shape != x.shape
        or lower.shape != x.shape
        or upper.shape != x.shape
    ):
        raise ValueError(
            "KKT coordinates, gradient, and bounds must be equally sized vectors."
        )
    n = x.size
    if h.ndim != 1 or c.ndim != 1 or jh.shape != (h.size, n) or jc.shape != (c.size, n):
        raise ValueError(
            "KKT constraint values and Jacobians have inconsistent shapes."
        )
    if not all(np.all(np.isfinite(a)) for a in (x, gradient, h, c, jh, jc)):
        raise ValueError("KKT values and derivatives must be finite.")
    if (
        np.any(np.isnan(lower))
        or np.any(np.isnan(upper))
        or np.any(lower > upper)
        or np.any(lower == np.inf)
        or np.any(upper == -np.inf)
    ):
        raise ValueError("Invalid KKT bounds.")
    if not all(np.isfinite(t) and t > 0 for t in (feasibility_tol, optimality_tol)):
        raise ValueError("KKT tolerances must be finite and positive.")

    finite_lower, finite_upper = np.isfinite(lower), np.isfinite(upper)
    lower_slack, upper_slack = (
        x[finite_lower] - lower[finite_lower],
        upper[finite_upper] - x[finite_upper],
    )
    normals = np.vstack((jh, jc, np.eye(n)[finite_lower], -np.eye(n)[finite_upper])).T
    slacks = np.r_[np.zeros(h.size), c, lower_slack, upper_slack]
    primal = max(
        float(np.max(np.abs(h), initial=0)),
        float(np.max(-c, initial=0)),
        float(np.max(-lower_slack, initial=0)),
        float(np.max(-upper_slack, initial=0)),
    )
    base = dict(
        method="pointwise_minimax",
        primal_feasibility=primal,
        feasibility_tol=float(feasibility_tol),
        optimality_tol=float(optimality_tol),
    )

    def unavailable(message):
        return dict(
            base,
            available=False,
            passed=False,
            message=message,
            stationarity=None,
            dual_feasibility=None,
            complementarity=None,
            optimality=None,
            kkt_residual=None,
            kkt_ratio=None,
            multipliers=None,
        )

    count = slacks.size
    multipliers = np.zeros(count)
    if count:
        column_scale = np.maximum(np.max(np.abs(normals), axis=0), np.abs(slacks))
        retained = column_scale > 0
        if np.any(retained):
            matrix = np.vstack(
                (
                    normals[:, retained] / column_scale[retained],
                    np.diag(slacks[retained] / column_scale[retained]),
                )
            )
            rhs = np.r_[gradient, np.zeros(np.count_nonzero(retained))]
            # Rescale unknown multipliers and t with the gradient magnitude as
            # well, so HiGHS' absolute feasibility tolerance cannot swallow a
            # small but meaningful gradient. Residuals below are never rounded.
            residual_scale = float(np.max(np.abs(gradient), initial=0)) or 1.0
            rhs /= residual_scale
            bounds = [
                (None, None) if index < h.size else (0, None)
                for index in np.flatnonzero(retained)
            ] + [(0, None)]
            objective = np.r_[np.zeros(matrix.shape[1]), 1.0]
            inequalities = np.vstack(
                (
                    np.column_stack((matrix, -np.ones(matrix.shape[0]))),
                    np.column_stack((-matrix, -np.ones(matrix.shape[0]))),
                )
            )
            fit = linprog(
                objective,
                A_ub=inequalities,
                b_ub=np.r_[rhs, -rhs],
                bounds=bounds,
                method="highs-ds",
                options={
                    "primal_feasibility_tolerance": 1e-10,
                    "dual_feasibility_tolerance": 1e-10,
                },
            )
            if not fit.success:
                return unavailable(f"Multiplier estimation failed: {fit.message}")
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                multipliers[retained] = (
                    fit.x[:-1] * residual_scale / column_scale[retained]
                )
            if not np.all(np.isfinite(multipliers)):
                return unavailable(
                    "Estimated multipliers exceed finite numerical range."
                )

    # Recompute with original coefficients, independently of LP slack/status.
    with np.errstate(over="ignore", invalid="ignore"):
        lagrangian_gradient = gradient - normals @ multipliers
        products = slacks * multipliers
    if not np.all(np.isfinite(lagrangian_gradient)) or not np.all(
        np.isfinite(products)
    ):
        return unavailable("Nonfinite KKT residual after multiplier reconstruction.")
    stationarity = float(np.max(np.abs(lagrangian_gradient), initial=0))
    complementarity = float(np.max(np.abs(products), initial=0))
    dual = max(0.0, float(np.max(-multipliers[h.size :], initial=0)))
    optimality = max(stationarity, complementarity, dual)
    ratio = max(primal / feasibility_tol, optimality / optimality_tol)
    offset = h.size + c.size
    bound_lower, bound_upper = np.zeros(n), np.zeros(n)
    bound_lower[finite_lower] = multipliers[offset : offset + lower_slack.size]
    bound_upper[finite_upper] = multipliers[offset + lower_slack.size :]
    return dict(
        base,
        available=True,
        passed=bool(ratio <= 1),
        message="",
        stationarity=stationarity,
        primal_feasibility=primal,
        dual_feasibility=dual,
        complementarity=complementarity,
        optimality=optimality,
        kkt_residual=max(primal, optimality),
        kkt_ratio=ratio,
        lagrangian_gradient=lagrangian_gradient,
        multipliers=dict(
            equality=multipliers[: h.size],
            inequality=multipliers[h.size : offset],
            lower=bound_lower,
            upper=bound_upper,
        ),
    )
