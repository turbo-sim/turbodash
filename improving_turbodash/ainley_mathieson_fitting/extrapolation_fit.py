"""Refine the selected explicit AM expression for ordered low-spacing tails.

Only the rational_monotone family is supported.  Its degree-three polynomial
G(x) = angle_scale * x * dY/dalpha makes continuous spacing constraints possible:
checking both endpoints and the real stationary points checks the whole interval.
Angle is checked on successively finer meshes and by an independent dense audit.
This module fits constants; the exported evaluator needs neither it nor SciPy.
"""
import numpy as np
from scipy.optimize import minimize

from fit_models import (angle_derivative, angle_scale, blocks, complex_jacobian,
                        evaluate, fit, softplus)

ORDER_SPACING_RANGE = (0., .5)
ORDER_DERIVATIVE_MARGIN = 2e-7
# The derivative margin is multiplied by x and vanishes at zero.  This separate
# tiny margin protects the polynomial's zero-spacing endpoint from roundoff.
POLYNOMIAL_MARGIN = 1e-10


def _require_supported(spec):
    if spec.form != "rational_monotone":
        raise ValueError("Bounded extrapolation refinement supports rational_monotone only")


def scaled_angle_polynomial(spec, parameters, alpha, family):
    """Ascending coefficients of G(x)=scale*x*dY/dalpha, with no 1/x term."""
    _require_supported(spec)
    center, scale = angle_scale(family)
    t = (np.asarray(alpha)-center)/scale
    coeff = blocks(parameters, spec.degrees)
    pp = [np.polynomial.polynomial.polyval(t, c) for c in coeff]
    dd = [np.polynomial.polynomial.polyval(t, np.polynomial.polynomial.polyder(c))
          for c in coeff]
    q, A, B = [np.exp(v) for v in pp[1:]]
    qp, Ap, Bp = [v*d for v, d in zip((q, A, B), dd[1:])]
    _, log_delta, tc, log_w = coeff[0]
    mp = np.exp(log_delta-softplus(-(t-tc)/np.exp(log_w)))
    return np.array((Bp*q*q+2*B*q*qp,
                     mp+Ap*q*q-2*Bp*q+2*A*q*qp-2*B*qp,
                     Bp-2*Ap*q-2*A*qp,
                     Ap))


def polynomial_extrema(coefficients):
    """Values and locations at endpoints and stationary points on [0,.5].

    The quadratic formula uses its cancellation-resistant form. Degenerate
    quadratic/linear derivatives, repeated roots and negative discriminants
    have explicit fallbacks. Roots outside the interval duplicate an endpoint.
    Real-part branch decisions preserve complex-step parameter derivatives.
    """
    c0, c1, c2, c3 = np.asarray(coefficients)
    a, b, c = 3*c3, 2*c2, c1
    magnitude = np.maximum.reduce((abs(np.real(a)), abs(np.real(b)),
                                   abs(np.real(c)), np.full_like(np.real(a), 1e-30)))
    quadratic = abs(np.real(a)) > 1e-13*magnitude
    linear = abs(np.real(b)) > 1e-13*magnitude
    discriminant = b*b-4*a*c
    real_roots = np.real(discriminant) >= 0
    root = np.sqrt(np.where(real_roots, discriminant, 1.))
    sign_b = np.where(np.real(b) >= 0, 1., -1.)
    q = -.5*(b+sign_b*root)
    safe_a = np.where(quadratic, a, 1.)
    safe_q = np.where(abs(np.real(q)) > 1e-30, q, 1.)
    linear_root = np.where(linear, -c/np.where(linear, b, 1.), 0.)
    roots = (q/safe_a, c/safe_q)
    locations = [np.zeros_like(c0), np.full_like(c0, ORDER_SPACING_RANGE[1])]
    for location in roots:
        location = np.where(quadratic, np.where(real_roots, location, 0.), linear_root)
        location = np.where(np.real(location) < 0., 0., location)
        location = np.where(np.real(location) > ORDER_SPACING_RANGE[1],
                            ORDER_SPACING_RANGE[1], location)
        locations.append(location)
    locations = np.array(locations)
    values = c0+locations*(c1+locations*(c2+locations*c3))
    return values, locations


def _constraint_polynomial(spec, parameters, alpha, family):
    coefficients = scaled_angle_polynomial(spec, parameters, alpha, family)
    coefficients[0] -= POLYNOMIAL_MARGIN
    coefficients[1] -= angle_scale(family)[1]*ORDER_DERIVATIVE_MARGIN
    return coefficients


def refine(spec, family, points, initial=None):
    """Return an optimized SciPy result with low-spacing ordering diagnostics.

    With no initial coefficients, initialize from the supplied training points
    only. Holdout folds must use this route to prevent full-data initialization
    from entering validation. No interpolation or saturation is fitted here;
    the explicit rational expression is unchanged apart from its constants.
    """
    _require_supported(spec)
    x, alpha, y, sigma = [np.array([p[k] for p in points]) for k in
                         ("pitch_chord", "alpha_deg", "yp", "sigma_yp")]
    initialization = None
    if initial is None:
        initialization = fit(spec, family, points)
        parameters = initialization.x.copy()
    else:
        parameters = np.asarray(initial, dtype=float).copy()
    center, scale = angle_scale(family)

    def residual(p):
        return (evaluate(spec, p, x, alpha, family)-y)/sigma

    def objective(p):
        r = residual(p)
        return .5*np.mean(r*r)

    def gradient(p):
        return complex_jacobian(residual, p).T @ residual(p)/len(y)

    attempts = []

    def optimize(p, angles, adaptive=False):
        def inequality(q):
            values, _ = polynomial_extrema(_constraint_polynomial(spec, q, angles, family))
            return values.ravel()/.01
        result = minimize(objective, p, jac=gradient, method="SLSQP",
                          bounds=[(-16., 16.)]*len(p),
                          constraints={"type": "ineq", "fun": inequality,
                                       "jac": lambda q: complex_jacobian(inequality, q)},
                          options={"maxiter": 500, "ftol": 1e-10})
        attempts.append(dict(success=bool(result.success), status=int(result.status),
                             message=str(result.message), iterations=int(result.nit),
                             nfev=int(result.nfev), objective=float(result.fun),
                             angle_nodes=int(len(angles)), adaptive=adaptive,
                             minimum_constraint_G=float(np.min(inequality(result.x))*.01)))
        return result

    for resolution in (81, 241, 1001):
        constraint_angles = np.linspace(center-scale, center+scale, resolution)
        result = optimize(parameters, constraint_angles)
        parameters = result.x

    dense_angles = np.linspace(center-scale, center+scale, 4001)
    for _ in range(6):
        values, _ = polynomial_extrema(_constraint_polynomial(spec, parameters, dense_angles, family))
        minima = np.min(values, axis=0)
        raw_values, _ = polynomial_extrema(scaled_angle_polynomial(spec, parameters, dense_angles, family))
        # Small negative solver residuals are acceptable inside the deliberate
        # positive margin; actual ordering must remain nonnegative.
        if np.min(minima) >= -POLYNOMIAL_MARGIN*.1 and np.min(raw_values) >= 0:
            break
        worst = np.argsort(minima)[:16]
        constraint_angles = np.unique(np.r_[constraint_angles, dense_angles[worst]])
        result = optimize(parameters, constraint_angles, adaptive=True)
        parameters = result.x

    coarse_values, _ = polynomial_extrema(scaled_angle_polynomial(
        spec, parameters, np.linspace(center-scale, center+scale, 241), family))
    dense_values, _ = polynomial_extrema(scaled_angle_polynomial(spec, parameters, dense_angles, family))
    dense_constraints, _ = polynomial_extrema(_constraint_polynomial(spec, parameters, dense_angles, family))
    xx, aa = np.meshgrid(np.linspace(1e-5, .5, 1001),
                         np.linspace(center-scale, center+scale, 2001))
    minimum_derivative = float(np.min(angle_derivative(spec, parameters, xx, aa, family)))
    feasible = (np.min(dense_values) >= 0 and
                minimum_derivative >= ORDER_DERIVATIVE_MARGIN*.5 and
                np.min(dense_constraints) >= -POLYNOMIAL_MARGIN*.1)
    result.x = parameters
    result.success = bool(result.success and feasible)
    result.attempts = attempts
    result.order_check = dict(
        method="Cubic G endpoints and stationary roots: continuous spacing; dense/adaptive angle mesh",
        spacing_range=ORDER_SPACING_RANGE,
        angle_range=(center-scale, center+scale),
        minimum_angle_derivative=minimum_derivative,
        derivative_check_spacing_range=(1e-5, .5),
        angle_derivative_margin=ORDER_DERIVATIVE_MARGIN,
        polynomial_endpoint_margin=POLYNOMIAL_MARGIN,
        coarse_G_minimum=float(np.min(coarse_values)),
        dense_G_minimum=float(np.min(dense_values)),
        dense_constraint_G_minimum=float(np.min(dense_constraints)),
        dense_angle_nodes=len(dense_angles),
        continuous_spacing_order_verified=bool(np.min(dense_values) >= 0),
        angle_verification="Numerical dense/adaptive verification, not an interval proof in angle",
        optimizer_success=bool(attempts[-1]["success"]),
        optimizer_status=int(attempts[-1]["status"]),
        optimizer_iterations=sum(a["iterations"] for a in attempts),
        constrained_optimizations=attempts,
        training_only_initialization=initial is None,
        initialization_success=None if initialization is None else bool(initialization.success))
    return result
