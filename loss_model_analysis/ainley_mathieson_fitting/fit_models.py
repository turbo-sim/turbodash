"""Fit explicit smooth functions to independently digitized AM chart points.

No model in this file interpolates a table at evaluation time. Angle polynomial
coefficients are fitted jointly to every selected pixel measurement.
"""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import least_squares, minimize

ORDER_SPACING_RANGE = (.30, .50)
ORDER_DERIVATIVE_MARGIN = 2e-7  # Yp per degree; deliberately much below scan resolution


@dataclass(frozen=True)
class Specification:
    name: str
    form: str
    degrees: tuple

    @property
    def count(self):
        return sum(d + 1 for d in self.degrees)


SPECIFICATIONS = (
    Specification("quadratic_233", "quadratic", (2, 3, 3)),
    Specification("rational_2222", "rational", (2, 2, 2, 2)),
    Specification("rational_3222", "rational", (3, 2, 2, 2)),
    Specification("rational_3322", "rational", (3, 3, 2, 2)),
    Specification("rational_3333", "rational", (3, 3, 3, 3)),
    Specification("rational_4222", "rational", (4, 2, 2, 2)),
    Specification("rational_softplus222", "rational_softplus", (4, 2, 2, 2)),
    Specification("rational_monotone222", "rational_monotone", (3, 2, 2, 2)),
    Specification("quartic_32222", "quartic", (3, 2, 2, 2, 2)),
)


def angle_scale(family):
    return (60., 20.) if family == "nozzle" else (55., 15.)


def blocks(parameters, degrees):
    result, start = [], 0
    for degree in degrees:
        result.append(parameters[start:start + degree + 1])
        start += degree + 1
    return result


def softplus(z):
    """Stable log(1+exp(z)), also supporting complex-step differentiation."""
    h = np.maximum(np.real(z), 0.)
    return h + np.log(np.exp(-h)+np.exp(z-h))


def evaluate(spec, parameters, x, alpha, family, derivative=0):
    center, scale = angle_scale(family)
    t = (np.asarray(alpha) - center) / scale
    x, t = np.broadcast_arrays(np.asarray(x), t)
    polynomials = [np.polynomial.polynomial.polyval(t, p)
                   for p in blocks(parameters, spec.degrees)]
    m, x0, A = [np.exp(p) for p in polynomials[:3]]
    if spec.form == "rational_softplus":
        b0, b1, log_delta, tc, log_w = blocks(parameters, spec.degrees)[0]
        w = np.exp(log_w)
        m = np.exp(b0+b1*t)+np.exp(log_delta)*w*softplus((t-tc)/w)
    elif spec.form == "rational_monotone":
        log_m0, log_delta, tc, log_w = blocks(parameters, spec.degrees)[0]
        w = np.exp(log_w)
        m = np.exp(log_m0)+np.exp(log_delta)*w*softplus((t-tc)/w)
    d = x - x0
    if spec.form == "quadratic":
        return (m + A*d*d, 2*A*d, 2*A)[derivative]
    if spec.form.startswith("rational"):
        B = np.exp(polynomials[3])
        return (m + (A+B/x)*d*d,
                2*A*d + B*(1-x0*x0/(x*x)),
                2*A + 2*B*x0*x0/(x*x*x))[derivative]
    D = np.exp(polynomials[3])
    C = np.sqrt(8*A*D/3)*np.tanh(polynomials[4])
    return (m+A*d*d+C*d**3+D*d**4,
            2*A*d+3*C*d*d+4*D*d**3,
            2*A+6*C*d+12*D*d*d)[derivative]


def angle_derivative(spec, parameters, x, alpha, family):
    """Exact angle derivative, per degree, including all coefficient changes."""
    center, scale = angle_scale(family)
    x, t = np.broadcast_arrays(np.asarray(x), (np.asarray(alpha)-center)/scale)
    coeff = blocks(parameters, spec.degrees)
    pp = [np.polynomial.polynomial.polyval(t, c) for c in coeff]
    dd = [np.polynomial.polynomial.polyval(t, np.polynomial.polynomial.polyder(c))
          for c in coeff]
    m, x0, A = [np.exp(p) for p in pp[:3]]
    mp, x0p, Ap = [v*q for v, q in zip((m, x0, A), dd[:3])]
    if spec.form in ("rational_softplus", "rational_monotone"):
        if spec.form == "rational_softplus":
            b0, b1, log_delta, tc, log_w = coeff[0]
            baseline_slope = b1*np.exp(b0+b1*t)
        else:
            log_m0, log_delta, tc, log_w = coeff[0]
            baseline_slope = 0.
        z = (t-tc)/np.exp(log_w)
        sigmoid = np.exp(-softplus(-z))
        mp = baseline_slope+np.exp(log_delta)*sigmoid
    d = x-x0
    if spec.form == "quadratic":
        return (mp+Ap*d*d-2*A*d*x0p)/scale
    if spec.form.startswith("rational"):
        B = np.exp(pp[3]); Bp = B*dd[3]
        return (mp+(Ap+Bp/x)*d*d-2*(A+B/x)*d*x0p)/scale
    D = np.exp(pp[3]); Dp = D*dd[3]
    root = np.sqrt(8*A*D/3); tanh = np.tanh(pp[4])
    C = root*tanh
    Cp = root*(.5*(dd[2]+dd[3])*tanh+(1-tanh*tanh)*dd[4])
    return (mp+Ap*d*d-2*A*d*x0p+Cp*d**3-3*C*d*d*x0p
            +Dp*d**4-4*D*d**3*x0p)/scale


def complex_jacobian(function, parameters):
    h = 1e-25
    return np.column_stack([np.imag(function(parameters+1j*h*np.eye(len(parameters))[i]))/h
                            for i in range(len(parameters))])


def initial_parameters(spec, family, x, alpha, y):
    center, scale = angle_scale(family)
    angles = np.unique(alpha)
    minima = [np.argmin(np.where(alpha == a, y, np.inf)) for a in angles]
    result = []
    for i, degree in enumerate(spec.degrees):
        if i == 0 and spec.form == "rational_monotone":
            result.extend([np.log(min(y)), np.log(.05), 0., np.log(.2)])
            continue
        if i == 0 and spec.form == "rational_softplus":
            low = np.polynomial.polynomial.polyfit((angles[:2]-center)/scale,
                                                   np.log(y[minima[:2]]), 1)
            result.extend([*low, np.log(.04), 0., np.log(.15)])
            continue
        if i < 2:
            targets = np.log(y[minima] if i == 0 else x[minima])
            coeff = np.polynomial.polynomial.polyfit((angles-center)/scale, targets,
                                                    min(degree, len(angles)-1))
            result.extend(np.pad(coeff, (0, degree+1-len(coeff))))
        else:
            constant = (np.log(.08) if i == 2 else np.log(.06)) if i < 4 else -.5
            result.extend([constant] + [0.]*degree)
    return np.array(result)


def fit(spec, family, points, initial=None):
    x, alpha, y, sigma = [np.array([p[k] for p in points]) for k in
                         ("pitch_chord", "alpha_deg", "yp", "sigma_yp")]
    p0 = initial_parameters(spec, family, x, alpha, y) if initial is None else initial
    # Weight each measured point by chart-reading tolerance; do not label this a
    # statistical likelihood (pixels on one printed curve are correlated).
    def residual(parameters):
        return (evaluate(spec, parameters, x, alpha, family)-y)/sigma
    starts = [p0]
    if spec.form != "quadratic":
        index = sum(d+1 for d in spec.degrees[:3])
        for value in (-6., -1.):
            alternative = p0.copy()
            alternative[index] = value
            starts.append(alternative)
    attempts = [least_squares(residual, np.clip(start, -15.99, 15.99),
                              bounds=(-16., 16.), max_nfev=1200,
                              ftol=2e-9, xtol=2e-9, gtol=2e-9)
                for start in starts]
    converged = [r for r in attempts if r.success]
    result = min(converged or attempts, key=lambda r: r.cost)
    result.attempts = [dict(success=bool(r.success), status=int(r.status),
                            cost=float(r.cost), nfev=int(r.nfev),
                            optimality=float(r.optimality),
                            active_bounds=int(np.count_nonzero(r.active_mask)))
                       for r in attempts]
    # User constraint: at low spacing, increasing exit angle must not reverse
    # the loss trend or cross neighboring curves. Constrain the derivative of
    # the actual expression, not a postprocessed/ordered plot. A second dense
    # mesh independently checks the constraints and triggers refinement.
    center, scale = angle_scale(family)
    order_iterations = []
    for resolution in (81, 241):
        xx, aa = np.meshgrid(np.linspace(*ORDER_SPACING_RANGE, 17),
                              np.linspace(center-scale, center+scale, resolution))
        def inequality(parameters):
            return (angle_derivative(spec, parameters, xx.ravel(), aa.ravel(), family)
                    - ORDER_DERIVATIVE_MARGIN)/.001
        if np.min(inequality(result.x)) < 0:
            def objective(parameters):
                r = residual(parameters)
                return .5*np.mean(r*r)
            def gradient(parameters):
                r = residual(parameters)
                return complex_jacobian(residual, parameters).T @ r / len(r)
            constrained = minimize(objective, result.x, method="SLSQP", jac=gradient,
                bounds=[(-16., 16.)]*len(result.x),
                constraints={"type": "ineq", "fun": inequality,
                             "jac": lambda p: complex_jacobian(inequality, p)},
                options=dict(maxiter=500, ftol=1e-10))
            order_iterations.append(dict(success=bool(constrained.success),
                                         message=str(constrained.message),
                                         iterations=int(constrained.nit)))
            result.x = constrained.x
            result.success = bool(constrained.success)
        if np.min(inequality(result.x)) < -1e-5:
            result.success = False
    xx, aa = np.meshgrid(np.linspace(*ORDER_SPACING_RANGE, 101),
                          np.linspace(center-scale, center+scale, 801))
    minimum = float(np.min(angle_derivative(spec, result.x, xx, aa, family)))
    # Exchange points: a dense check can locate a minimum between constraint
    # nodes. Add those actual locations and solve again; never accept a tiny
    # reversal merely because the initial mesh missed it.
    cx, ca = np.meshgrid(np.linspace(*ORDER_SPACING_RANGE, 17),
                         np.linspace(center-scale, center+scale, 241))
    cx, ca = cx.ravel(), ca.ravel()
    for _ in range(6):
        slopes = angle_derivative(spec, result.x, xx, aa, family).ravel()
        if np.min(slopes) >= ORDER_DERIVATIVE_MARGIN*.75:
            break
        worst = np.argsort(slopes)[:12]
        cx, ca = np.r_[cx, xx.ravel()[worst]], np.r_[ca, aa.ravel()[worst]]
        def inequality(parameters):
            return (angle_derivative(spec, parameters, cx, ca, family)-ORDER_DERIVATIVE_MARGIN)/.001
        def objective(parameters):
            r = residual(parameters)
            return .5*np.mean(r*r)
        def gradient(parameters):
            r = residual(parameters)
            return complex_jacobian(residual, parameters).T @ r / len(r)
        constrained = minimize(objective, result.x, method="SLSQP", jac=gradient,
            bounds=[(-16., 16.)]*len(result.x),
            constraints={"type": "ineq", "fun": inequality,
                         "jac": lambda p: complex_jacobian(inequality, p)},
            options=dict(maxiter=500, ftol=1e-10))
        order_iterations.append(dict(success=bool(constrained.success),
                                     message=str(constrained.message),
                                     iterations=int(constrained.nit), adaptive=True))
        result.x = constrained.x
        result.success = bool(constrained.success)
    minimum = float(np.min(angle_derivative(spec, result.x, xx, aa, family)))
    result.order_check = dict(minimum_angle_derivative=minimum,
                              spacing_range=ORDER_SPACING_RANGE,
                              angle_range=(center-scale, center+scale),
                              constrained_optimizations=order_iterations)
    if minimum < -1e-9:
        result.success = False
    return result


def metrics(y, predicted, sigma):
    error = np.asarray(predicted) - y
    return dict(rmse=float(np.sqrt(np.mean(error**2))),
                mae=float(np.mean(abs(error))), max_abs=float(np.max(abs(error))),
                relative_rmse_percent=float(100*np.sqrt(np.mean((error/y)**2))),
                tolerance_scaled_rmse=float(np.sqrt(np.mean((error/sigma)**2))))
