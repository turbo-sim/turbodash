"""Explicit monotone turning with prescribed integrated conformal slope.

This optional scalar law complements the established angle/location law. It
fixes the mean of tan(beta), so a curved section's angular wrap can be imposed
without a root solve. Coordinates ``u`` are normalized conformal distance;
angles passed by name are degrees and returned angles are radians.
"""

from numbers import Real

import numpy as np
from scipy.special import betainc


def _finite_scalar(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite real number.")
    return float(value)


def _cdf_integral(u, shape_a, shape_b, fraction, cdf):
    """Integral from 0 to u of a regularized Beta CDF, without quadrature.

    Integrating by parts gives u*I_u(a,b)-a/(a+b)*I_u(a+1,b).
    Above the CDF midpoint, reflection evaluates the remaining tail instead,
    reducing cancellation near u=1. The full integral is b/(a+b)=fraction.
    """
    result = np.empty_like(u)
    lower = cdf <= 0.5
    x = u[lower]
    result[lower] = x * cdf[lower] - (1 - fraction) * betainc(shape_a + 1, shape_b, x)
    x = 1 - u[~lower]
    tail = x * betainc(shape_b, shape_a, x) - fraction * betainc(shape_b + 1, shape_a, x)
    result[~lower] = fraction - x + tail
    # The analytic integral lies in [0,u]. Discard only arithmetic roundoff.
    tolerance = 64 * np.finfo(float).eps
    if np.any(result < -tolerance) or np.any(result > u + tolerance):
        raise ValueError("Constrained turning integral lost numerical accuracy; relax its mean or concentration.")
    result = np.clip(result, 0, u)
    return np.where(u == 0, 0, np.where(u == 1, fraction, result))


def constrained_turning(u, inlet_angle_deg, outlet_angle_deg, mean_slope, *, concentration=1.0):
    """Return ``(beta, integral_of_tan_beta, metadata)`` with an explicit mean.

    Let the endpoint slopes be a and b, and f=(mean_slope-a)/(b-a).
    For nonconstant endpoints, require 0<f<1. Define
    k=2*concentration/min(f,1-f), A=k*(1-f), B=k*f, then interpolate
    tan(beta)=a+(b-a)*I_u(A,B). Its mean is exactly a+(b-a)*f.

    Both Beta shape parameters are at least two, giving zero slope derivative
    at the endpoints. The slope and angle stay monotone between their endpoint
    values. ``concentration >= 1`` sharpens the transition without changing its
    prescribed integral. Inlet/outlet angles must be strictly inside +/-90 deg.
    Equal endpoint angles admit only their constant slope, within roundoff.

    No numerical integration, iterative fit, or inverse special function is
    used. Metadata records the actual law and its derived parameters so callers
    do not mistake this for the original turning-location parametrization.
    """
    incoming = _finite_scalar(inlet_angle_deg, "inlet_angle_deg")
    outgoing = _finite_scalar(outlet_angle_deg, "outlet_angle_deg")
    target = _finite_scalar(mean_slope, "mean_slope")
    concentration = _finite_scalar(concentration, "concentration")
    if not -90 < incoming < 90 or not -90 < outgoing < 90:
        raise ValueError("Endpoint metal angles must lie strictly between -90 and 90 degrees.")
    if concentration < 1:
        raise ValueError("concentration must be at least one.")
    try:
        u = np.asarray(u, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("u must contain finite coordinates in [0, 1].") from error
    if np.any(~np.isfinite(u)) or np.any((u < 0) | (u > 1)):
        raise ValueError("u must contain finite coordinates in [0, 1].")
    a, b = np.tan(np.deg2rad([incoming, outgoing]))
    details = {"law": "integral_constrained_beta_cdf", "requested_mean_slope": target,
               "concentration": concentration, "constant_angle": incoming == outgoing}
    if incoming == outgoing:
        tolerance = 64 * np.finfo(float).eps * max(1.0, abs(a))
        if abs(target - a) > tolerance:
            raise ValueError("Equal endpoint angles require their constant mean slope.")
        details.update(mean_slope=float(a), mean_fraction=None, shape_a=None, shape_b=None)
        return np.full_like(u, np.deg2rad(incoming)), a * u, details

    fraction = (target - a) / (b - a)
    if not 0 < fraction < 1:
        raise ValueError("Prescribed mean slope must lie strictly between endpoint slopes for monotone turning.")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        k = 2 * concentration / min(fraction, 1 - fraction)
        shape_a, shape_b = k * (1 - fraction), k * fraction
    if not np.all(np.isfinite([k, shape_a, shape_b])):
        raise ValueError("Constrained turning shape parameters must remain finite; relax its mean or concentration.")
    cdf = betainc(shape_a, shape_b, u)
    integral_cdf = _cdf_integral(u, shape_a, shape_b, fraction, cdf)
    slope = a + (b - a) * cdf
    beta = np.arctan(slope)
    beta = np.where(u == 0, np.deg2rad(incoming), np.where(u == 1, np.deg2rad(outgoing), beta))
    integral = a * u + (b - a) * integral_cdf
    integral = np.where(u == 0, 0, np.where(u == 1, target, integral))
    if not np.all(np.isfinite(beta)) or not np.all(np.isfinite(integral)):
        raise ValueError("Constrained turning evaluation must remain finite.")
    details.update(mean_slope=target, mean_fraction=float(fraction),
                   shape_a=float(shape_a), shape_b=float(shape_b))
    return beta, integral, details
