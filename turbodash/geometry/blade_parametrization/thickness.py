"""Global rational Denton body law, independent of mapping and the GUI."""
import numpy as np


def fullness_coefficients(parameters):
    """Return the center and slope of a(z); reject singular combinations."""
    f = np.array([parameters.get("leading_edge_fullness", 0.),
                  parameters.get("trailing_edge_fullness", 0.)], dtype=float)
    if not np.isfinite(f).all() or np.any(np.abs(f) > 1):
        raise ValueError("Leading and trailing fullness must each lie in [-1, 1].")
    if np.any(np.abs(f) == 1):
        raise ValueError("Shoulder fullness endpoints -1 and +1 produce degenerate thickness profiles; use values strictly between them.")
    n = parameters["shape_exponent"]
    a = 2*f/(1+f)
    center = float(a.mean())
    slope = float((a[1]-a[0])/(2*2**(-1/n)))
    if center + abs(slope) >= 1:
        raise ValueError("The LE/TE fullness combination makes the rational thickness law singular. "
                         "Reduce the difference between the two fullness values.")
    return center, slope


def denton_body(xi, parameters):
    """Unrounded full thickness / conformal chord; zero fullness is Denton."""
    x = np.asarray(xi, dtype=float)
    location = parameters["maximum_thickness_location"]
    leading = parameters["leading_edge_thickness_ratio"]
    trailing = parameters["trailing_edge_thickness_ratio"]
    p = np.log(.5)/np.log(location)
    z = 2*x**p-1
    u = np.abs(z)**parameters["shape_exponent"]
    center, slope = fullness_coefficients(parameters)
    amplitude = parameters["maximum_thickness_ratio"]-(leading+location*(trailing-leading))
    body = leading+x*(trailing-leading)+amplitude*(1-u)/(1-(center+slope*z)*u)
    if np.any(body < -1e-14):
        raise ValueError("The Denton parameters produce negative body thickness.")
    return np.maximum(body, 0.)
