"""Small candidate-only numerical transitions; the shared math module is unchanged."""

from ._backend import jnp


def positive_part(x, half_width):
    r"""C2 upper approximation to :math:`\max(x,0)` with compact support.

    For :math:`w>0`, put :math:`t=(x+w)/(2w)` and use
    :math:`w(2t^3-t^4)` inside :math:`[-w,w]`, zero below and the identity
    above. Value, first and second derivatives match the outer branches.
    The derivative lies in [0,1]; the result enforces the lower bound and
    exceeds the exact maximum by at most :math:`3w/16`. Width and input have
    the same units and may broadcast. A zero width gives the exact maximum;
    callers using a variable width must establish regularity at its zeros.
    This is a numerical transition, not an empirical correlation.
    """
    w = jnp.asarray(half_width)
    safe_width = jnp.where(w > 0, w, 1.0)
    t = jnp.clip((x + safe_width) / (2 * safe_width), 0.0, 1.0)
    patch = safe_width * t**3 * (2 - t)
    result = jnp.where(x <= -w, 0.0, jnp.where(x >= w, x, patch))
    return jnp.where(w > 0, result, jnp.maximum(x, 0.0))


def smooth_maximum(a, b, half_width):
    """Enforce max(a,b) using the compact C2 positive-part upper envelope."""
    patch = b + positive_part(a - b, half_width)
    return jnp.where(a <= b - half_width, b, jnp.where(a >= b + half_width, a, patch))


def smooth_minimum(a, b, half_width):
    """Enforce min(a,b) using the compact C2 positive-part lower envelope."""
    patch = b - positive_part(b - a, half_width)
    return jnp.where(a <= b - half_width, a, jnp.where(a >= b + half_width, b, patch))


def smoothstep5(t):
    r"""Return :math:`6t^5-15t^4+10t^3` on [0,1], constant outside (C2)."""
    t = jnp.clip(t, 0.0, 1.0)
    return t**3 * (10 + t * (-15 + 6 * t))


def blend_branches(x, left, right, center, half_width):
    """C2 compact blend of two independently smooth branch evaluations."""
    weight = smoothstep5((x - center + half_width) / (2 * half_width))
    return (1 - weight) * left + weight * right


def quintic_hermite(x, lower, upper, y0, y1, dy0, dy1, d2y0, d2y1):
    r"""Interpolate endpoint value, first derivative and curvature on [lower,upper].

    With :math:`h=u-l`, :math:`t=(x-l)/h`, this is the unique polynomial
    of degree at most five satisfying all six endpoint conditions. Derivatives
    supplied here are with respect to x, not t. It makes a C2 join when callers
    select the corresponding outer branches. No monotonicity or bound is
    automatic: those must be verified for each use. Inputs may broadcast.
    """
    h = upper - lower
    t = jnp.clip((x - lower) / h, 0.0, 1.0)
    h00 = 1 - 10 * t**3 + 15 * t**4 - 6 * t**5
    h10 = t - 6 * t**3 + 8 * t**4 - 3 * t**5
    h20 = (t**2 - 3 * t**3 + 3 * t**4 - t**5) / 2
    h01 = 10 * t**3 - 15 * t**4 + 6 * t**5
    h11 = -4 * t**3 + 7 * t**4 - 3 * t**5
    h21 = (t**3 - 2 * t**4 + t**5) / 2
    return (
        h00 * y0
        + h10 * h * dy0
        + h20 * h**2 * d2y0
        + h01 * y1
        + h11 * h * dy1
        + h21 * h**2 * d2y1
    )


def monotone_cubic_table(x, knots, values, *, flat_left=True, flat_right=True):
    r"""C1 shape-preserving cubic table with exact knots and constant extension.

    Interior slopes use the weighted harmonic mean of adjacent secants when
    they have the same sign, and zero otherwise. This is the PCHIP slope
    construction; each cubic is evaluated in Hermite form. Constant extension
    is C1 at endpoints whose ``flat_*`` flag is true (slope zero). A false flag
    retains the one-sided shape-preserving endpoint slope, suitable when that
    endpoint is a domain boundary rather than a continuation. At least three
    strictly increasing knots are required. Shape and bound preservation hold
    for monotone ordinate tables; second derivatives generally jump at knots.
    The tabulated calibration values are not changed by this interpolation.
    """
    xp, yp = jnp.asarray(knots), jnp.asarray(values)
    h = jnp.diff(xp)
    d = jnp.diff(yp) / h
    same_sign = (d[:-1] * d[1:]) > 0
    dl = jnp.where(same_sign, d[:-1], 1.0)
    dr = jnp.where(same_sign, d[1:], 1.0)
    w1, w2 = 2 * h[1:] + h[:-1], h[1:] + 2 * h[:-1]
    interior = jnp.where(same_sign, (w1 + w2) / (w1 / dl + w2 / dr), 0.0)

    def endpoint(ha, hb, da, db):
        slope = ((2 * ha + hb) * da - ha * db) / (ha + hb)
        slope = jnp.where(jnp.sign(slope) != jnp.sign(da), 0.0, slope)
        return jnp.where(
            (jnp.sign(da) != jnp.sign(db)) & (jnp.abs(slope) > 3 * jnp.abs(da)),
            3 * da,
            slope,
        )

    left = 0.0 if flat_left else endpoint(h[0], h[1], d[0], d[1])
    right = 0.0 if flat_right else endpoint(h[-1], h[-2], d[-1], d[-2])
    slopes = jnp.concatenate((jnp.atleast_1d(left), interior, jnp.atleast_1d(right)))
    index = jnp.clip(jnp.searchsorted(xp, x, side="right") - 1, 0, len(knots) - 2)
    interval = xp[index + 1] - xp[index]
    t = jnp.clip((x - xp[index]) / interval, 0.0, 1.0)
    return (
        (2 * t**3 - 3 * t**2 + 1) * yp[index]
        + (t**3 - 2 * t**2 + t) * interval * slopes[index]
        + (-2 * t**3 + 3 * t**2) * yp[index + 1]
        + (t**3 - t**2) * interval * slopes[index + 1]
    )
