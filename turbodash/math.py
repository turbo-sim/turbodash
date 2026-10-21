"""Shared array-based mathematics, smoothing and interpolation utilities.

These helpers currently execute with NumPy. Compact transitions preserve exact
outer branches and bounds; the legacy smooth-extrema methods retain their
existing defaults and numerical behavior.
"""

import numpy as np
import numpy as jnp


def smooth_maximum(x1, x2, method="boltzmann", alpha=10, *, half_width=None):
    r"""Return an elementwise smooth maximum of two broadcastable inputs.

    Parameters
    ----------
    x1, x2 : array_like
        Input values.
    method : {"boltzmann", "logsumexp", "p-norm", "compact"}, optional
        Boltzmann uses a smooth weighted average (the default); log-sum-exp
        and p-norm use their respective smooth-extremum approximations.
        Compact uses a C2 upper envelope that equals the exact maximum
        when the inputs differ by at least ``half_width``. See
        :func:`positive_part` for its approximation error and regularity.
    alpha : float, optional
        Sharpness of the noncompact methods. Default is 10. Ignored by the
        compact method, which instead uses a width in the input's units.
    half_width : array_like, optional
        Required nonnegative transition half-width for ``method="compact"``.
        May broadcast with the inputs. Zero gives the exact maximum.
        Must be omitted for other methods.

    Returns
    -------
    ndarray or scalar
        Smooth maximum with the broadcast shape of the inputs.

    Raises
    ------
    ValueError
        If the method is unsupported or its width argument is missing or
        supplied to a noncompact method.
    """
    if method == "compact":
        if half_width is None:
            raise ValueError("half_width is required for method='compact'")
        a, b, half_width = jnp.broadcast_arrays(x1, x2, half_width)
        patch = b + positive_part(a - b, half_width)
        return jnp.where(a <= b - half_width, b, jnp.where(a >= b + half_width, a, patch))
    if half_width is not None:
        raise ValueError("half_width is only supported for method='compact'")

    # Ensure x1 and x2 have the same shape by broadcasting
    x1, x2 = jnp.broadcast_arrays(x1, x2)

    # Stack the input arrays along a new axis to treat them as a single array
    x = jnp.stack([x1, x2], axis=0)

    # Compute smooth maximum approximation according to specified method
    if method == "logsumexp":
        return _smooth_max_logsumexp(x, jnp.abs(alpha), axis=0)
    elif method == "boltzmann":
        return _smooth_max_boltzmann(x, jnp.abs(alpha), axis=0)
    elif method == "p-norm":
        return _smooth_max_pnorm(x, jnp.abs(alpha), axis=0)
    else:
        raise ValueError(
            f"Unsupported method '{method}'. Supported methods are:\n"
            "- 'logsumexp'\n"
            "- 'boltzmann'\n"
            "- 'p-norm'\n"
            "- 'compact'"
        )


def smooth_minimum(x1, x2, method="boltzmann", alpha=10, *, half_width=None):
    r"""Return an elementwise smooth minimum of two broadcastable inputs.

    Parameters
    ----------
    x1, x2 : array_like
        Input values.
    method : {"boltzmann", "logsumexp", "p-norm", "compact"}, optional
        Boltzmann uses a smooth weighted average (the default); log-sum-exp
        and p-norm use their respective smooth-extremum approximations.
        Compact uses a C2 lower envelope that equals the exact minimum
        when the inputs differ by at least ``half_width``. See
        :func:`positive_part` for its approximation error and regularity.
    alpha : float, optional
        Sharpness of the noncompact methods. Default is 10. Ignored by the
        compact method, which instead uses a width in the input's units.
    half_width : array_like, optional
        Required nonnegative transition half-width for ``method="compact"``.
        May broadcast with the inputs. Zero gives the exact minimum.
        Must be omitted for other methods.

    Returns
    -------
    ndarray or scalar
        Smooth minimum with the broadcast shape of the inputs.

    Raises
    ------
    ValueError
        If the method is unsupported or its width argument is missing or
        supplied to a noncompact method.
    """
    if method == "compact":
        if half_width is None:
            raise ValueError("half_width is required for method='compact'")
        a, b, half_width = jnp.broadcast_arrays(x1, x2, half_width)
        patch = b - positive_part(b - a, half_width)
        return jnp.where(a <= b - half_width, a, jnp.where(a >= b + half_width, b, patch))
    if half_width is not None:
        raise ValueError("half_width is only supported for method='compact'")

    # Ensure x1 and x2 have the same shape by broadcasting
    x1, x2 = jnp.broadcast_arrays(x1, x2)

    # Stack the input arrays along a new axis to treat them as a single array
    x = jnp.stack([x1, x2], axis=0)

    # Compute smooth maximum approximation according to specified method
    if method == "logsumexp":
        return _smooth_max_logsumexp(x, -jnp.abs(alpha), axis=0)
    elif method == "boltzmann":
        return _smooth_max_boltzmann(x, -jnp.abs(alpha), axis=0)
    elif method == "p-norm":
        return _smooth_max_pnorm(x, -jnp.abs(alpha), axis=0)
    else:
        raise ValueError(
            f"Unsupported method '{method}'. Supported methods are:\n"
            "- 'logsumexp'\n"
            "- 'boltzmann'\n"
            "- 'p-norm'\n"
            "- 'compact'"
        )


def _smooth_max_logsumexp(x, alpha, axis=None, keepdims=False):
    """Smooth approximation to the maximum of an array using the log-sum-exp method"""

    # Determine the shift for numerical stability
    shift_value = jnp.sign(alpha) * jnp.max(
        jnp.sign(alpha) * x, axis=axis, keepdims=True
    )

    # Compute log-sum-exp with the shift and scale by alpha
    log_sum = jnp.log(
        jnp.sum(jnp.exp(alpha * (x - shift_value)), axis=axis, keepdims=True)
    )

    # Normalize the result by alpha and correct for the shift
    smooth_max = (log_sum + alpha * shift_value) / alpha

    # Remove the dimensions of size one if keepdims is False
    if not keepdims:
        smooth_max = jnp.squeeze(smooth_max, axis=axis)

    return smooth_max


def _smooth_max_boltzmann(x, alpha, axis=None, keepdims=False):
    """Smooth approximation to the maximum of an array using the Boltzmann weighted average"""

    # Compute the shift for numerical stability
    shift = jnp.sign(alpha) * jnp.max(jnp.sign(alpha) * x, axis=axis, keepdims=True)

    # Compute the weighted sum (numerator) of the elements of x
    weighted_sum = jnp.sum(x * jnp.exp(alpha * (x - shift)), axis=axis, keepdims=True)

    # Compute the sum of the smooth_max weights (denominator)
    weight_sum = jnp.sum(jnp.exp(alpha * (x - shift)), axis=axis, keepdims=True)

    # Compute the Boltzmann-weighted average avoiding division by zero
    smooth_max = weighted_sum / (weight_sum + jnp.finfo(float).eps)

    # Remove the dimensions of size one if keepdims is False
    if not keepdims:
        smooth_max = jnp.squeeze(smooth_max, axis=axis)

    return smooth_max


def _smooth_max_pnorm(x, alpha, axis=None, keepdims=False):
    """Smooth approximation to the maximum of an array using the p-norm method"""

    # Compute the p-norm approximation
    smooth_max = jnp.sum(jnp.power(x, alpha), axis=axis, keepdims=True) ** (1 / alpha)

    # Remove the dimensions of size one if keepdims is False
    if not keepdims:
        smooth_max = jnp.squeeze(smooth_max, axis=axis)

    return smooth_max


def smooth_abs(x, method="quadratic", epsilon=1e-5):
    r"""
    Compute a smooth approximation of the absolute value function according to the specified method.

    1. The quadratic approximation is given by :cite:p:`ramirez_x_2013`:

    .. math::
       f_{\epsilon}(x) = \sqrt{x^2 + \epsilon}

    2. The hyperbolic tangent approximation is given by :cite:p:`bagul_smooth_2017`:

    .. math::
       f_{\epsilon}(x) = \epsilon \tanh{x / \epsilon}

    3. The log-cosh approximation is given by :cite:p:`saleh_statistical_2022`:

    .. math::
       f_{\epsilon}(x) = \epsilon \log\left(\cosh\left(\frac{x}{\epsilon}\right)\right)

    The quadratic method is the most computationally efficient, but also requires a smaller value of :math:`\epsilon` to yield a good approximation.
    The transcendental methods give a better approximation to the absolute value function, at the expense of a higher computational time.

    Parameters
    ----------
    x : float or np.ndarray
        Input value(s) for which the approximation is computed.
    method : str, optional
        The method of approximation:

        - ``quadratic`` (default)
        - ``hyperbolic``
        - ``logcosh``

    epsilon : float, optional
        A small positive constant affecting the approximation. Default is 1e-5.

    Returns
    -------
    float or np.ndarray
        Smooth approximation value(s) of the absolute function.

    Raises
    ------
    ValueError
        If an unsupported approximation method is provided.
    """

    if method == "quadratic":
        return _smooth_abs_quadratic(x, epsilon)
    elif method == "hyperbolic":
        return _smooth_abs_tanh(x, epsilon)
    elif method == "logarithmic":
        return _smooth_abs_logcosh(x, epsilon)

    else:
        raise ValueError(
            f"Unsupported method '{method}'. Supported methods are:\n"
            "- 'quadratic'\n"
            "- 'hyperbolic'\n"
            "- 'logcosh'"
        )


def _smooth_abs_quadratic(x, epsilon):
    """Quadratic approximation of the absolute value function."""
    return jnp.sqrt(x**2 + epsilon)


def _smooth_abs_tanh(x, epsilon):
    """Hyperbolic tangent approximation of the absolute value function."""
    return x * jnp.tanh(x / epsilon)


def _smooth_abs_logcosh(x, epsilon):
    """Log-cosh approximation of the absolute value function."""
    return epsilon * jnp.log(jnp.cosh(x / epsilon))


def sind(x):
    """Compute the sine of an angle given in degrees."""
    return jnp.sin(x * np.pi / 180.0)


def cosd(x):
    """Compute the cosine of an angle given in degrees."""

    return jnp.cos(x * np.pi / 180.0)


def tand(x):
    """Compute the tangent of an angle given in degrees."""
    return jnp.tan(x * np.pi / 180.0)


def arcsind(x):
    """Compute the arcsine of a value and return the result in degrees."""
    return jnp.arcsin(x) * 180.0 / np.pi


def arccosd(x):
    """Compute the arccosine of a value and return the result in degrees."""
    return jnp.arccos(x) * 180 / np.pi


def arctand(x):
    """Compute the arctangent of a value and return the result in degrees."""
    return jnp.arctan(x) * 180.0 / np.pi


def is_odd(number):
    """Check if a number is odd. Returns True if the provided number is odd, and False otherwise."""
    return number % 2 != 0


def is_even(number):
    """Check if a number is even. Returns True if the provided number is even, and False otherwise."""
    return number % 2 == 0


def all_numeric(array):
    """Check if all items in Numpy array are numeric (floats or ints)"""
    return np.issubdtype(array.dtype, np.number)


def all_non_negative(array):
    "Check if all items in Numpy array are non-negative"
    return np.all(array >= 0)


def sigmoid_hyperbolic(x, x0=0.5, alpha=1):
    r"""
    Compute the sigmoid hyperbolic function.

    This function calculates a sigmoid function based on the hyperbolic tangent.
    The formula is given by:

    .. math::
        \sigma(x) = \frac{1 + \tanh\left(\frac{x - x_0}{\alpha}\right)}{2}

    Parameters
    ----------
    x : array_like
        Input data.
    x0 : float, optional
        Center of the sigmoid function. Default is 0.5.
    alpha : float, optional
        Scale parameter. Default is 0.1.

    Returns
    -------
    array_like
        Output of the sigmoid hyperbolic function.

    """
    x = np.array(x)  # Ensure x is a NumPy array for vectorized operations
    sigma = (1 + np.tanh((x - x0) / alpha)) / 2
    return sigma


def sigmoid_rational(x, n, m):
    r"""
    Compute the sigmoid rational function.

    This function calculates a sigmoid function using an algebraic approach based on a rational function.
    The formula is given by:

    .. math::
        \sigma(x) = \frac{x^n}{x^n + (1-x)^m}

    Parameters
    ----------
    x : array_like
        Input data.
    n : int
        Power of the numerator.
    m : int
        Power of the denominator.

    Returns
    -------
    array_like
        Output of the sigmoid algebraic function.

    """
    x = np.array(x)  # Ensure x is a NumPy array for vectorized operations
    x = np.where(x < 0, 0, x)  # Set x to 0 where x < 0
    x = np.where(x > 1, 1, x)  # Set x to 1 where x > 1
    sigma = x**n / (x**n + (1 - x) ** m)
    return sigma


def sigmoid_smoothstep(x):
    """
    Compute the smooth step function.

    This function calculates a smooth step function with zero first-order
    derivatives at the endpoints. More information available at:
    https://resources.wolframcloud.com/FunctionRepository/resources/SmoothStep/

    Parameters
    ----------
    x : array_like
        Input data.

    Returns
    -------
    array_like
        Output of the sigmoid smoothstep function.

    """
    x = np.array(x)  # Ensure x is a NumPy array for vectorized operations
    x = np.where(x < 0, 0, x)  # Set x to 0 where x < 0
    x = np.where(x > 1, 1, x)  # Set x to 1 where x > 1
    return x**2 * (3 - 2 * x)


def sigmoid_smootherstep(x):
    """
    Compute the smoother step function.

    This function calculates a smoother step function with zero second-order
    derivatives at the endpoints. It is a modification of the smoothstep
    function to provide smoother transitions.

    Parameters
    ----------
    x : array_like
        Input data.

    Returns
    -------
    array_like
        Output of the sigmoid smootherstep function.

    """
    x = jnp.clip(jnp.asarray(x), 0.0, 1.0)
    return x**3 * (10 + x * (-15 + 6 * x))


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
    x = jnp.asarray(x)
    w = jnp.asarray(half_width)
    safe_width = jnp.where(w > 0, w, 1.0)
    t = jnp.clip((x + safe_width) / (2 * safe_width), 0.0, 1.0)
    patch = safe_width * t**3 * (2 - t)
    result = jnp.where(x <= -w, 0.0, jnp.where(x >= w, x, patch))
    return jnp.where(w > 0, result, jnp.maximum(x, 0.0))


def blend_branches(x, left, right, center, half_width):
    """C2 compact blend of two independently smooth branch evaluations."""
    weight = sigmoid_smootherstep((x - center + half_width) / (2 * half_width))
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
