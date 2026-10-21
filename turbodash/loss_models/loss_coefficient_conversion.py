"""Thermodynamic conversions between blade-row loss coefficients.

Ideal-gas identities use cancellation-resistant logarithms. Optional compact
numerical guards are documented with their parameters; they do not change
the definition of either loss coefficient. Array operations use NumPy.
"""

from ._backend import jnp
from .. import math


def convert_stagnation_pressure_to_kinetic_energy_loss(Ma, gamma, Y):
    r"""
    Convert stagnation pressure loss coefficient (:math:`Y`) to kinetic energy loss coefficient (:math:`\Delta \phi^2`).

    This function calculates the kinetic energy loss coefficient (:math:`\Delta \phi^2`) using the
    given Mach number (:math:`\text{Ma}`), specific heat capacity ratio (:math:`\gamma`), and the stagnation
    pressure loss coefficient (:math:`Y`).

    .. math::
        \Delta \phi^2=\frac{\left(\frac{p_2}{p_{02}}\right)^{\frac{\gamma-1}{\gamma}} \left(\left[1+Y \left(1- \frac{p_2}{p_{02}}\right)\right]^{\frac{\gamma-1}{\gamma}}-1\right)}{\left[1+Y \left(1-\frac{p_2}{p_{02}}\right)\right]^{\frac{\gamma-1}{\gamma}}-\left(\frac{p_2}{p_{02}}\right)^{\frac{\gamma-1}{\gamma}}}

    where the static-to-total pressure ratio is given by:

    .. math::
        \left(\frac{p_2}{p_{02}}\right)=\left(\frac{T_2}{T_{02}}\right)^{\frac{\gamma}{\gamma-1}}=\left[1+\left(\frac{\gamma-1}{2}\right) \text{Ma}^2\right]^{-\frac{\gamma}{\gamma-1}}

    Parameters
    ----------
    Ma : float
        Mach number at the exit of the nozzle.
    gamma : float
        Specific heat capacity ratio of the gas.
    Y : float
        Dimensionless stagnation-pressure loss coefficient.

    Returns
    -------
    float
        The computed value of the kinetic energy loss coefficient (Δφ²).

    Notes
    -----
    All coefficients are dimensionless. The thermodynamic identities
    assume a calorically perfect gas with constant :math:`\gamma`;
    they do not perform a real-fluid equation-of-state conversion.
    The admissible physical domain is :math:`M\geq0`, :math:`\gamma>1`,
    :math:`Y\geq0`. Signed increments are also evaluated for :math:`Y>-1`;
    this algebraic continuation is not a nonnegative total-loss model.
    Other inputs return NaN. At zero Mach the continuous limit is
    :math:`\Delta\phi^2=Y/(1+Y)`.

    To avoid small-Mach cancellation, define :math:`q=(\gamma-1)M^2/2`,
    :math:`e=(\gamma-1)/\gamma`,
    :math:`B=-\operatorname{expm1}[-\log(1+q)/e]`, and
    :math:`v=\operatorname{expm1}[e\log(1+YB)]`. The identical gas relation
    is then evaluated as

    .. math::

        \Delta\phi^2=\frac{v}{q+(1+q)v}.

    ``log1p`` evaluates both logarithms. No empirical smoothing band is
    introduced; the zero-Mach extension is analytic in :math:`M^2`.
    """
    Ma, gamma, Y = jnp.broadcast_arrays(Ma, gamma, Y)
    valid = (
        jnp.isfinite(Ma)
        & (Ma >= 0)
        & jnp.isfinite(gamma)
        & (gamma > 1)
        & jnp.isfinite(Y)
        & (Y > -1)
    )
    gas = jnp.where(valid, gamma, 1.4)
    mach = jnp.where(valid, Ma, 0.0)
    loss = jnp.where(valid, Y, 0.0)
    exp = (gas - 1) / gas
    q = (gas - 1) / 2 * mach**2
    pressure_drop = -jnp.expm1(-jnp.log1p(q) / exp)
    v = jnp.expm1(exp * jnp.log1p(loss * pressure_drop))
    denominator = q + (1 + q) * v
    delta_phi2 = v / jnp.where(q == 0, 1.0, denominator)
    delta_phi2 = jnp.where(q == 0, loss / (1 + loss), delta_phi2)
    return jnp.where(valid, delta_phi2, jnp.nan)


def convert_kinetic_energy_to_stagnation_pressure_loss(
    Ma, gamma, delta_phi2, limit_output=True
):
    r"""
    Convert kinetic energy loss coefficient (:math:`\Delta \phi^2`) to stagnation pressure loss coefficient (:math:`Y`).

    This function calculates the stagnation pressure loss coefficient (:math:`Y`) using the given
    Mach number (:math:`\text{Ma}`), specific heat capacity ratio (:math:`\gamma`), and kinetic energy loss
    coefficient (:math:`\Delta \phi^2`).

    .. math::
        Y=\frac{\left[\left(\frac{1}{1-\Delta \phi^2}\right) \left(1-\Delta \phi^2 \left(\frac{p_2}{p_{02}}\right)^{-\frac{\gamma-1}{\gamma}}\right)\right]^{-\left(\frac{\gamma}{\gamma-1}\right)}-1}{1- \left(\frac{p_2}{p_{02}}\right)}

    where the static-to-total pressure ratio is given by:

    .. math::
        \left(\frac{p_2}{p_{02}}\right)=\left(\frac{T_2}{T_{02}}\right)^{\frac{\gamma}{\gamma-1}}=\left[1+\left(\frac{\gamma-1}{2}\right) \text{Ma}^2\right]^{-\frac{\gamma}{\gamma-1}}

    Parameters
    ----------
    Ma : float
        Mach number at the exit of the nozzle.
    gamma : float
        Specific heat capacity ratio of the gas.
    delta_phi2 : float
        The kinetic energy loss coefficient (:math:`\Delta \phi^2`).

    limit_output : bool, optional
        If True (default), apply the compact Mach guard and upper pressure-loss
        cap described below. If False, evaluate the unmodified ideal-gas
        identity in its domain, without either guard.

    Returns
    -------
    float
        The computed value of the stagnation pressure loss (:math:`Y`).

    Notes
    -----
    Write :math:`d=\Delta\phi^2`, :math:`q=(\gamma-1)M^2/2`, and
    :math:`e=(\gamma-1)/\gamma`. The unrestricted relation is evaluated as

    .. math::

        Y=\frac{\operatorname{expm1}\left[-\log\left(1-
            \frac{dq}{1-d}\right)/e\right]}
            {-\operatorname{expm1}[-\log(1+q)/e]}.

    ``log1p`` and ``expm1`` avoid subtractive cancellation. At :math:`M=0`
    the analytic limit is :math:`Y=d/(1-d)`; this extension is smooth in
    :math:`M^2`. The unrestricted domain is :math:`M\geq0`,
    :math:`\gamma>1`, and :math:`d<1/(1+q)`. Nonnegative :math:`d` denotes
    a physical total loss; negative :math:`d` is retained as a signed
    increment continuation for the incidence correlations. Values outside
    the algebraic domain return NaN, without clipping to an admissible input.

    With ``limit_output=True``, :math:`0<d<1` uses

    .. math::

        M_\mathrm{sing}=\sqrt{\frac{2(1-d)}{(\gamma-1)d}},\qquad
        L=0.95M_\mathrm{sing},\qquad
        M_\mathrm{eff}=\operatorname{compactmin}_{0.02L}(M,L),\qquad
        Y=\operatorname{compactmin}_{0.02}(Y_\mathrm{formula}(M_\mathrm{eff}),1).

    The Mach guard retains the 0.95 safety factor. The arbitrary
    :math:`d+10^{-4}` denominator offset is removed using masked safe
    arithmetic; the guard is inactive for :math:`d\leq0`. The numerical
    compact minimum from :mod:`turbodash.math` is C2,
    enforces its bound, and is exactly its smaller argument outside the
    transition. The Mach transition is :math:`[0.98L,1.02L]` and the raw-Y
    transition is [0.98,1.02]. These widths are numerical choices, not
    published gas relations. Both guards are optional regularizations;
    they intentionally break round-trip inversion where active. The
    pressure-loss cap remains 1 and coefficient normalization is unchanged.

    All coefficients are dimensionless. The thermodynamic identities
    assume a calorically perfect gas with constant :math:`\gamma`;
    they do not perform a real-fluid equation-of-state conversion.
    Invalid gamma, negative Mach, nonfinite inputs, and :math:`d\geq1`
    remain explicit NaN results even with regularization enabled.
    """
    if limit_output:
        Ma = _guard_conversion_mach(Ma, gamma, delta_phi2)
    Y = _unrestricted_kinetic_to_pressure(Ma, gamma, delta_phi2)
    if limit_output:
        Y = math.smooth_minimum(Y, 1.0, method="compact", half_width=0.02)
    return Y


def _guard_conversion_mach(Ma, gamma, delta_phi2):
    r"""Apply the C2 Mach guard documented by the public inverse conversion."""
    Ma, gamma, delta_phi2 = jnp.broadcast_arrays(Ma, gamma, delta_phi2)
    active = (
        jnp.isfinite(Ma)
        & (Ma >= 0)
        & jnp.isfinite(gamma)
        & (gamma > 1)
        & jnp.isfinite(delta_phi2)
        & (delta_phi2 > 0)
        & (delta_phi2 < 1)
    )
    safe_gamma = jnp.where(active, gamma, 1.4)
    safe_loss = jnp.where(active, delta_phi2, 0.5)
    limit = 0.95 * jnp.sqrt(2 * (1 - safe_loss) / ((safe_gamma - 1) * safe_loss))
    guarded = math.smooth_minimum(Ma, limit, method="compact", half_width=0.02 * limit)
    return jnp.where(active, guarded, Ma)


def _unrestricted_kinetic_to_pressure(Ma, gamma, delta_phi2):
    r"""Stable ideal-gas inverse, with analytic zero-Mach limit and NaN gaps."""
    Ma, gamma, delta_phi2 = jnp.broadcast_arrays(Ma, gamma, delta_phi2)
    valid = (
        jnp.isfinite(Ma)
        & (Ma >= 0)
        & jnp.isfinite(gamma)
        & (gamma > 1)
        & jnp.isfinite(delta_phi2)
        & (delta_phi2 < 1)
    )
    gas = jnp.where(valid, gamma, 1.4)
    mach = jnp.where(valid, Ma, 0.0)
    loss = jnp.where(valid, delta_phi2, 0.0)
    exp = (gas - 1) / gas
    q = (gas - 1) / 2 * mach**2
    argument = loss * q / (1 - loss)
    valid = valid & (argument < 1)
    argument = jnp.where(valid, argument, 0.0)
    numerator = jnp.expm1(-jnp.log1p(-argument) / exp)
    denominator = -jnp.expm1(-jnp.log1p(q) / exp)
    Y = numerator / jnp.where(q == 0, 1.0, denominator)
    Y = jnp.where(q == 0, loss / (1 - loss), Y)
    return jnp.where(valid, Y, jnp.nan)


def get_kinetic_energy_conversion_diagnostics(Ma, gamma, delta_phi2):
    r"""
    Expose the ideal-gas inverse and each optional numerical regularization.

    Parameters use the units and signed-increment domain documented in
    :func:`convert_kinetic_energy_to_stagnation_pressure_loss`. ``Y_raw``
    is the unrestricted identity and can be NaN when ``Y_limited`` is
    finite because of Mach preprocessing. ``domain_margin`` is
    :math:`1-d[1+(\gamma-1)M^2/2]`, which must be positive for that raw
    identity. ``Y_guarded`` uses the effective Mach before the Y cap.
    These diagnostics expose a regularization's effect without modifying
    the supplied input or silently treating a cap as a physical asymptote.

    Returns
    -------
    dict
        ``Y_raw``, ``Y_guarded``, ``Y_limited``, ``Mach_effective`` and
        ``domain_margin``; each is an array or NumPy scalar.
    """
    mach = _guard_conversion_mach(Ma, gamma, delta_phi2)
    raw = _unrestricted_kinetic_to_pressure(Ma, gamma, delta_phi2)
    guarded = _unrestricted_kinetic_to_pressure(mach, gamma, delta_phi2)
    return dict(
        Y_raw=raw,
        Y_guarded=guarded,
        Y_limited=math.smooth_minimum(guarded, 1.0, method="compact", half_width=0.02),
        Mach_effective=mach,
        domain_margin=1
        - jnp.asarray(delta_phi2)
        * (1 + (jnp.asarray(gamma) - 1) / 2 * jnp.asarray(Ma) ** 2),
    )


def convert_enthalpy_to_stagnation_pressure_loss(Ma, gamma, zeta):
    r"""
    Convert enthalpy loss coefficient (:math:`\zeta`) to stagnation pressure loss coefficient (:math:`Y`).

    This function calculates the stagnation pressure loss coefficient (:math:`Y`) using the
    given Mach number (:math:`\text{Ma}`), specific heat capacity ratio (:math:`\gamma`), and enthalpy loss
    coefficient (:math:`\zeta`).

    The enthalpy loss coefficient (:math:`\zeta`) is first converted to the kinetic energy loss
    coefficient (:math:`\Delta \phi^2`) using the formula:

    .. math::
        \Delta \phi^2 = \frac{\zeta}{1 + \zeta}

    The resulting :math:`\Delta \phi^2` is then used in the function
    `convert_kinetic_energy_to_stagnation_pressure_loss()` to compute :math:`Y`.

    Parameters
    ----------
    Ma : float
        Mach number at the exit of the nozzle.
    gamma : float
        Specific heat capacity ratio of the gas.
    zeta : float
        Enthalpy loss coefficient (:math:`\zeta`).

    Returns
    -------
    float
        The computed value of the stagnation pressure loss coefficient (:math:`Y`).

    Notes
    -----
    All coefficients are dimensionless. The thermodynamic identities
    assume a calorically perfect gas with constant :math:`\gamma`;
    they do not perform a real-fluid equation-of-state conversion.
    The physical domain is :math:`M\geq0`, :math:`\gamma>1`,
    :math:`\zeta\geq0`, with signed-increment continuation for
    :math:`\zeta>-1`. Invalid :math:`\zeta` returns NaN, including the
    singular value -1. The delegated conversion supplies the analytic
    zero-Mach limit, compact guards and explicit NaN invalid-domain results;
    see :func:`convert_kinetic_energy_to_stagnation_pressure_loss`.
    """
    zeta = jnp.asarray(zeta)
    valid = jnp.isfinite(zeta) & (zeta > -1)
    safe_zeta = jnp.where(valid, zeta, 0.0)
    delta_phi2 = jnp.where(valid, safe_zeta / (1 + safe_zeta), jnp.nan)
    Y = convert_kinetic_energy_to_stagnation_pressure_loss(Ma, gamma, delta_phi2)
    return Y


def convert_stagnation_pressure_to_enthalpy_loss(Ma, gamma, Y):
    r"""
    Convert stagnation pressure loss coefficient (:math:`Y`) to enthalpy loss coefficient (:math:`\zeta`).

    This function calculates the enthalpy loss coefficient (:math:`\zeta`) using the given Mach number
    (:math:`\text{Ma}`), specific heat capacity ratio (:math:`\gamma`), and stagnation pressure loss
    coefficient (:math:`Y`).

    The stagnation pressure loss coefficient (:math:`Y`) is first converted to the kinetic
    energy loss coefficient (:math:`\Delta \phi^2`) using the function
    `convert_stagnation_pressure_to_kinetic_energy_loss()`. The resulting :math:`\Delta \phi^2` is then
    used to compute :math:`\zeta` using the formula:

    .. math::
        \zeta = \frac{\Delta \phi^2}{1 - \Delta \phi^2}

    Parameters
    ----------
    Ma : float
        Mach number at the exit of the nozzle.
    gamma : float
        Specific heat capacity ratio of the gas.
    Y : float
        Stagnation pressure loss coefficient (:math:`Y`).

    Returns
    -------
    float
        The computed value of the enthalpy loss coefficient (:math:`\zeta`).

    Notes
    -----
    All coefficients are dimensionless. The thermodynamic identities
    assume a calorically perfect gas with constant :math:`\gamma`;
    they do not perform a real-fluid equation-of-state conversion.
    The physical domain is :math:`M\geq0`, :math:`\gamma>1`,
    :math:`Y\geq0`. The delegated conversion supplies the analytic
    zero-Mach limit and explicit NaN invalid-domain results; see
    :func:`convert_stagnation_pressure_to_kinetic_energy_loss`.
    """
    delta_phi2 = convert_stagnation_pressure_to_kinetic_energy_loss(Ma, gamma, Y)
    zeta = delta_phi2 / (1 - delta_phi2)
    return zeta
