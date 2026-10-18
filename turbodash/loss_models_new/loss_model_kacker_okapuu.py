"""Independent cleaned candidate: loss_model_kacker_okapuu.

Lengths use a consistent unit, angles are degrees from the meridional direction. chord
is true chord; height is full blade height. Array operations retain the baseline NumPy
backend and limited JAX-oriented structure; no JIT guarantee is made.

Candidate-only numerical transitions and the analytical AM reference calibration
are documented below and in improving_turbodash/loss_model_audit/.

Constants are documented beside the functions they control. They remain at
module scope so audit scripts can inspect the exact settings used by the model.
"""

from ._backend import jnp
from . import _numerics as numerics
from . import _trailing_edge as trailing_edge
from .. import math

# Shared row identifiers: translate input names to the integer convention used
# throughout the loss helpers and the Benner/Moustapha model wrappers.
_CASCADE_TO_ID = {"stator": 0, "rotor": 1}

_FLOW_KEYS = {
    "Re_out",
    "Ma_rel_out",
    "Ma_rel_in",
    "p0_rel_in",
    "p0_rel_is",
    "p_in",
    "p0_rel_out",
    "p_out",
    "beta_out",
    "beta_in",
    "gamma_out",
}
_GEOM_KEYS = {
    "hub_tip_ratio_in",
    "pitch",
    "chord",
    "leading_edge_angle",
    "maximum_thickness",
    "meridional_chord",
    "height",
    "opening",
    "trailing_edge_thickness",
    "tip_clearance",
    "A_throat",
    "A_out",
}


def compute_losses(input_parameters):
    r"""
    Evaluate loss coefficient according to :cite:`kacker_mean_1982`.

    This model split the total loss coefficient into profile, secondary, trailing edge and tip clearance.
    The loss coefficient are combined through the following relation:

    .. math::

        \mathrm{Y_{tot}} = \mathrm{Y_{p}} + \mathrm{Y_{te}} + \mathrm{Y_{s}} + \mathrm{Y_{cl}}

    The function calls a function for each loss component:

        - :math:`\mathrm{Y_{p}}` : `get_profile_loss`
        - :math:`\mathrm{Y_{te}}` : `get_trailing_edge_loss`
        - :math:`\mathrm{Y_{s}}` : `get_secondary_loss`
        - :math:`\mathrm{Y_{cl}}` : `get_tip_clearance_loss`

    Parameters
    ----------
    input_parameters : dict
        A dictionary containing containing the keys, ``flow``, ``geometry`` and ``loss_model``.

    Returns
    -------
    dict
        A dictionary with the loss components.
    """
    flow_parameters = input_parameters["flow"]
    geometry = input_parameters["geometry"]
    cascade_type = geometry.get("cascade_type", "stator")
    if cascade_type not in _CASCADE_TO_ID:
        raise ValueError(
            f"Unsupported cascade_type '{cascade_type}'. Expected one of {list(_CASCADE_TO_ID)}."
        )
    cascade_type_id = _CASCADE_TO_ID[cascade_type]
    flow_jax = {
        k: jnp.asarray(v) for k, v in flow_parameters.items() if k in _FLOW_KEYS
    }
    geom_jax = {k: jnp.asarray(v) for k, v in geometry.items() if k in _GEOM_KEYS}
    losses = _compute_losses_jit(flow_jax, geom_jax, cascade_type_id)
    return losses


def _compute_losses_jit(flow_parameters, geometry, cascade_type_id: int):
    r"""
    Assemble the four Kacker–Okapuu components and zero incidence. Historical function
    name retained; this function is not JIT compiled.
    """
    Y_p = get_profile_loss(flow_parameters, geometry, cascade_type_id)
    Y_s = get_secondary_loss(flow_parameters, geometry, cascade_type_id)
    Y_cl = get_tip_clearance_loss(flow_parameters, geometry, cascade_type_id)
    Y_te = get_trailing_edge_loss(flow_parameters, geometry)
    return {
        "loss_profile": Y_p,
        "loss_incidence": jnp.asarray(0.0),
        "loss_trailing": Y_te,
        "loss_secondary": Y_s,
        "loss_clearance": Y_cl,
        "loss_total": Y_p + Y_te + Y_s + Y_cl,
    }


def get_profile_loss(flow_parameters, geometry, cascade_type_id: int):
    r"""
    Calculate the Kacker--Okapuu profile pressure-loss coefficient.

    The model of :cite:`kacker_mean_1982` combines the reaction and impulse
    analytical blade-reference fits calibrated to the data of
    :cite:`ainley_method_1951`. With pitch :math:`s`, true chord :math:`c`,
    inlet metal angle :math:`\theta_\mathrm{in}` and exit flow angle
    :math:`\beta_\mathrm{out}`, define

    .. math::

        z = -\frac{\theta_\mathrm{in}}{\beta_\mathrm{out}}, \qquad
        \widetilde{\beta}_\mathrm{out}
          = \operatorname{smoothabs}(\beta_\mathrm{out})
          = \sqrt{\beta_\mathrm{out}^2+10^{-5}},

        Y_\mathrm{reaction}
          = f_\mathrm{nozzle}(s/c,\widetilde{\beta}_\mathrm{out}), \qquad
        Y_\mathrm{impulse}
          = f_\mathrm{impulse}(s/c,\widetilde{\beta}_\mathrm{out}),

        Y_{p,\mathrm{ref}}(z)=\begin{cases}
          Y_\mathrm{reaction}\left(0.8+\dfrac{0.2}{1+z^2}\right) & z<0,\\
          (1-z^2)Y_\mathrm{reaction}+z^2Y_\mathrm{impulse} & z\ge0,
        \end{cases}

        Y_{p,1} = Y_{p,\mathrm{ref}}(z)F_t(z,t_\mathrm{max}/c).

    Upper extrapolation requires the impulse reference to be at least the
    reaction reference. The thickness factor :math:`F_t` is 1 for z<=0,
    follows the original power law for z>=0.05, and blends the two with a
    quintic smoothstep on 0<z<0.05, centered at z=0.025. See
    :func:`get_profile_thickness_factor` for the expression and derivatives.

    The Reynolds-number correction uses the exit Reynolds number based on
    true chord and the velocity in the blade-row frame:

    .. math::

        f_\mathrm{Re} =
        \begin{cases}
          (\mathrm{Re}_\mathrm{out}/(2\times10^5))^{-0.4}
            & \mathrm{Re}_\mathrm{out}<2\times10^5,\\
          1 & 2\times10^5\leq\mathrm{Re}_\mathrm{out}\leq10^6,\\
          (\mathrm{Re}_\mathrm{out}/10^6)^{-0.2}
            & \mathrm{Re}_\mathrm{out}>10^6.
        \end{cases}

    The inlet shock term is evaluated by :func:`get_shock_loss`. Its Figure 6
    hub correction flattens smoothly below inlet hub/tip ratio 0.3. A quintic
    smoothstep activates shock loss over hub Mach 0.4--0.5; the value and first
    two derivatives match at both boundaries. It uses the actual inlet and
    exit total-minus-static pressure heads and applies no output loss floor:

    .. math::

        Y_{\mathrm{shock,raw}}=0.75\,\Phi(M_{1,\mathrm{hub}})
        \frac{r_h}{r_t}\frac{p_{0,\mathrm{rel,in}}-p_\mathrm{in}}
        {p_{0,\mathrm{rel,out}}-p_\mathrm{out}}.

    The shock contribution is Y_shock = continue_shock_loss(Y_shock_raw),
    unchanged through 0.25 and smoothly bounded by an asymptote of 0.5.
    The returned profile coefficient is

    .. math::

        Y_p = 0.914 f_\mathrm{Re}
          \left(\frac{2}{3}Y_{p,1}K_p+Y_\mathrm{shock}\right),

    where :math:`K_p` comes from :func:`get_compressible_correction_factors`.

    Parameters
    ----------
    flow_parameters : dict
        Row-frame inlet/outlet Mach numbers, ``Re_out``, ``beta_out`` in degrees,
        and pressures ``p0_rel_in``, ``p_in``, ``p0_rel_out`` and
        ``p_out`` in a consistent unit. The shock term uses the actual inlet head.
    geometry : dict
        ``pitch``, true ``chord``, ``maximum_thickness``, inlet metal angle
        ``leading_edge_angle`` in degrees, and ``hub_tip_ratio_in``.
    cascade_type_id : int
        Blade-row identifier: 0 for a stator, 1 for a rotor. Stator flow inputs
        are absolute-frame quantities; rotor inputs are relative-frame quantities.

    Returns
    -------
    float or array_like
        Dimensionless profile pressure-loss coefficient, before any Benner
        penetration weighting.

    Notes
    -----
    The code calculates ``1 + 60*(Ma_rel_out-1)**2`` above Mach one but then
    overwrites that multiplier with 1.0. The displayed result therefore includes
    the inlet shock term but no direct outlet supersonic multiplier.

    :func:`get_profile_loss_blend` performs exact quadratic interpolation and
    bounded gradual extrapolation. At zero turning (z=-1) the reference loss is
    90% of the reaction reference; toward increasingly negative z it approaches 80% without a flat
    cutoff. These levels are engineering assumptions, not AM calibration data.
    The reference coefficient is C1, not C2, at z=0.
    No downstream profile-loss floor is applied.

    Numerical modifications retain the C2 Reynolds joins in
    :func:`get_reynolds_correction_factor`. The reference-angle input is
    sqrt(beta_out**2 + 1e-5), the existing smooth magnitude, without a minimum
    outlet angle. Below the chart range the analytical references extrapolate
    directly. The signed beta_out still defines z and must be nonzero.
    Reference continuation and table interpolation are described in their
    helpers; these choices preserve a spacing optimum.

    The shock helper uses a C2 onset over hub Mach 0.4--0.5 and a C2 hub fit.
    Exit-normalized shock coefficients above 0.25 use a C2 continuation
    toward 0.5; see continue_shock_loss. This is an engineering bound before
    the outer Reynolds and 0.914 multipliers, not a bound on the full profile.
    Admissible inputs have positive Reynolds number, chord,
    thickness and shock-pressure denominator, nonzero exit angle, and positive
    physical hub/tip ratio. The thickness factor is C2 at both ends of its
    transition. The full profile is C1 at z=0; its reference coefficient
    generally retains a curvature jump there. The transition width is a
    numerical modeling choice, not an additional AM calibration.

    See Also
    --------
    get_profile_loss_blend, get_profile_thickness_factor, nozzle_blades, impulse_blades, get_shock_loss
    """
    Re = flow_parameters["Re_out"]
    Ma_rel_out = flow_parameters["Ma_rel_out"]
    Ma_rel_in = flow_parameters["Ma_rel_in"]
    beta_out = flow_parameters["beta_out"]
    s = geometry["pitch"]
    c = geometry["chord"]
    theta_in = geometry["leading_edge_angle"]
    t_max = geometry["maximum_thickness"]
    f_Re = get_reynolds_correction_factor(Re)
    f_Ma = 1 + 60 * (Ma_rel_out - 1) ** 2 * (Ma_rel_out > 1)
    Y_shock = get_shock_loss(flow_parameters, geometry, cascade_type_id)["Y_shock"]
    Kp, K2, K1 = get_compressible_correction_factors(Ma_rel_in, Ma_rel_out)
    angle_out_bis = math.smooth_abs(beta_out)
    Yp_reaction = nozzle_blades(s / c, angle_out_bis)
    Yp_impulse = impulse_blades(s / c, angle_out_bis)
    z = -theta_in / beta_out
    Y_p = get_profile_loss_blend(z, Yp_reaction, Yp_impulse)
    thickness_factor = get_profile_thickness_factor(z, t_max / c)
    Y_p = Y_p * thickness_factor
    Y_p = 0.914 * (2 / 3 * Y_p * Kp + Y_shock)
    f_Ma = 1.0
    Y_p = f_Re * f_Ma * Y_p
    return Y_p


# Width in the dimensionless coordinate z=-theta_in/beta_out. Turn on the
# thickness correction over 0 < z < 0.05 (center 0.025), matching value, slope
# and curvature to avoid the derivative jump of an abrupt exponent switch.
# This is a numerical smoothing choice; it does not limit large positive z.
PROFILE_THICKNESS_TRANSITION = 0.05


def get_profile_thickness_factor(z, thickness_chord):
    r"""Return the positive thickness correction with a C2 onset.

    ``PROFILE_THICKNESS_TRANSITION`` is the width delta of the onset in z,
    not a thickness ratio. With tau=t_max/c>0, delta=0.05, q=z/delta and
    S(q)=10*q**3-15*q**4+6*q**5, the factor is

    .. math::

        F_t=\begin{cases}
          1, & z\le0,\\
          1+S(z/\delta)\left[(\tau/0.2)^z-1\right], & 0<z<\delta,\\
          (\tau/0.2)^z, & z\ge\delta.
        \end{cases}

    The smoothstep's first two derivatives vanish at both ends. Thus the
    transition matches the value, slope and curvature of the outer factors.
    It is a convex blend of positive factors, introduces no thickness-factor
    overshoot, and equals 1 identically at tau=0.2. The outer branches are
    returned exactly. Inputs broadcast and complex-step perturbations are
    preserved; branch decisions use the real part of z.
    """
    z, thickness_chord = jnp.broadcast_arrays(jnp.asarray(z), jnp.asarray(thickness_chord))
    real_z = jnp.real(z)
    width = PROFILE_THICKNESS_TRANSITION
    power = (thickness_chord / 0.2) ** jnp.where(real_z > 0.0, z, 0.0)
    q = jnp.where((real_z > 0.0) & (real_z < width), z / width, 0.0)
    weight = q**3 * (10 + q * (-15 + 6*q))
    transition = 1 + weight * (power - 1)
    return jnp.where(real_z >= width, power, transition)


# Fractions of the reaction reference used only for negative-z extrapolation.
# As z tends to minus infinity, retain 80% of the reaction loss rather than
# allowing a negative loss or introducing a flat hard floor.
PROFILE_EXTRAPOLATION_ASYMPTOTE = 0.8
# At zero turning (z=-1, equal signed inlet/outlet angles), retain 90% of the
# reaction reference. Together these fractions set how gradually the loss
# decreases below z=0. They are engineering assumptions, not AM/KO fit data;
# they do not modify the interpolation over 0 <= z <= 1 or its upper extension.
PROFILE_EXTRAPOLATION_ZERO_TURNING = 0.9


def get_profile_loss_blend(z, reaction, impulse):
    r"""Interpolate positive AM references and continue toward low turning.

    For z=-theta_in/beta_out, use R*[f+(1-f)/(1+(z/L)^2)] below zero,
    with f=0.8, q=0.9 and L^2=(q-f)/(1-q)=1. Thus B(-1)=0.9R and
    the negative tail approaches 0.8R without a flat cutoff. For 0<=z<=1,
    B=(1-z^2)R+z^2I exactly; above one continue the quadratic only if I>=R.
    The negative-side derivative is -0.4R*z/(1+z^2)^2. The join at zero
    is C1, not C2. The fractions are explicit extrapolation assumptions,
    not additional empirical calibration.

    ``PROFILE_EXTRAPOLATION_ASYMPTOTE`` sets the limiting fraction f of the
    reaction reference; ``PROFILE_EXTRAPOLATION_ZERO_TURNING`` sets its
    fraction q at z=-1. Zero turning here describes the blade flow angles,
    not zero stage reaction.

    Inputs broadcast. Finite coordinates and positive finite references are
    required; upper extrapolation with I<R raises ValueError. Complex-step
    perturbations are preserved and branch decisions use real parts. Current
    execution uses NumPy; this validation does not promise JAX tracing/JIT.
    This helper returns the reference coefficient before thickness, Reynolds,
    compressibility, shock and the KO scaling factors.
    """
    dtype = complex if any(jnp.iscomplexobj(v) for v in (z, reaction, impulse)) else float
    z, reaction, impulse = jnp.broadcast_arrays(
        jnp.asarray(z, dtype=dtype), jnp.asarray(reaction, dtype=dtype),
        jnp.asarray(impulse, dtype=dtype))
    real_z = jnp.real(z)
    if not (jnp.isfinite(z).all() and jnp.isfinite(reaction).all()
            and jnp.isfinite(impulse).all()):
        raise ValueError("z and profile reference losses must be finite.")
    if jnp.any(jnp.real(reaction) <= 0) or jnp.any(jnp.real(impulse) <= 0):
        raise ValueError("Profile reference losses must be strictly positive.")
    if jnp.any((real_z > 1) & (jnp.real(impulse) < jnp.real(reaction))):
        raise ValueError("Upper profile extrapolation requires impulse >= reaction.")
    f, q = PROFILE_EXTRAPOLATION_ASYMPTOTE, PROFILE_EXTRAPOLATION_ZERO_TURNING
    length = jnp.sqrt((q-f)/(1-q))
    negative_z = jnp.where(real_z < 0, z, 0.0)
    if jnp.iscomplexobj(negative_z):
        inverse_squared = 1/(1+(negative_z/length)**2)
    else:
        inverse_squared = (1/jnp.hypot(1, negative_z/length))**2
    low = reaction*(f+(1-f)*inverse_squared)
    weight = jnp.where(real_z > 0, z, 0.0)**2
    interior_weight = jnp.where(real_z <= 1, weight, 1.0)
    interior = (1-interior_weight)*reaction+interior_weight*impulse
    upper = reaction+weight*(impulse-reaction)
    result = jnp.where(real_z < 0, low, jnp.where(real_z > 1, upper, interior))
    if not jnp.isfinite(result).all():
        raise ValueError("Profile extrapolation overflowed its numerical domain.")
    return result


# Multiplicative half-width around each Reynolds breakpoint (200,000 and
# 1,000,000): smooth over [Re_break/1.5, Re_break*1.5], using log(1.5) as the
# width in log-Re coordinates. This removes derivative kinks while preserving
# the empirical power laws outside the bands. It is not a multiplier on Re.
REYNOLDS_TRANSITION_FACTOR = 1.5


def get_reynolds_correction_factor(reynolds):
    r"""Return the positive, nonincreasing C2 candidate Reynolds correction.

    For :math:`P_w(x)` the compact positive-part upper envelope and
    :math:`w=\ln(1.5)`,

    .. math::

       f_\mathrm{Re}=\exp\left[
         0.4P_w(-\ln(\mathrm{Re}/(2\times10^5)))
         -0.2P_w(\ln(\mathrm{Re}/10^6))\right].

    Reynolds number is dimensionless, positive, and based on exit velocity and
    true chord. The empirical powers -0.4 and -0.2 and the central unit plateau
    are exact outside bands :math:`[\mathrm{Re}_0/1.5,1.5\mathrm{Re}_0]`.
    The lower transition spans :math:`1.33333\times10^5` to
    :math:`3\times10^5`; the upper spans :math:`6.66667\times10^5` to
    :math:`1.5\times10^6`. The correction is exactly one between the bands.
    This factor-1.5 margin, exposed as ``REYNOLDS_TRANSITION_FACTOR``, replaces
    the previous narrow :math:`\exp(\pm0.05)` margin. It is a numerical choice,
    not a published calibration or a change to the empirical exponents.

    Value, first derivative and curvature match at every band boundary.
    The logarithmic slope stays in [-0.4, 0] through the lower band and
    [-0.2, 0] through the upper band, so no new extrema are introduced.
    Relative to the hard correlation, the greatest deviations occur at the
    nominal thresholds: :math:`1.5^{0.075}-1` (about +3.09%) at 200,000 and
    :math:`1.5^{-0.0375}-1` (about -1.51%) at 1,000,000. Scalar and array-like
    inputs are accepted. Invalid nonpositive inputs remain nonfinite instead
    of being clipped.
    """
    reynolds = jnp.asarray(reynolds)
    half_width = jnp.log(REYNOLDS_TRANSITION_FACTOR)
    low = numerics.positive_part(-jnp.log(reynolds / 200000.0), half_width)
    high = numerics.positive_part(jnp.log(reynolds / 1000000.0), half_width)
    return jnp.exp(0.4 * low - 0.2 * high)


def get_secondary_loss(flow_parameters, geometry, cascade_type_id: int):
    r"""Return the modified KO secondary total-pressure loss coefficient.

    In consistent length units, let A=H/c be full-height/true-chord aspect
    ratio and b the meridional chord. The implemented assembly is

    .. math::

        G=1.2(0.0334)f_{\mathrm{AR}}(A)Z
        \frac{\cos\beta_{\mathrm{out}}}{\cos\theta_{\mathrm{in}}},
        \qquad Y_s=GK_s,

        \beta_m=\tan^{-1}\!\left[
          \frac{\tan\beta_{\mathrm{in}}+\tan\beta_{\mathrm{out}}}{2}
        \right],\qquad
        Z=4(\tan\beta_{\mathrm{in}}-\tan\beta_{\mathrm{out}})^2
          \frac{\cos^2\beta_{\mathrm{out}}}{\cos\beta_m},

        K_p=K_p(M_1,M_2),\quad K_3=(b/H)^2,\quad
        \widetilde K_3=C(K_3),\quad
        K_s=(1-\widetilde K_3)+\widetilde K_3K_p.

    :func:`get_secondary_aspect_ratio_factor` smooths the original
    square-root law over 1.5<A<2.5 to remove its singular slope at two,
    then applies the common C2 rational continuation C toward one.
    :func:`get_secondary_compressibility_factor` applies C to K3 as well:
    this prevents the unbounded (b/H)^2 multiplier from driving Ks negative
    at very small height. C is the identity through 0.8; see
    :func:`continue_secondary_factor` for its equation and matched derivatives.

    These joins and limits are engineering modifications to the correlation,
    not additional experimental calibration. They give 0<f_AR<=1 and
    Kp<=Ks<=1 on the valid domain. No separate 0.1 floor is applied to Ks,
    and no upper cap is applied to Ys. Zero turning gives exactly zero;
    large angle loading can give Ys greater than 0.5. Bounding the two
    factors does not guarantee monotonicity of their product with height.

    Parameters
    ----------
    flow_parameters : dict
        ``Ma_rel_in`` and ``Ma_rel_out`` are nonnegative row-frame Mach
        numbers; ``beta_in`` and ``beta_out`` are signed row-frame flow
        angles in degrees, measured from the meridional direction.
    geometry : dict
        Positive ``height`` H and true ``chord`` c, nonnegative
        ``meridional_chord`` b, and signed ``leading_edge_angle`` theta_in
        in degrees. Lengths must use a common unit. The inlet metal angle
        is independent of the inlet flow angle; equality means zero incidence.
    cascade_type_id : int
        Row identifier, conventionally 0 stator or 1 rotor. It is unused
        in this expression: identical row-frame inputs give identical losses.

    Returns
    -------
    float or ndarray
        Dimensionless secondary total-pressure loss coefficient. Inputs
        broadcast according to NumPy rules. Complex-step perturbations are
        preserved on the valid branches.

    Notes
    -----
    The forward-flow domain requires finite inputs, H>0, c>0, b>=0 and
    absolute inlet/outlet flow and inlet metal angles strictly below 90 deg.
    This helper does not validate or clip those inputs. No minimum aspect
    ratio or outlet angle is imposed; zero outlet angle is admissible.
    Pitch/chord cancels from the loading; no Reynolds, thickness or shock
    factor is applied here. Leading-edge shock is included in profile loss.

    The shared :func:`get_compressible_correction_factors` supplies the actual
    implemented Kp, including its own smooth continuations. The actual M2 is
    used above unity; the original paper instead froze secondary loss at
    sonic exit. For fixed geometry, dKs/dKp=K3_eff, so this secondary
    continuation creates no separate Mach-sensitivity plateau, while
    stationary points or constant branches in Kp can remain.
    The new Moustapha model uses this KO component before its incidence
    multiplier; Benner uses its own secondary-loss expression.
    """
    Ma_rel_out = flow_parameters["Ma_rel_out"]
    Ma_rel_in = flow_parameters["Ma_rel_in"]
    beta_out = flow_parameters["beta_out"]
    beta_in = flow_parameters["beta_in"]
    b = geometry["meridional_chord"]
    H = geometry["height"]
    c = geometry["chord"]
    theta_in = geometry["leading_edge_angle"]
    Kp = get_compressible_correction_factors(Ma_rel_in, Ma_rel_out)[0]
    Ks = get_secondary_compressibility_factor(Kp, b / H)
    angle_m = math.arctand((math.tand(beta_in) + math.tand(beta_out)) / 2)
    Z = (
        4
        * (math.tand(beta_in) - math.tand(beta_out)) ** 2
        * math.cosd(beta_out) ** 2
        / math.cosd(angle_m)
    )
    far = get_secondary_aspect_ratio_factor(H / c)
    return 1.2 * Ks * 0.0334 * far * Z * math.cosd(beta_out) / math.cosd(theta_in)


# Dimensionless raw-factor level where the upper continuation begins. Keep
# factors through 0.8 unchanged, then bend smoothly toward the unit limit.
SECONDARY_FACTOR_CONTINUATION_START = 0.8
# Common limit for the aspect-ratio factor and K3=(meridional_chord/height)^2.
# Bounding K3 by one prevents a negative secondary compressibility factor;
# bounding the aspect factor limits amplification for very short blades.
# These are engineering extrapolation choices, not fitted KO constants or a
# bound on the final secondary loss. The continuation matches two derivatives.
SECONDARY_FACTOR_ASYMPTOTE = 1.0


def continue_secondary_factor(raw_factor):
    r"""Return the common C2 upper continuation for secondary-loss factors.

    For a nonnegative dimensionless raw factor x, let a=0.8, L=1,
    d=L-a=0.2 and t=(x-a)/d. The implemented mathematical function is

    .. math::

        C(x)=\begin{cases}
          x,&x\le a,\\
          L-\dfrac{d}{1+t+t^2},&x>a,
        \end{cases}
        \qquad
        C'(x)=\begin{cases}
          1,&x\le a,\\
          \dfrac{1+2t}{(1+t+t^2)^2},&x>a.
        \end{cases}

    At x=a the value, slope and curvature are (a,1,0), matching the
    identity branch. For x>a the curvature is

    .. math::

        C''(x)=-\frac{6t(1+t)}{d(1+t+t^2)^3}.

    Thus C is C2, increasing and no larger than either x or one for x>=0.
    It preserves factors through 0.8 exactly and approaches one without a
    constant branch in exact arithmetic. The positive slope avoids an abrupt
    loss of sensitivity at a hard cutoff. Float rounding can still return
    exactly one and extremely small derivatives can underflow.

    The same continuation bounds the patched aspect factor and K3 separately,
    before multiplication by angle loading. It does not bound the final Ys.
    The thresholds are engineering choices, not values calibrated by KO.
    ``SECONDARY_FACTOR_CONTINUATION_START`` is a; the limiting factor L is
    ``SECONDARY_FACTOR_ASYMPTOTE``. Their difference sets the transition scale.

    Parameters
    ----------
    raw_factor : float or array_like
        Dimensionless raw factor. Physical callers supply finite x>=0.
        Negative x is returned unchanged; this is not an input-domain check.

    Returns
    -------
    float or ndarray
        Continued factor. The reciprocal evaluation u=d/(d+max(x-a,0))
        uses L-d*u**2/(1-u+u**2), avoiding a large t squared. Branch decisions
        use real parts and preserve complex-step perturbations.
    """
    raw = jnp.asarray(raw_factor)
    active = jnp.real(raw) > SECONDARY_FACTOR_CONTINUATION_START
    delta = SECONDARY_FACTOR_ASYMPTOTE - SECONDARY_FACTOR_CONTINUATION_START
    excess = jnp.where(active, raw - SECONDARY_FACTOR_CONTINUATION_START, 0.)
    inverse = delta / (delta + excess)
    continued = SECONDARY_FACTOR_ASYMPTOTE - delta*inverse**2/(1-inverse+inverse**2)
    return jnp.where(active, continued, raw)


def continue_secondary_aspect_ratio_factor(raw_factor):
    r"""Apply the common C2 continuation to the patched aspect-ratio factor.

    .. math::

        f_{\mathrm{AR}}=C(f_0),\qquad
        \frac{d f_{\mathrm{AR}}}{d f_0}=C'(f_0).

    f0 is the intermediate factor after the 1.5--2.5 Hermite join. Values
    through 0.8 are unchanged; larger values approach one smoothly to limit
    low-height amplification. This wrapper uses :func:`continue_secondary_factor`
    so that the aspect and compressibility continuations share the same law.
    Scalar/array inputs, domain and floating-point limitations follow that helper.
    """
    return continue_secondary_factor(raw_factor)


def get_secondary_compressibility_factor(profile_factor, meridional_chord_height):
    r"""Return Ks using a smoothly bounded endwall-compressibility influence.

    For r=b/H, where b is meridional chord and H is full blade height,
    use the implemented profile correction Kp and the common continuation C:

    .. math::

        K_3=r^2,\qquad \widetilde K_3=C(K_3),\qquad
        K_s=(1-\widetilde K_3)+\widetilde K_3K_p,

        \frac{\partial K_s}{\partial K_p}=\widetilde K_3,\qquad
        \frac{\partial K_s}{\partial r}
          =2r(K_p-1)C'(r^2).

    For Kp in [0,1], 0<=K3_eff<=1 makes Ks a weighted average of one and
    Kp, so Kp<=Ks<=1. There is no independent Ks floor. Tall blades give
    small influence and Ks near one; as height decreases, Ks approaches Kp.
    A larger r cannot increase Ks at fixed Kp. The product Ys=G*Ks can
    nevertheless have a reversal with height because G also varies.

    The original KO expression 1-r**2*(1-Kp) is recovered for r**2<=0.8.
    Above that threshold C matches value, slope and curvature, then tends to
    one. This avoids negative Ks and an arbitrary floor-induced plateau when
    extending KO to short blades. The 0.8 onset and unit asymptote are
    engineering choices; the modification begins at b/H=sqrt(0.8) and also
    changes the upper part of the original Figure 13 range.

    Parameters
    ----------
    profile_factor : float or array_like
        Kp in [0,1], supplied by :func:`get_compressible_correction_factors`
        in the complete secondary-loss function.
    meridional_chord_height : float or array_like
        Dimensionless ratio b/H>=0. Positive height is required by the model.

    Returns
    -------
    float or ndarray
        Dimensionless Ks. Inputs broadcast and complex-step perturbations
        are preserved; this helper does not validate or clip invalid inputs.
        Very large ratios can overflow when squared; the supported audit
        range includes AR down to 0.01, not arbitrary floating-point extremes.
    """
    influence = continue_secondary_factor(jnp.asarray(meridional_chord_height)**2)
    return (1-influence)+influence*jnp.asarray(profile_factor)


def get_secondary_aspect_ratio_factor(aspect_ratio):
    r"""Return the C2-regularized KO aspect factor with a smooth unit limit.

    For positive aspect ratio A=H/c (full height/true chord), first define
    the uncontinued low-height branch l(A) and patched factor f0:

    .. math::

        l(A)=\frac{1-0.25\sqrt{2-A}}{A},\qquad
        f_0(A)=\begin{cases}
          l(A),&0<A\le1.5,\\
          H_5(A),&1.5<A<2.5,\\
          1/A,&A\ge2.5,
        \end{cases}
        \qquad f_{\mathrm{AR}}(A)=C(f_0(A)).

    H5 is the unique quintic Hermite polynomial matching the original
    branch value, first derivative and second derivative at a=1.5 and b=2.5:

    .. math::

        H_5^{(j)}(a)=l^{(j)}(a),\qquad
        H_5^{(j)}(b)=(1/A)^{(j)}\big|_{A=b},\qquad j=0,1,2.

    The original KO branches meet at A=2 with a singular left slope. The
    1.5--2.5 patch removes that singularity and matches slope and curvature
    for gradient-based calculations. Its positivity and decreasing trend
    have been checked for these endpoints; they are not general properties
    of every quintic interpolant.

    :func:`continue_secondary_aspect_ratio_factor` applies the common C2 law
    C: identity through f0=0.8, then a gradual approach to one. It controls
    the original 1/A growth without a hard cutoff or constant upper branch.
    This continuation starts at A=0.9261693781, so it does not overlap the
    Hermite patch. Original values are recovered outside both modifications.
    The joins and limits are numerical modeling choices, not a revised fit
    to experimental data. In particular, a bounded factor is not a cap on Ys.

    Parameters
    ----------
    aspect_ratio : float or array_like
        Finite positive, dimensionless A=H/c. No lower clipping or input
        validation is performed; A=0 is outside the model domain.

    Returns
    -------
    float or ndarray
        Positive, decreasing factor no larger than one. Inputs broadcast
        and branch decisions use real A to preserve complex-step derivatives.
        At extreme small A the limit can round to one; see
        :func:`continue_secondary_factor`.
    """
    aspect_ratio = jnp.asarray(aspect_ratio)
    real_ar = jnp.real(aspect_ratio)
    a, b = 1.5, 2.5
    q = 2 - a
    n = 1 - 0.25 * jnp.sqrt(q)
    dn, d2n = 0.125 / jnp.sqrt(q), 0.0625 / q**1.5
    y0, dy0 = n / a, dn / a - n / a**2
    d2y0 = d2n / a - 2 * dn / a**2 + 2 * n / a**3
    patch = numerics.quintic_hermite(
        aspect_ratio, a, b, y0, 1 / b, dy0, -1 / b**2, d2y0, 2 / b**3
    )
    # The inactive low branch is evaluated away from its singular endpoint.
    low_ar = jnp.where(real_ar <= a, aspect_ratio, a)
    low = (1 - 0.25 * jnp.sqrt(2 - low_ar)) / low_ar
    raw_factor = jnp.where(
        real_ar <= a, low, jnp.where(real_ar >= b, 1 / aspect_ratio, patch)
    )
    return continue_secondary_aspect_ratio_factor(raw_factor)


def get_trailing_edge_loss(flow_parameters, geometry):
    r"""Return the integrated KO trailing-edge pressure-loss coefficient.

    For x=t_TE/o and z=-theta_in/beta_out, evaluate the Figure 14 energy
    increment d using get_trailing_edge_kinetic_energy_loss. Convert
    it with KO equation (18), using outlet row-frame Mach and gamma:

    .. math::

        q=(\gamma-1)M_e^2/2,\quad n=\gamma/(\gamma-1),\quad A=d/(1-d),
        \qquad
        Y_\mathrm{TE}=\frac{(1-qA)^{-n}-1}{1-(1+q)^{-n}}.

    At M=0 the exact analytic limit is d/(1-d), evaluated without 0/0
    through logarithmic relative functions. For M<=0.9, M_e=M. Above 0.9,

    .. math::

        M_e=0.9+0.1\,\frac{t}{\sqrt{1+t^2}},\qquad t=(M-0.9)/0.1.

    This C2 continuation tends to one without a finite Mach plateau. It
    prevents the gas-conversion singularity during extrapolation; it also
    modifies the upper tenth of the subsonic range. The tail is a numerical
    choice, not prescribed by KO. Additional supersonic drag belongs to the
    separate profile-loss model. No extra final-loss floor or cap is applied.

    Parameters
    ----------
    flow_parameters : dict
        beta_out: finite nonzero signed exit flow angle in degrees;
        Ma_rel_out: finite nonnegative row-frame outlet Mach;
        gamma_out: finite ideal-gas heat-capacity ratio, 1<gamma<=5/3.
    geometry : dict
        Finite nonnegative trailing_edge_thickness, finite positive opening
        in the same length units, and finite signed leading_edge_angle
        (inlet metal angle, degrees). Inlet flow angle is not used.

    Returns
    -------
    float or ndarray
        Dimensionless stagnation-pressure loss coefficient; valid inputs
        broadcast and retain complex-step derivatives. Invalid domains
        return NaN. Zero thickness returns exactly zero for valid inputs.

    Notes
    -----
    All empirical constants and numerical continuation settings live in
    _trailing_edge. That module contains full equations for the cubic
    references, rational thickness tails, smoothed signed-square weight
    and bounded z extrapolation. No runtime study/data dependency exists.
    The reference fits use 42 scan readings; the continuation thresholds
    are explicit modeling choices. With 0<=d<=0.3 and M_e<=1, the conversion
    margin 1-q*A is at least 6/7 on the specified gas domain.

    The function now uses the Mach-dependent paper conversion instead of
    applying its zero-Mach limit at every Mach. New KO, Moustapha and Benner
    share this helper. The original loss_models package is unchanged.

    z has no unique value when beta_out=0; no epsilon or arbitrary nozzle
    assignment hides that ambiguity. This wrapper returns NaN there.
    For prescribed finite-z limits at axial exit use
    trailing_edge.trailing_loss(x, z, Mach, gamma) directly.
    """
    d = get_trailing_edge_kinetic_energy_loss(flow_parameters, geometry)
    return trailing_edge.pressure_loss(
        d, flow_parameters["Ma_rel_out"], flow_parameters["gamma_out"])


def get_trailing_edge_kinetic_energy_loss(flow_parameters, geometry):
    r"""Return the fitted/blended KO Figure 14 energy increment d, before conversion.

    Put x=t_TE/o>=0 and z=-theta_in/beta_out. The fitted nozzle R is higher
    than the impulse I (the reverse ordering from the profile references).
    At 0.05<=z<=1 the blend is exactly KO equation (17):

    .. math::

        d=(1-z^2)R+z^2 I.

    For |z|<0.05, replace z*abs(z) by the odd C2 polynomial
    W=w^2*(3*q/8+3*q^3/4-q^5/8), q=z/w, w=0.05. It preserves both
    reference endpoints and a small nonzero slope at z=0. For W<0 use
    d=R+0.2*R*S((R-I)*(-W)/(0.2*R)); for W>1 use
    d=I-0.2*I*S((R-I)*(W-1)/(0.2*I)), S(t)=t/sqrt(1+t^2).
    Values, slopes and curvatures match at every join.

    The cubic references recover the measured chart on 0<=x<=0.4 and
    continue monotonically with C2 rational tails toward R=0.25, I=0.13.
    Consequently 0.8*I<=d<=1.2*R<=0.3. Negative z increases energy loss;
    z>1 can reduce it by at most 20% below the impulse reference.
    These extrapolation allowances are numerical choices, not KO data.

    Zero thickness gives exactly zero with finite right thickness
    derivatives; reduced reference polynomials avoid 0/0 at x=0.
    Invalid thickness/opening/angles return NaN. beta_out must be nonzero:
    prescribed finite-z axial paths can be evaluated by
    trailing_edge.energy_loss(x, z). Inactive divisions are protected.
    """
    thickness = jnp.asarray(geometry["trailing_edge_thickness"])
    opening = jnp.asarray(geometry["opening"])
    theta = jnp.asarray(geometry["leading_edge_angle"])
    beta = jnp.asarray(flow_parameters["beta_out"])
    valid = (jnp.isfinite(thickness) & jnp.isfinite(opening)
             & jnp.isfinite(theta) & jnp.isfinite(beta)
             & (jnp.real(thickness) >= 0) & (jnp.real(opening) > 0)
             & (jnp.real(beta) != 0))
    x = jnp.where(valid, thickness, 0.)/jnp.where(valid, opening, 1.)
    z = -jnp.where(valid, theta, 0.)/jnp.where(valid, beta, 1.)
    value = trailing_edge.energy_loss(x, z)
    return jnp.where(valid, value, jnp.nan)


# KO equation (20): effective seal gap / true chord, with the published exponent.
# Below 0.1% chord, a C2 endpoint patch removes the fractional-power singularity.
TIP_CLEARANCE_EXPONENT = 0.78
TIP_CLEARANCE_TRANSITION = 1e-3
# Numerical continuations, not parameters fitted to KO measurements.
TIP_CLEARANCE_ANGLE_START = 80.0
TIP_CLEARANCE_LOADING_START = 16.0
TIP_CLEARANCE_LOADING_LIMIT = 32.0
TIP_CLEARANCE_AR_START = 1.0
TIP_CLEARANCE_AR_LIMIT = 2.0


def get_tip_clearance_gap_factor(gap_chord):
    r"""Evaluate the KO fractional gap law with a finite zero-gap slope.

    For effective gap ratio :math:`x=k'/c\ge0`, exponent :math:`p=0.78`,
    transition :math:`\delta=10^{-3}` and :math:`q=x/\delta`, use

    .. math::

        F(x)=\begin{cases}
          \delta^p(aq+bq^2+dq^3),&0\le x<\delta,\\
          x^p,&x\ge\delta,
        \end{cases}
        \qquad
        a=\frac{(p-2)(p-3)}{2},\quad
        b=(p-1)(3-p),\quad d=\frac{(p-1)(p-2)}{2}.

    The cubic matches the published power law's value, slope and curvature
    at delta. It is positive, increasing and concave for positive x, lies
    below the unsmoothed law inside the patch, and gives F(0)=0 with finite
    nonzero right derivative :math:`F'(0)=a\delta^{p-1}`. The right curvature
    at zero is also finite. This avoids both the original infinite gradient
    and an artificial residual loss at zero gap. The outer correlation is
    recovered exactly at and above 0.1% chord; the patch width is a numerical
    choice, not a measured clearance threshold or a loss cap.

    Inputs are dimensionless scalars or arrays. Negative gaps return NaN;
    zero is a physical domain boundary, so differentiation there is one-sided
    (no smooth extension to negative gaps is claimed). Branch selection uses
    the real part, preserving complex-step derivatives on the valid domain.
    Inactive branches use safe inputs to avoid evaluating a fractional power
    at zero or at negative values.
    """
    x = jnp.asarray(gap_chord)
    p, delta = TIP_CLEARANCE_EXPONENT, TIP_CLEARANCE_TRANSITION
    real_x = jnp.real(x)
    patch_active = (real_x >= 0.0) & (real_x < delta)
    q = jnp.where(patch_active, x / delta, 0.0)
    a = (p - 2) * (p - 3) / 2
    b = (p - 1) * (3 - p)
    d = (p - 1) * (p - 2) / 2
    patch = delta**p * q * (a + q * (b + d * q))
    power = jnp.where(real_x >= delta, x, delta)**p
    result = jnp.where(real_x >= delta, power, patch)
    return jnp.where(real_x >= 0.0, result, jnp.nan)


def get_tip_clearance_aspect_ratio_factor(aspect_ratio):
    r"""Return a bounded C2 continuation of the clearance multiplier 1/AR.

    With A=H/c>=0, a=1 (onset), L=2 (limit), D=L-1/a and
    u=1-A/a, use

    .. math::

        G(A)=\begin{cases}
          L-\dfrac{D^3 A^2}{D^2 A^2+D A u+u^2},&0\le A<a,\\
          1/A,&A\ge a.
        \end{cases}

    For A>0 below the join this is the rational continuation
    L-D/(1+t+t^2), t=(1/A-1/a)/D. The expression above cancels the
    singular inverse aspect ratio before evaluation: G(0)=2, G'(0)=0,
    G''(0)=-2 for the default constants. Value, slope and curvature match
    1/A at A=1; G decreases strictly on A>0, without a finite plateau.
    Normal aspect ratios A>=1 are unchanged. The unit onset and limit two
    are numerical choices for very short blades, not a KO calibration.
    This is the clearance c/H multiplier, not the secondary-loss f_AR law.

    Zero is accepted as a mathematical endpoint with finite right
    derivatives, not a physical zero-height blade. Negative/nonfinite
    inputs return NaN. Complex-step perturbations and broadcasting work.
    """
    A = jnp.asarray(aspect_ratio)
    valid = jnp.isfinite(A) & (jnp.real(A) >= 0)
    a, limit = TIP_CLEARANCE_AR_START, TIP_CLEARANCE_AR_LIMIT
    d = limit - 1/a
    low = valid & (jnp.real(A) < a)
    x = jnp.where(low, A, 0.)
    u = 1-x/a
    continued = limit-d**3*x*x/(d*d*x*x+d*x*u+u*u)
    inverse = 1/jnp.where(valid & ~low, A, a)
    return jnp.where(valid, jnp.where(low, continued, inverse), jnp.nan)


def _tip_clearance_tangent(angle):
    """Continue tan(angle) quadratically beyond +/-80 degrees, with C2 joins.

    The odd continuation is monotone and has no pole or periodic wrapping.
    The angles themselves are never clipped. This private coordinate is
    used consistently for both the tangent and outlet cosine in loading.
    """
    angle = jnp.asarray(angle)
    sign = jnp.where(jnp.real(angle) < 0, -1., 1.)
    magnitude = sign*angle
    a = TIP_CLEARANCE_ANGLE_START
    outer = jnp.real(magnitude) > a
    d = jnp.where(outer, magnitude-a, 0.)
    t0 = math.tand(a)
    t1 = (jnp.pi/180)*(1+t0*t0)
    t2 = 2*(jnp.pi/180)**2*t0*(1+t0*t0)
    tail = sign*(t0+t1*d+.5*t2*d*d)
    # Never evaluate the tangent on its singular/inactive outer branch.
    original = math.tand(jnp.where(outer, 0., angle))
    return jnp.where(outer, tail, original)


def get_tip_clearance_loading(beta_in, beta_out):
    r"""Return bounded, C2 KO clearance loading without angle clipping.

    Let T(beta)=tan(beta) for |beta|<=a=80 degrees. Beyond a, use the
    odd quadratic Taylor continuation
    T(beta)=sign(beta)*(t0+t1*d+t2*d^2/2), d=|beta|-a, where
    t0=tan(a), t1=(pi/180)*(1+t0^2), t2=2*(pi/180)^2*t0*(1+t0^2).
    This removes tangent poles before applying the loading continuation.
    For U=T(beta_in), V=T(beta_out), define

    .. math::

        Z_0=\frac{4(U-V)^2}{1+V^2}
                \sqrt{1+\left(\frac{U+V}{2}\right)^2},\qquad
        Z=\begin{cases}
          Z_0,&Z_0\le Z_s,\\
          Z_L-\dfrac{D}{1+t+t^2},&
            Z_0>Z_s,\quad t=(Z_0-Z_s)/D,
        \end{cases}

    Here Z_s=16, Z_L=32 and D=Z_L-Z_s. Both joins in T and the loading
    join are C2. Z is nonnegative, bounded by 32, and has no finite-input
    saturation plateau in exact arithmetic. The thresholds are numerical
    modeling choices, not KO data. Z_s=16 preserves the entire original
    symmetric-impulse loading range, 16*sin(beta_out)^2.

    The original KO loading is recovered when both angles are within
    +/-80 degrees and Z0<=16. With the continued tangents, use the
    consistent cosine squared 1/(1+V^2); mixing the physical cosine with
    a continued tangent would create artificial zeros at 90 degrees.
    Simultaneous angle reflection preserves Z. Equal angles give exactly
    zero loading, including at zero and beyond 90 degrees. At both angles
    zero the partial angle derivatives vanish. No inlet/outlet angle ratio
    z is ever computed, and there is no 0/0 at axial flow.

    Scaled arithmetic evaluates the bounded result without forming large
    raw loading. Inputs are finite signed row-frame angles in degrees;
    nonfinite angles return NaN. Broadcasting and complex-step derivatives
    are supported. Beyond the joins this is a numerical continuation for
    optimizer excursions, not an aerodynamic model of reversed flow.
    """
    bi, bo = jnp.broadcast_arrays(jnp.asarray(beta_in), jnp.asarray(beta_out))
    valid = jnp.isfinite(bi) & jnp.isfinite(bo)
    U = _tip_clearance_tangent(jnp.where(valid, bi, 0.))
    V = _tip_clearance_tangent(jnp.where(valid, bo, 0.))
    # Z0=N/Q. The common scale cancels algebraically and hence preserves
    # complex-step derivatives even though its selection uses real parts.
    scale = jnp.maximum(1., jnp.maximum(jnp.abs(jnp.real(U)), jnp.abs(jnp.real(V))))
    u, v, h = U/scale, V/scale, 1/scale
    N = 4*(u-v)**2*jnp.sqrt(h*h+((u+v)/2)**2)
    Q = (h*h+v*v)*h
    start, limit = TIP_CLEARANCE_LOADING_START, TIP_CLEARANCE_LOADING_LIMIT
    d = limit-start
    outer = jnp.real(N) > start*jnp.real(Q)
    raw = jnp.where(outer, 0., N)/jnp.where(outer, 1., Q)
    # vtail=1/(1+t); no large raw ratio or squared t is formed.
    numerator = d*Q
    denominator = N+(d-start)*Q
    vtail = jnp.where(outer, numerator, 0.)/jnp.where(outer, denominator, 1.)
    tail = limit-d*vtail**2/(1-vtail+vtail**2)
    return jnp.where(valid, jnp.where(outer, tail, raw), jnp.nan)


def get_tip_clearance_loss(flow_parameters, geometry, cascade_type_id: int):
    r"""Return the KO shrouded-rotor clearance loss with C2 continuations.

    Equation (20) of :cite:`kacker_mean_1982` uses effective gap divided by
    true chord, not by blade height. With AR=H/c and x=k'/c, use

    .. math::

        Y_\mathrm{cl}=B\,Z\,G(\mathrm{AR})\,F(x),\qquad
        B=\begin{cases}0,&\text{stator},\\0.37,&\text{rotor}.\end{cases}

    F is :func:`get_tip_clearance_gap_factor`: exactly x**0.78 for x>=0.001,
    with a C2 cubic patch below that ratio. It removes the infinite zero-gap
    derivative while retaining exactly zero loss and a finite positive gap
    sensitivity for a loaded rotor.

    G is :func:`get_tip_clearance_aspect_ratio_factor`: exactly 1/AR
    for AR>=1, with a C2 continuation toward G(0)=2 below one. Z is
    :func:`get_tip_clearance_loading`: the original loading through Z=16
    for angles within +/-80 degrees, followed by a C2 tail toward 32.
    Continued tangents remove the +/-90-degree poles without clipping.
    See those helpers for the implemented equations and their rationale.

    Parameters
    ----------
    flow_parameters : dict
        Signed row-frame beta_in and beta_out, in degrees from the
        meridional direction. Equal angles give exactly zero loading.
    geometry : dict
        Positive chord (true chord), nonnegative height (full mean blade
        height), and nonnegative tip_clearance (effective gap k'). All
        lengths must share units and may broadcast. Equation (21) defines
        k'=k/N**0.42 for N seals. There is no seal-count input: a physical gap
        corresponds to one seal; for multiple seals the caller supplies
        the effective gap. No seal count is inferred from other geometry.
    cascade_type_id : int
        Row identifier: 0 for stator (zero clearance loss), 1 for rotor.

    Returns
    -------
    float or ndarray
        Tip-clearance pressure-loss coefficient, with no separate upper cap.

    Notes
    -----
    The paper uses a separate efficiency-penalty treatment for unshrouded
    blades (equation 19). Applying this helper to all rotors retains the
    existing interface assumption; it is not a validated unshrouded model.
    New KO, Moustapha and Benner all share this helper.

    The component has finite right derivatives at both zero gap and zero
    height, including their joint origin, for positive chord. It never
    divides by height or computes z=-beta_in/beta_out. Thus axial flow is
    evaluated directly, with zero loss and zero angle derivatives when both
    angles are zero. Zero height is a mathematical continuation endpoint,
    not a valid physical turbine row; other components need not accept it.
    At fixed gap/chord the AR=0 limit is B*Z*2*F(k'/c). At fixed gap/height,
    k' goes to zero with H, so the loss instead tends to zero.

    Negative gaps/heights, nonpositive chord, nonfinite inputs or invalid
    row IDs return NaN. All continuations are numerical choices, not extra
    empirical data. No separate final-loss cap is imposed. There are no
    separate Mach, Reynolds or secondary-loss multipliers here.
    """
    beta_in = jnp.asarray(flow_parameters["beta_in"])
    beta_out = jnp.asarray(flow_parameters["beta_out"])
    H = jnp.asarray(geometry["height"])
    c = jnp.asarray(geometry["chord"])
    t_cl = jnp.asarray(geometry["tip_clearance"])
    row = jnp.asarray(cascade_type_id)
    valid = (jnp.isfinite(H) & jnp.isfinite(c) & jnp.isfinite(t_cl)
             & jnp.isfinite(beta_in) & jnp.isfinite(beta_out)
             & (jnp.real(H) >= 0) & (jnp.real(c) > 0) & (jnp.real(t_cl) >= 0)
             & ((row == 0) | (row == 1)))
    safe_c = jnp.where(valid, c, 1.)
    ar = jnp.where(valid, H, 0.)/safe_c
    gap = jnp.where(valid, t_cl, 0.)/safe_c
    Z = get_tip_clearance_loading(beta_in, beta_out)
    Y_cl = jnp.where(
        row == _CASCADE_TO_ID["stator"],
        0.0,
        0.37 * Z * get_tip_clearance_aspect_ratio_factor(ar) * get_tip_clearance_gap_factor(gap),
    )
    return jnp.where(valid, Y_cl, jnp.nan)


# Current analytical calibration to independently digitized AM Figure 4.
# Coefficients are fixed literals; production has no study/data dependency.
_AM_REFERENCE_COEFFICIENTS = {'nozzle': (-4.0128686886354545,
            -3.052868603548856,
            0.20555012945295192,
            -1.0999753446815,
            -0.17461295872612137,
            -0.25454345075616785,
            -0.10633814163886038,
            -2.233374449845362,
            0.33190392665656865,
            -0.16904474455190652,
            -4.322642585955608,
            0.6264285690866168,
            0.2713471113970221),
 'impulse': (-2.736522086621256,
             -2.896573715093353,
             -0.283643507785789,
             -1.1946056522838966,
             -0.3588246016451461,
             -0.22502441783637991,
             -0.10952710904031018,
             -2.150331405768899,
             -0.9754071725098622,
             -0.5824544280040551,
             -2.5172323544345763,
             0.7804913595799361,
             0.14100852926587318)}


def _am_reference_loss(r_sc, angle_out, family):
    r"""Evaluate the bounded analytical AM reference surface.

    With x=s/c, t=(alpha-center)/scale and quadratic polynomials P,
    x0=exp(P_x0), A=exp(P_A), B=exp(P_B), and
    m=exp(log_m0)+exp(log_delta)*w*log(1+exp((t-tc)/w)), w=exp(log_w).
    The raw surface is m+(A+B/x)*(x-x0)^2. Above T=.25 it continues as
    T+D*(Yr-T)/sqrt(D^2+(Yr-T)^2), D=.25. The join is C2.

    Cancel the raw 1/x algebraically with N=m*x+(A*x+B)*(x-x0)^2 and
    R=N-T*x. The bounded branch T+D*R/sqrt(R^2+D^2*x^2) gives .5 at
    x=0, zero first spacing derivative, and finite right-hand curvature.
    Branches use real parts so complex-step derivatives remain supported.
    """
    x = jnp.asarray(r_sc)
    if jnp.any(jnp.real(x) < 0):
        raise ValueError("Pitch/chord must be nonnegative.")
    center, scale = (60.0, 20.0) if family == "nozzle" else (55.0, 15.0)
    t = (jnp.asarray(angle_out) - center) / scale
    p = _AM_REFERENCE_COEFFICIENTS[family]
    log_m0, log_delta, tc, log_w = p[:4]
    w = jnp.exp(log_w)
    z = (t - tc) / w
    shift = jnp.maximum(jnp.real(z), 0.0)
    softplus = shift + jnp.log(jnp.exp(-shift) + jnp.exp(z - shift))
    m = jnp.exp(log_m0) + jnp.exp(log_delta) * w * softplus
    x0, A, B = [jnp.exp(p[i] + t * (p[i + 1] + t * p[i + 2]))
                for i in (4, 7, 10)]
    numerator = m * x + (A * x + B) * (x - x0)**2
    remainder = numerator - 0.25 * x
    bounded = 0.25 + 0.25 * remainder / jnp.sqrt(remainder**2 + (0.25 * x)**2)
    safe_x = jnp.where(jnp.real(x) == 0, 1.0, x)
    return jnp.where(jnp.real(remainder) > 0, bounded, numerator / safe_x)


def nozzle_blades(r_sc, angle_out):
    r"""Return the analytical Ainley--Mathieson nozzle reference Yp.

    ``r_sc`` is nonnegative pitch / true chord. ``angle_out`` is the positive
    outlet-angle magnitude from axial, in degrees; no complementary-angle
    conversion is applied. Scalars and broadcast arrays are supported with
    the existing NumPy backend. Source angles are 40--80 degrees and measured
    spacing is approximately .3--1.1, with shorter high-angle curves.

    Reference conditions: zero incidence, maximum thickness/chord=.2,
    Re=2e5 and Mach<.6. See ``_am_reference_loss`` for the expression and
    fixed coefficients above. The numerical continuation bounds only this
    reference curve by .5, leaves all digitized predictions unchanged, and
    preserves low-spacing angle ordering for 0<x<=.5. Outside chart coverage
    it is unvalidated extrapolation. It supersedes the former Aungier blends;
    the complete corrected KO component and row total are not capped here.
    """
    return _am_reference_loss(r_sc, angle_out, "nozzle")


def impulse_blades(r_sc, angle_out):
    r"""Return the analytical Ainley--Mathieson impulse reference Yp.

    Inputs, reference conditions, bounded continuation and broadcasting follow
    ``nozzle_blades``. Source angles are 40--70 degrees and measured spacing
    approximately .3--1.0. The signed reference has beta_in=-beta_out.
    Large-spacing angle crossings remain outside chart coverage; the numerical
    ceiling supplies no additional aerodynamic evidence.
    """
    return _am_reference_loss(r_sc, angle_out, "impulse")


def _continue_profile_compressibility(Kp_raw):
    """Continue a raw profile factor below 0.6 with a C2 rational tail.

    For the physical raw-factor domain [0, 1], the result increases strictly
    from 0.4 + 0.2/13 to 1, with derivative between 7/169 and 1. It equals
    the input at and above 0.6. The formal lower asymptote is 0.4, attained
    only as the mathematical continuation takes the raw input to -infinity.
    Real-part branch selection retains complex-step derivatives.
    """
    raw = jnp.asarray(Kp_raw)
    continued = jnp.real(raw) < .6
    # Restrict the inactive operand as well as selecting the result.
    t = jnp.where(continued, (.6-raw)/.2, 0.)
    tail = .4+.2/(1+t+t*t)
    return jnp.where(continued, tail, raw)


def get_compressible_correction_factors(Ma_rel_in, Ma_rel_out):
    r"""
    Compute bounded Kacker--Okapuu factors with gradual asymptotic tails.

    The original outlet-Mach reference is 1 below M2=0.2, decreases as
    1.25*(1-M2) up to M2=1, and is continued as zero above one. Figure 9 of
    Kacker and Okapuu (1982) gives K2=(M1/M2)**2 for ratios from zero to one.
    The following continuations are numerical modeling choices, not additional
    empirical calibration.

    With M=M2, the implemented outlet-Mach factor is

    .. math::

        K_1(M)=\begin{cases}
          1-0.005(M/0.2)^2, & 0\le M\le0.15,\\
          H(M), & 0.15<M<0.25,\\
          1.25(1-M), & 0.25\le M\le0.9,\\
          \dfrac{0.125}{1+t+t^2},\quad t=(M-0.9)/0.1, & M>0.9.
        \end{cases}

    H is the quintic Hermite interpolant matching value, slope and curvature
    (0.9971875,-0.0375,-0.25) at 0.15 and (0.9375,-1.25,0) at 0.25.
    All joins are C2. K1 decreases strictly for positive Mach; K1(0)=1
    and K1'(0)=0. The upper tail matches the linear value, slope and
    curvature at 0.9, remains positive, and tends to zero.

    For r=M1/M2, a=0.8**2=0.64, d=1-a=0.36 and u=r**2-a,

    .. math::

        K_2=\begin{cases}
          r^2, & 0\le r\le0.8,\\
          a+\dfrac{d u}{\sqrt{d^2+u^2}}, & r>0.8.
        \end{cases}

    This C2 saturation stays below r**2, increases strictly toward 1, and
    has no finite-ratio plateau in exact arithmetic. Inverse-ratio arithmetic
    avoids squaring a huge ratio or dividing by zero.

    The raw combined factor uses a C2 rational continuation below 0.6:

    .. math::

        x=K_{p,0}=1-K_2(1-K_1),\qquad
        K_p=F(x)=\begin{cases}
          x, & x\ge0.6,\\
          0.4+\dfrac{0.2}{1+t+t^2},\quad t=(0.6-x)/0.2, & x<0.6.
        \end{cases}

    At x=0.6 the value, slope and curvature match the identity: (0.6,1,0).
    Below the join, F'(x)=(1+2*t)/(1+t+t*t)**2 is strictly positive, and
    F(x)>=x. The continuation weakens the loss reduction gradually, without
    a finite-input plateau. On the valid raw domain [0,1], Kp lies in
    [0.4+0.2/13,1], approximately [0.415385,1], and F' lies in [7/169,1].
    The formal asymptote 0.4 would require x tending to -infinity; the bounded
    K1/K2 combination cannot reach it. Examples: F(0.5)=0.514286,
    F(0.4)=0.466667, F(0.2)=0.428571.

    The onset 0.6 and asymptote 0.4 are explicit regularization choices,
    not new aerodynamic calibration. Kacker and Okapuu (1982), equation (7),
    prescribe no such continuation. Aungier, Turbine Aerodynamics (2006),
    p. 75, equations (4-59)--(4-63), motivates limiting excessive profile
    reduction, but his modified Mach inputs are a different formulation.
    The bounded K1 and K2 already prevent negative Kp0. This continuation
    limits compressibility reduction of the reference profile contribution;
    it does not directly bound the complete loss or scale the additive shock.

    The KO secondary correction also uses this returned Kp, blending it with
    one through a smoothly bounded K3 influence. The positive continuation slope preserves
    raw-factor sensitivities; stationary points in Kp0 and zero derivatives
    from other loss-model terms can still occur.
    At M2=0, K1=Kp=1; K2=1 for M1>0 and is assigned 0 at (0,0).
    K2 alone has no unique limit at the origin. Kp has the unique limit 1
    and zero first derivatives there because 1-K1 is O(M2**2).
    Curvature at the joint origin can depend on the approach direction.

    Parameters
    ----------
    Ma_rel_in, Ma_rel_out : float or array_like
        Finite nonnegative Mach numbers in the blade-row frame. Inputs
        broadcast; valid complex-step perturbations are retained.

    Returns
    -------
    Kp, K2, K1 : list of float or array_like
        Combined profile factor, Mach-ratio factor and outlet-Mach factor.
        Negative or nonfinite Mach inputs produce NaN. Asymptotes can round
        to their limits for extreme arguments in finite precision.
    """
    m1, m2 = jnp.broadcast_arrays(jnp.asarray(Ma_rel_in), jnp.asarray(Ma_rel_out))
    a, b = jnp.real(m1), jnp.real(m2)
    valid = jnp.isfinite(m1) & jnp.isfinite(m2) & (a >= 0) & (b >= 0)

    # Restrict inactive operands as well as selecting the final branches.
    low_mach = jnp.where(valid & (b <= .15), m2, 0.)
    low = 1 - .005*(low_mach/.2)**2
    join = numerics.quintic_hermite(
        m2, .15, .25, .9971875, .9375, -.0375, -1.25, -.25, 0.)
    middle_mach = jnp.where(valid & (b >= .25) & (b <= .9), m2, .5)
    middle = 1-1.25*(middle_mach-.2)
    delta = jnp.where(valid & (b > .9), m2-.9, 0.)
    # Equivalent to .125/(1+t+t*t), without a large squared t.
    denom = .1+delta
    v, w = .1/denom, delta/denom
    tail = .125*v*v/(v*v+v*w+w*w)
    K1 = jnp.where(b <= .15, low, jnp.where(b < .25, join,
                   jnp.where(b <= .9, middle, tail)))

    squared_branch = valid & (b > 0) & (a <= .8*b)
    ratio = jnp.where(squared_branch, m1, 0.)/jnp.where(squared_branch, m2, 1.)
    tail_branch = valid & (a > 0) & ~squared_branch
    inverse = jnp.where(tail_branch, m2, 0.)/jnp.where(tail_branch, m1, 1.)
    inverse_squared = inverse**2
    u, v = 1-.64*inverse_squared, .36*inverse_squared
    saturated = .64+.36*u/jnp.sqrt(u*u+v*v)
    K2 = jnp.where(squared_branch, ratio**2, saturated)
    K2 = jnp.where((a == 0) & (b == 0), 0., K2)

    Kp = _continue_profile_compressibility(1-K2*(1-K1))
    return [jnp.where(valid, value, jnp.nan) for value in (Kp, K2, K1)]


# Hub/tip radius ratio below which the hub-Mach fit gradually flattens.
# The chart does not cover very small ratios; continuing its fitted rise
# without modification would give excessive hub-Mach amplification. This map
# keeps the original coordinate at and above 0.3, but approaches an effective
# ratio of 0.15 with zero slope and curvature as the physical ratio tends to 0.
# The 0.3 threshold is an extrapolation choice, not a chart calibration.
HUB_FLATTENING_RATIO = 0.3


def effective_hub_tip_ratio(hub_tip_ratio):
    """Return the C2 hub/tip coordinate used for low-ratio extrapolation.

    ``HUB_FLATTENING_RATIO`` is the physical radius ratio where the map
    leaves the identity branch. At zero physical ratio the effective ratio
    is half that threshold; its first two derivatives vanish there.
    """
    ratio = jnp.asarray(hub_tip_ratio)
    u = jnp.where(jnp.real(ratio) < HUB_FLATTENING_RATIO, ratio / HUB_FLATTENING_RATIO, 1.)
    continued = HUB_FLATTENING_RATIO * (.5 + u**3 - .5*u**4)
    return jnp.where(jnp.real(ratio) < HUB_FLATTENING_RATIO, continued, ratio)


# Figure 6 hub/mean inlet-Mach fits: (amplitude, end_ratio), stator first,
# rotor second. Away from the rounded join, f_hub = 1 + amplitude *
# (end_ratio - effective_hub_tip_ratio)^2 below end_ratio, and 1 above it.
# Amplitude controls the rise toward the hub; end_ratio locates the return to
# f_hub=1. These values were fitted to digitized chart data, not chosen as caps.
HUB_FIT = ((4.348048474772493, 0.8084516270970612),
           (5.94625398189089, 0.9468539887756984))
# Half-width in hub/tip ratio around each fitted end_ratio. Rounding over
# end_ratio +/- 0.01 removes the curvature kink where the quadratic meets
# f_hub=1, giving continuous value, first derivative and second derivative.
HUB_ROUNDING_WIDTH = 0.01


def get_hub_to_mean_mach_ratio(hub_tip_ratio, cascade_type_id: int):
    """C2 Figure 6 fit; 0 <= hub/tip <= 1, stator=0, rotor=1.

    The fit is unchanged above 0.3. Below 0.3 the effective-coordinate map
    flattens the correction toward a finite limit with zero slope and
    curvature. Ratios below the chart limit 0.5 remain extrapolation.
    Zero ratio is accepted here only to inspect the mathematical limit.

    ``HUB_FIT`` supplies each row's quadratic amplitude and end ratio.
    ``HUB_ROUNDING_WIDTH`` sets the half-width of the smooth join to unity;
    it is a radius-ratio interval, not a Mach interval.
    """
    ratio = jnp.asarray(hub_tip_ratio)
    row = jnp.asarray(cascade_type_id)
    amplitude = jnp.where(row == 0, HUB_FIT[0][0], HUB_FIT[1][0])
    end = jnp.where(row == 0, HUB_FIT[0][1], HUB_FIT[1][1])
    v = end - effective_hub_tip_ratio(ratio)
    # Evaluate the transition polynomial only on its bounded active argument.
    u = jnp.where((jnp.real(v) > -HUB_ROUNDING_WIDTH) & (jnp.real(v) < HUB_ROUNDING_WIDTH),
                  (v + HUB_ROUNDING_WIDTH) / (2 * HUB_ROUNDING_WIDTH), 0.)
    rounded = HUB_ROUNDING_WIDTH * (2 * u**3 - u**4)
    positive = jnp.where(jnp.real(v) <= -HUB_ROUNDING_WIDTH, 0.,
                         jnp.where(jnp.real(v) >= HUB_ROUNDING_WIDTH, v, rounded))
    value = 1 + amplitude * positive**2
    valid = (jnp.isfinite(ratio) & (jnp.real(ratio) >= 0) & (jnp.real(ratio) <= 1)
             & ((row == 0) | (row == 1)))
    return jnp.where(valid, value, jnp.nan)


# Hub inlet-Mach interval for turning on the KO shock power law. The center
# and half-width describe the interval; ONSET_START and ONSET_END are its
# explicit endpoints used below (center -/+ half-width, respectively).
ONSET_CENTER = 0.45
ONSET_HALF_WIDTH = 0.05
# Zero shock through hub Mach 0.4, the original KO onset threshold.
ONSET_START = 0.4
# Recover the full KO power law at hub Mach 0.5. The intervening smoothstep
# avoids an abrupt activation and matches the first two derivatives. The
# center and width are numerical smoothing choices, not extra fitted data.
ONSET_END = 0.5


def shock_onset(hub_mach):
    """Return (smoothstep weight, weighted KO power) for the hub Mach number.

    Both are exactly zero at and below 0.4. A quintic smoothstep centered
    at 0.45 with half-width 0.05 blends to the original power at 0.5.
    Values and first two derivatives match at both joins. The weighted
    power increases monotonically and needs no maximum or output floor.
    ``ONSET_START`` and ``ONSET_END`` apply to hub Mach f_hub*Ma_rel_in,
    not the mean inlet Mach. The center and half-width document that interval.
    """
    mach = jnp.asarray(hub_mach)
    active = jnp.real(mach) > ONSET_START
    in_transition = active & (jnp.real(mach) < ONSET_END)
    u = jnp.where(in_transition, (mach - ONSET_START) / (ONSET_END - ONSET_START), 0.)
    smoothstep = u**3 * (10. - 15.*u + 6.*u**2)
    weight = jnp.where(jnp.real(mach) >= ONSET_END, 1., jnp.where(active, smoothstep, 0.))
    # Array selection evaluates inactive branches too. A positive dummy argument prevents
    # fractional-power NaNs and singular inactive derivatives below onset.
    power_argument = jnp.where(active, mach - ONSET_START, 1.)
    strength = jnp.where(active, weight * power_argument**1.75, 0.)
    return weight, strength


# Exit-head-normalized shock coefficient below which the KO result is exact.
# Above 0.25, replace further unbounded growth from the pressure-head ratio
# with a smooth continuation. This acts before the outer 0.914*f_Re scaling.
SHOCK_CONTINUATION_START = 0.25
# Limiting shock coefficient, approached gradually rather than hard-clipped.
# A value of 0.5 is an engineering safeguard for extreme extrapolation; it
# is not fitted to KO data and does not cap the complete profile loss.
SHOCK_ASYMPTOTE = 0.5


def continue_shock_loss(raw_loss):
    """C2 upper continuation of the exit-normalized shock coefficient.

    Preserve raw values through 0.25; above that join, increase smoothly
    toward 0.5 without a finite hard plateau. With d=0.25 and
    t=(raw_loss-0.25)/d, the upper branch is 0.5-d/(1+t+t*t).
    Its slope is (1+2*t)/(1+t+t*t)**2: positive, at most one, and
    equal to one at the join, where curvature is zero. These are chosen
    safeguards, not parameters fitted to the KO paper. The reciprocal
    form below avoids squaring large raw values and is autodiff safe.
    ``SHOCK_CONTINUATION_START`` sets the join and ``SHOCK_ASYMPTOTE`` the
    limiting coefficient; their difference d sets the continuation scale.
    """
    raw = jnp.asarray(raw_loss)
    active = jnp.real(raw) > SHOCK_CONTINUATION_START
    delta = SHOCK_ASYMPTOTE - SHOCK_CONTINUATION_START
    excess = jnp.where(active, raw - SHOCK_CONTINUATION_START, 0.)
    inverse = delta / (delta + excess)
    continued = SHOCK_ASYMPTOTE - delta*inverse**2/(1-inverse+inverse**2)
    return jnp.where(active, continued, raw)


def get_shock_loss(flow_parameters, geometry, cascade_type_id: int):
    """Evaluate the complete KO inlet shock term and return its components.

    Parameters
    ----------
    flow_parameters : mapping
        ``Ma_rel_in`` is the nonnegative mean inlet Mach in the blade frame.
        ``p0_rel_in``, ``p_in``, ``p0_rel_out`` and ``p_out`` are the actual
        row-frame total/static pressures in consistent units. For a stator
        this frame is absolute; for a rotor it is relative to the blade.
    geometry : mapping
        ``hub_tip_ratio_in`` is the inlet hub/tip ratio, in (0, 1].
    cascade_type_id : int
        0 for stator, 1 for rotor. Numerical inputs may be broadcast arrays.

    Returns
    -------
    dict of arrays
        ``Y_shock``: exit-head-normalized shock coefficient, before the
        outer KO factor 0.914*f_Re. ``f_hub``, ``hub_tip_ratio`` and
        ``effective_hub_tip_ratio`` describe the hub correction. ``Ma_hub``
        is f_hub*Ma_rel_in. ``onset_weight`` is the quintic smoothstep;
        ``onset_strength`` is its weighted (Ma_hub-0.4)**1.75 power.
        ``Y_shock_hub`` is 0.75*onset_strength; ``Y_shock_mean_inlet``
        includes the physical hub/tip span weight. ``inlet_head``,
        ``exit_head`` and ``head_ratio`` expose the pressure conversion.
        ``Y_shock_raw`` is the unbounded exit-normalized coefficient.
        ``head_ratio_effective`` reconstructs Y_shock from the span-weighted
        inlet loss after the upper continuation; it equals head_ratio when
        the raw loss is at most 0.25 (also when the inlet shock is zero).
        All entries have the broadcast output shape. Example integration:
        ``Y_shock = get_shock_loss(flow, geometry, row)["Y_shock"]``.

    Notes
    -----
    The hub fit is C2, with flattening below hub/tip 0.3. Shock onset is C2:
    zero through hub Mach 0.4, smoothly activated over 0.4-0.5, and exactly
    the KO onset power law above 0.5. After pressure-head conversion, raw
    losses through 0.25 are unchanged; a C2 rational continuation bounds
    larger values by an asymptote of 0.5, before the outer 0.914*f_Re.
    No hard maximum or minimum operator is applied. The pressure heads
    themselves are unchanged; the effective ratio is a diagnostic factor.
    Every loss/component plot can use this single evaluation. Real-part
    branch predicates preserve complex-step derivatives on the NumPy backend.

    Finite states must have positive pressures, inlet head >= 0 and exit
    head > 0. Invalid states return NaN for Y_shock; the component outputs
    remain available for diagnosis. Call validate_shock_inputs at the input boundary
    for descriptive errors. No pressure epsilon, EOS or Reynolds correction
    is imposed. The caller supplies thermodynamically consistent states.
    The fitted constants are stored in this module; no runtime fitting or
    study-folder dependencies are required.
    """
    ratio = jnp.asarray(geometry["hub_tip_ratio_in"])
    m1 = jnp.asarray(flow_parameters["Ma_rel_in"])
    factor = get_hub_to_mean_mach_ratio(ratio, cascade_type_id)
    p01, p1, p02, p2 = map(jnp.asarray, (flow_parameters["p0_rel_in"],
        flow_parameters["p_in"], flow_parameters["p0_rel_out"], flow_parameters["p_out"]))
    q1, q2 = p01 - p1, p02 - p2
    mach_hub = factor * m1
    weight, strength = shock_onset(mach_hub)
    loss_hub = .75 * strength
    loss_mean = loss_hub * ratio
    head_ratio = q1 / jnp.where(jnp.real(q2) > 0, q2, jnp.nan)
    raw = loss_mean * head_ratio
    value = continue_shock_loss(raw)
    effective_head_ratio = jnp.where(jnp.real(raw) > SHOCK_CONTINUATION_START,
        value / jnp.where(jnp.real(loss_mean) > 0, loss_mean, 1.), head_ratio)
    pressure_valid = (jnp.isfinite(p01) & jnp.isfinite(p1) & jnp.isfinite(p02)
                      & jnp.isfinite(p2) & (jnp.real(p01) > 0) & (jnp.real(p1) > 0) & (jnp.real(p02) > 0) & (jnp.real(p2) > 0))
    valid = (pressure_valid & jnp.isfinite(m1) & (jnp.real(m1) >= 0) & (jnp.real(ratio) > 0)
             & (jnp.real(ratio) <= 1) & jnp.isfinite(factor) & (jnp.real(q1) >= 0) & (jnp.real(q2) > 0))
    result = dict(Y_shock=jnp.where(valid, value, jnp.nan), f_hub=factor,
        hub_tip_ratio=ratio, effective_hub_tip_ratio=effective_hub_tip_ratio(ratio),
        Ma_hub=mach_hub, onset_weight=weight, onset_strength=strength,
        Y_shock_hub=loss_hub, Y_shock_mean_inlet=loss_mean,
        inlet_head=q1, exit_head=q2, head_ratio=head_ratio,
        Y_shock_raw=raw, head_ratio_effective=effective_head_ratio)
    arrays = jnp.broadcast_arrays(*result.values())
    return dict(zip(result, arrays))


def validate_shock_inputs(flow_parameters, geometry, cascade_type_id: int):
    """Optional input-boundary validation for descriptive domain errors.

    Does not impose an ideal-gas equation of state. The caller must supply
    thermodynamically consistent states. The numerical helper itself returns NaN for invalid states.
    """
    checks = {
        "cascade_type_id must be 0 (stator) or 1 (rotor)":
            jnp.isin(jnp.asarray(cascade_type_id), (0, 1)),
        "Ma_rel_in must be finite and nonnegative":
            jnp.isfinite(flow_parameters["Ma_rel_in"]) & (jnp.asarray(flow_parameters["Ma_rel_in"]) >= 0),
        "hub_tip_ratio_in must be finite and in (0, 1]":
            jnp.isfinite(geometry["hub_tip_ratio_in"]) & (jnp.asarray(geometry["hub_tip_ratio_in"]) > 0)
            & (jnp.asarray(geometry["hub_tip_ratio_in"]) <= 1),
    }
    for key in ("p0_rel_in", "p_in", "p0_rel_out", "p_out"):
        value = jnp.asarray(flow_parameters[key])
        checks[f"{key} must be finite and positive"] = jnp.isfinite(value) & (value > 0)
    checks["inlet pressure head p0_rel_in - p_in must be nonnegative"] = (
        jnp.asarray(flow_parameters["p0_rel_in"]) >= jnp.asarray(flow_parameters["p_in"]))
    checks["exit pressure head p0_rel_out - p_out must be positive"] = (
        jnp.asarray(flow_parameters["p0_rel_out"]) > jnp.asarray(flow_parameters["p_out"]))
    for message, valid in checks.items():
        if not jnp.all(valid):
            raise ValueError(message)
