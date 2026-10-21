"""Kacker--Okapuu losses with documented numerical continuations.

Lengths use consistent units, chord is true chord and height is full blade
height. Angles are degrees from the meridional direction. NumPy evaluates
these correlations; the backend seam does not imply a JIT guarantee.

A shared positive hub/tip coordinate continues from 0.5 toward 0.3 so that
negative meanline hub radii remain evaluable. Shock loss continues above
0.25 toward 0.7 before Reynolds scaling, avoiding excessive pressure-head
amplification. Effective Reynolds number continues above 1e7 toward 5e7,
preventing profile and shock losses from vanishing at unrealistically large
Reynolds numbers. Physical-turning loading retains the tangential velocity
change and continues only its mean-secant factor toward sec(87.5 degrees).
These are modeling choices, not extensions of the empirical calibration.

Equations and transition constants are documented at their functions and in
loss_model_analysis/loss_model_audit/. Constants remain at module scope so
analysis scripts can inspect the settings used by the production models.
"""

from ._backend import jnp
from . import _trailing_edge as trailing_edge
from . import _blade_loading as blade_loading
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
    "metal_angle_out",
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
    r"""Return the Kacker--Okapuu profile pressure-loss coefficient.

    .. math::

        Y_p=0.914f_\mathrm{Re}\left[\frac23 B(z;R,I)F_t(z,\tau)K_p
                                   +Y_\mathrm{shock}\right],
        \qquad \tau=t_\mathrm{max}/c.

    R and I are the fitted AM nozzle/impulse references at pitch/true chord
    and smooth_abs(arctan(T(beta_eff))). T is the common angle continuation:
    the original tangent through +/-80 degrees, with rational C2 tails
    toward +/-tan87.5. The reference angle therefore has no lower floor,
    remains exact to rounding below 80 degrees, and stays bounded above.

    Below 10 degrees in the design outlet direction, beta_eff uses
    get_effective_outlet_angle: a rational C2 continuation of the operating
    trend that keeps that direction through zero trial outlet angle. For
    b=d*beta, a=10deg,t=(a-b)/a, beta_eff=d*a/(1+t+t^2) below b=a.
    The coordinate z=-theta/beta_eff therefore does not reverse sign or
    reset to a nozzle at zero. The sign of geometry metal_angle_out sets
    d; zero or an absent metal angle selects +1 for standalone calls.
    The direction is fixed geometry information,
    never recomputed from the trial flow angle.
    B retains the AM quadratic on 0<=z<=1, approaches 0.8R for negative
    z, and continues beyond one within 20% of I for either R/I ordering.
    See get_profile_loss_blend for its equations and C1 join at zero.

    The thickness factor retains its C2 onset over 0<z<0.05. Its exponent
    is z through one, then approaches two, so large z cannot create an
    unbounded thickness power or drive a positive-thickness factor to zero.
    The Reynolds factor uses Re_eff=Re on [1e4,1e7], approaching 1000 for
    large negative trial Re and 5e7 for large positive Re. The high-Re
    factor tends to 50**(-.2), so it cannot eliminate the complete profile
    or its shock contribution. Negative Re is a numerical solver
    continuation, not a physical flow state.

    Inlet shock uses a shared positive hub/tip coordinate in its hub-Mach
    fit and span weight: identity from 0.5 upward, continued toward 0.3
    below 0.5 with C2 matching. This permits signed hub radii without
    changing geometry and prevents a vanishing span multiplier from
    eliminating an active shock. The exit-head-normalized shock is exact
    through 0.25, then approaches 0.7 smoothly to bound amplification at
    very small exit pressure heads. These are explicit modeling choices,
    not fitted chart parameters. The Reynolds correction multiplies both
    reference and shock terms; its nonzero high-Re limit keeps both
    contributions present during extreme size extrapolation.

    K1,K2,Kp use their documented smooth continuations. The optional
    1+60*(M2-1)^2 supersonic multiplier is disabled. No separate cap is
    imposed on the complete Yp.

    Inputs
    ------
    flow_parameters supplies finite signed beta_out (degrees), Re_out,
    nonnegative Ma_rel_in/Ma_rel_out and the positive pressure-head inputs
    required by get_shock_loss. Geometry supplies positive true chord and
    maximum thickness, nonnegative pitch, signed inlet metal angle
    leading_edge_angle and finite raw hub_tip_ratio_in <= 1 (zero and
    negative ratios use the continued positive shock coordinate). Lengths
    share units. Inputs broadcast with the current NumPy backend and
    valid complex-step perturbations are retained. Invalid geometry,
    Machs or pressure states are not silently made physically admissible.
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
    # The optional 1+60*(M2-1)^2 supersonic multiplier is deliberately
    # disabled for now; keep f_Ma=1 rather than activating that extra growth.
    Y_shock = get_shock_loss(flow_parameters, geometry, cascade_type_id)["Y_shock"]
    Kp, K2, K1 = get_compressible_correction_factors(Ma_rel_in, Ma_rel_out)
    metal_angle_out = geometry.get("metal_angle_out", 1.)
    effective_outlet = get_effective_outlet_angle(beta_out, metal_angle_out)
    angle_out_bis = math.smooth_abs(get_blade_angle(effective_outlet))
    Yp_reaction = nozzle_blades(s / c, angle_out_bis)
    Yp_impulse = impulse_blades(s / c, angle_out_bis)
    z = get_cascade_angle_ratio(theta_in, beta_out, metal_angle_out)
    Y_p = get_profile_loss_blend(z, Yp_reaction, Yp_impulse)
    thickness_factor = get_profile_thickness_factor(z, t_max / c)
    Y_p = Y_p * thickness_factor
    Y_p = 0.914 * (2 / 3 * Y_p * Kp + Y_shock)
    f_Ma = 1.0
    Y_p = f_Re * f_Ma * Y_p
    return Y_p


# Common numerical angle coordinates, also used by clearance and secondary.
get_blade_tangent = blade_loading.tangent
get_blade_cosine = blade_loading.cosine
get_blade_angle = blade_loading.effective_angle
get_blade_loading = blade_loading.loading
get_blade_mean_secant = blade_loading.mean_secant
get_physical_cosine_ratio = blade_loading.physical_cosine_ratio
get_effective_outlet_angle = blade_loading.effective_outlet_angle
get_cascade_angle_ratio = blade_loading.angle_ratio

PROFILE_UPPER_EXTRAPOLATION_ALLOWANCE = 0.2
PROFILE_THICKNESS_EXPONENT_LIMIT = 2.


def get_profile_thickness_exponent(z):
    r"""Preserve positive z through one; continue its exponent smoothly to two.

    E=z for 0<=z<=1; E=1+S(z-1) for z>1, S(t)=t/sqrt(1+t^2).
    Negative z uses E=0. The onset of the full thickness factor still makes
    its z=0 join C2. The upper join is C2, removes exponential growth in z,
    and keeps the original interpolation interval unchanged. Limit two is a
    numerical extrapolation choice, not a new AM calibration.
    """
    z = jnp.asarray(z)
    outer = jnp.real(z) > 1
    width = PROFILE_THICKNESS_EXPONENT_LIMIT-1
    t = jnp.where(outer, (z-1)/width, 0.)
    small = jnp.where(jnp.real(t) <= 1, t, 1.)
    inv = 1/jnp.where(jnp.real(t) > 1, t, 1.)
    sat = jnp.where(jnp.real(t) <= 1, small/jnp.sqrt(1+small*small),
                    1/jnp.sqrt(1+inv*inv))
    return jnp.where(outer, 1+width*sat, jnp.where(jnp.real(z) > 0, z, 0.))


# Width in the dimensionless coordinate z=-theta_in/beta_out. Turn on the
# thickness correction over 0 < z < 0.05 (center 0.025), matching value, slope
# and curvature to avoid the derivative jump of an abrupt exponent switch.
# The upper exponent continuation separately controls large positive z.
PROFILE_THICKNESS_TRANSITION = 0.05


def get_profile_thickness_factor(z, thickness_chord):
    r"""Return the positive thickness correction with bounded exponent.

    With tau=t_max/c>0, delta=0.05, q=z/delta,
    S5(q)=10q^3-15q^4+6q^5 and exponent E=get_profile_thickness_exponent(z),

    .. math::

        F_t=\begin{cases}
          1,&z\le0,\\
          1+S_5(z/\delta)[(\tau/0.2)^{E(z)}-1],&0<z<\delta,\\
          (\tau/0.2)^{E(z)},&z\ge\delta.
        \end{cases}

    E=z on0<=z<=1, then approaches2 with a C2 unit-saturation tail.
    The original AM power is therefore exact on0.05<=z<=1; extrapolation
    beyond impulse cannot grow exponentially in unbounded z. It remains
    bounded between the positive endpoint powers tau/0.2 and(tau/0.2)^2
    above z=1. The factor equals one identically at tau=0.2. Existing C2
    matching at zero and delta is retained; no curvature smoothing of the
    reference blend at z=0 is required. The exponent limit is an explicit
    extrapolation choice. Broadcasting and complex-step derivatives work.
    """
    z, thickness_chord = jnp.broadcast_arrays(jnp.asarray(z), jnp.asarray(thickness_chord))
    real_z = jnp.real(z)
    width = PROFILE_THICKNESS_TRANSITION
    power = (thickness_chord / 0.2) ** get_profile_thickness_exponent(z)
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
# they do not modify the interpolation over 0 <= z <= 1.
PROFILE_EXTRAPOLATION_ZERO_TURNING = 0.9


def get_profile_loss_blend(z, reaction, impulse):
    r"""Blend positive AM references with bounded extrapolation in both directions.

    For z<0 retain R*[0.8+0.2/(1+z^2)], hence B(-1)=0.9R and limit
    0.8R. On 0<=z<=1 use exactly B=(1-z^2)R+z^2I. Above one use

    .. math::

        B=I+0.2I\,S\!\left(\frac{(I-R)(z^2-1)}{0.2I}\right),
        \qquad S(t)=t/\sqrt{1+t^2}.

    S is odd: either reference ordering is admissible. For I>R the tail
    approaches 1.2I; for I<R it approaches 0.8I; for I=R it stays I.
    Value, slope and curvature match the original quadratic at z=1.
    The z=0 join is C1, with an accepted curvature jump. The 20% allowance
    is an explicit extrapolation choice, not additional empirical data.

    All finite z and strictly positive finite references broadcast. Complex
    perturbations are preserved. Reciprocal arithmetic avoids constructing
    large z^2 or a singular ratio at I=R. No ordering exception is raised.
    """
    dtype = complex if any(jnp.iscomplexobj(v) for v in (z, reaction, impulse)) else float
    z, reaction, impulse = jnp.broadcast_arrays(
        jnp.asarray(z, dtype=dtype), jnp.asarray(reaction, dtype=dtype),
        jnp.asarray(impulse, dtype=dtype))
    real_z = jnp.real(z)
    if not (jnp.isfinite(z).all() and jnp.isfinite(reaction).all() and jnp.isfinite(impulse).all()):
        raise ValueError("z and profile reference losses must be finite.")
    if jnp.any(jnp.real(reaction) <= 0) or jnp.any(jnp.real(impulse) <= 0):
        raise ValueError("Profile reference losses must be strictly positive.")
    f, q = PROFILE_EXTRAPOLATION_ASYMPTOTE, PROFILE_EXTRAPOLATION_ZERO_TURNING
    length = jnp.sqrt((q-f)/(1-q))
    negative = jnp.real(z) < 0
    negative_z = jnp.where(negative, z/length, 0.)
    local = jnp.abs(jnp.real(negative_z)) <= 1
    small = jnp.where(local, negative_z, 0.)
    inv = 1/jnp.where(local, 1., negative_z)
    inverse_squared = jnp.where(local, 1/(1+small*small), inv*inv/(1+inv*inv))
    low = reaction*(f+(1-f)*inverse_squared)
    middle_z = jnp.where((real_z >= 0) & (real_z <= 1), z, 0.)
    interior = (1-middle_z**2)*reaction+middle_z**2*impulse
    inverse_z = 1/jnp.where(real_z > 1, z, 1.)
    inv_square = inverse_z*inverse_z
    numerator = (impulse-reaction)*(1-inv_square)
    denominator = PROFILE_UPPER_EXTRAPOLATION_ALLOWANCE*impulse*inv_square
    scale = jnp.maximum(jnp.abs(jnp.real(numerator)), jnp.abs(jnp.real(denominator)))
    safe_scale = jnp.where(scale > 0, scale, 1.)
    u, v = numerator/safe_scale, denominator/safe_scale
    norm = jnp.sqrt(u*u+v*v)
    sat = u/jnp.where(scale > 0, norm, 1.)
    upper = impulse*(1+PROFILE_UPPER_EXTRAPOLATION_ALLOWANCE*sat)
    return jnp.where(negative, low, jnp.where(real_z > 1, upper, interior))


# Multiplicative half-width around each Reynolds breakpoint (200,000 and
# 1,000,000): smooth over [Re_break/1.5, Re_break*1.5], using log(1.5) as the
# width in log-Re coordinates. This removes derivative kinks while preserving
# the empirical power laws outside the bands. It is not a multiplier on Re.
REYNOLDS_TRANSITION_FACTOR = 1.5


REYNOLDS_CONTINUATION_START = 1e4
REYNOLDS_ASYMPTOTE = 1e3
# High-Re extrapolation: preserve the correlation through ten million, then
# approach its value at fifty million. These are explicit modeling choices.
REYNOLDS_HIGH_CONTINUATION_START = 1e7
REYNOLDS_HIGH_ASYMPTOTE = 5e7


def get_effective_reynolds_number(reynolds):
    r"""C2 positive Reynolds coordinate with finite low and high limits.

    Re_eff=Re on [1e4,1e7]. Below a=10000 use
    r0+D/(1+t+t^2), r0=1000, D=a-r0, t=(a-Re)/D.
    The join matches value, slope and curvature of the identity; as trial
    Re tends to minus infinity the coordinate tends to 1000 without a hard
    plateau. Negative Re is a solver continuation, not a physical state.
    Above 1e7 use 5e7-D/(1+t+t^2), D=4e7, t=(Re-1e7)/D.
    It also matches the identity through curvature, and tends to 5e7 as
    Re tends to infinity. The coordinate remains exactly Re on [1e4,1e7].
    The correction factor therefore lies between 50**(-.2)=0.457305052
    and 200**.4=8.325532074. The high limit prevents unlimited Reynolds
    reduction of the complete profile, including its additive shock term.
    """
    re = jnp.asarray(reynolds)
    valid = jnp.isfinite(re)
    active = valid & (jnp.real(re) < REYNOLDS_CONTINUATION_START)
    d = REYNOLDS_CONTINUATION_START-REYNOLDS_ASYMPTOTE
    excess = jnp.where(active, REYNOLDS_CONTINUATION_START-re, 0.)
    inverse = d/(d+excess)
    low = REYNOLDS_ASYMPTOTE+d*inverse**2/(1-inverse+inverse**2)
    high_active = valid & (jnp.real(re) > REYNOLDS_HIGH_CONTINUATION_START)
    high_delta = REYNOLDS_HIGH_ASYMPTOTE-REYNOLDS_HIGH_CONTINUATION_START
    high_excess = jnp.where(high_active, re-REYNOLDS_HIGH_CONTINUATION_START, 0.)
    high_inverse = high_delta/(high_delta+high_excess)
    high = REYNOLDS_HIGH_ASYMPTOTE-high_delta*high_inverse**2/(1-high_inverse+high_inverse**2)
    return jnp.where(valid, jnp.where(active, low, jnp.where(high_active, high, re)), jnp.nan)


def get_reynolds_correction_factor(reynolds):
    r"""Positive C2 Reynolds correction with nonzero high-Re asymptote.

    The profile assembly multiplies both its reference and shock terms by
    this factor. A finite high-Re limit prevents extreme meanline sizing
    from eliminating either loss contribution as chord grows. The limit
    is the empirical high-Re power-law value at fifty million, 0.457305052;
    the continuation starts at ten million to preserve the correlation
    and its two blending bands over the ordinary design range. Low and
    negative trial values also receive a finite positive coordinate so
    nonlinear solver iterates remain evaluable without a hard clamp.

    First compute r=get_effective_reynolds_number(Re): r=Re on [1e4,1e7];
    below 1e4 r approaches 1000 as trial Re tends to minus infinity, and
    above 1e7 r approaches 5e7 as Re tends to infinity.
    Then with P_w the compact C2 positive-part function and w=log(1.5),

    .. math::

        f_\mathrm{Re}=\exp\left[
          0.4P_w(-\ln(r/(2\,10^5)))
          -0.2P_w(\ln(r/10^6))\right].

    The original empirical powers and middle unit plateau are unchanged
    for 10000<=Re<=1e7, including the factor 1.5 transition bands
    around 200000 and 1000000. The low-Re coordinate is C2 at 10000
    and gives finite nonzero sensitivity for finite negative trial inputs.
    It limits f_Re above by 200**0.4=8.325532074 and below by the
    power-law value at 5e7: 50**(-.2)=0.457305052. Both outer coordinate
    joins are C2. These limits are numerical modeling choices; the low
    continuation is not a physical negative-Re correlation.
    Finite scalar/array inputs broadcast; nonfinite Re yields
    NaN. Real-part branch choices retain complex-step differentiation.
    """
    reynolds = get_effective_reynolds_number(reynolds)
    half_width = jnp.log(REYNOLDS_TRANSITION_FACTOR)
    low = math.positive_part(-jnp.log(reynolds / 200000.0), half_width)
    high = math.positive_part(jnp.log(reynolds / 1000000.0), half_width)
    return jnp.exp(0.4 * low - 0.2 * high)


def get_secondary_loss(flow_parameters, geometry, cascade_type_id: int):
    r"""Return KO secondary loss preserving physical turning and projection.

    .. math::

        Y_s=1.2(0.0334)f_\mathrm{AR}(H/c)\,K_s\,
            Z(\beta_\mathrm{in},\beta_\mathrm{out})
            \frac{\cos(\beta_\mathrm{out})}{\cos(\theta_\mathrm{in})}.

    Z=get_blade_loading preserves the physical sine/cosine turning factor
    and continues only the mean secant: identity through sec80deg, C2 tail
    toward sec87.5deg. No individual tangent is saturated. The projection
    ratio is physical too: at zero incidence and equal meridional speed it
    equals w_in/w_out. These choices prevent artificial loss collapse when
    both rotor relative angles approach -90deg at high reaction. Original
    KO angular algebra is preserved whenever the mean angle is within
    +/-80deg. There is no separate loading or final secondary-loss cap.

    The aspect factor retains its existing 1.5--2.5 Hermite join and
    low-AR cap, now cancelled algebraically before division by H. Ks uses
    get_secondary_compressibility_factor_from_geometry, also without H
    division. Thus zero height with positive meridional chord is a finite
    mathematical endpoint: f_AR=1, Ks=Kp. The b=H=0 corner is undefined.
    See both helpers and _blade_loading for the implemented equations.

    Lengths must be finite, H>=0,c>0,b>=0, with b or H positive; invalid
    inputs return NaN. Flow/metal angles must be strictly within (-90,+90)
    signed degrees; exactly grazing and reversed flow return NaN rather
    than a periodic extrapolation. Machs are nonnegative. Negative heights
    are not continued. Equal flow angles
    still give zero loss, and sign reflection preserves the result.
    The shared numerical settings are safeguards, not new KO data. There
    is no separate final Ys cap. New Moustapha inherits this component;
    Benner retains its own secondary model. Row identifier is unused.
    """
    m2, m1 = flow_parameters["Ma_rel_out"], flow_parameters["Ma_rel_in"]
    beta_out, beta_in = flow_parameters["beta_out"], flow_parameters["beta_in"]
    b, H, c = map(jnp.asarray, (geometry["meridional_chord"], geometry["height"], geometry["chord"]))
    theta = geometry["leading_edge_angle"]
    valid = (jnp.isfinite(b) & jnp.isfinite(H) & jnp.isfinite(c)
             & (jnp.real(b) >= 0) & (jnp.real(H) >= 0) & (jnp.real(c) > 0)
             & ((jnp.real(b) > 0) | (jnp.real(H) > 0)))
    safe_c = jnp.where(valid, c, 1.)
    Kp = get_compressible_correction_factors(m1, m2)[0]
    Ks = get_secondary_compressibility_factor_from_geometry(Kp, b, H)
    Z = get_blade_loading(beta_in, beta_out)
    far = get_secondary_aspect_ratio_factor(jnp.where(valid, H, 0.)/safe_c)
    cosine_ratio = get_physical_cosine_ratio(beta_out, theta)
    value = 1.2*Ks*0.0334*far*Z*cosine_ratio
    return jnp.where(valid, value, jnp.nan)


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
    r"""Return Ks for an already formed nonnegative meridional-chord/height ratio.

    This compatibility interface delegates to the cancelled geometry
    implementation with b=ratio,H=1, so a large finite ratio is never
    squared directly. Physical callers at zero height should instead use
    get_secondary_compressibility_factor_from_geometry(Kp,b,H).
    The implemented law remains Ks=(1-C((b/H)^2))+C((b/H)^2)*Kp,
    where C is the common identity-through0.8, asymptote-one C2 tail.
    For0<=Kp<=1 the factor stays betweenKp andone. No independent Ks
    floor is applied; see the geometry helper for valid zero endpoints.
    """
    ratio = jnp.asarray(meridional_chord_height)
    # Finite ordinary callers retain the old public ratio interface.
    return get_secondary_compressibility_factor_from_geometry(profile_factor, ratio, 1.)


def get_secondary_compressibility_factor_from_geometry(profile_factor, chord_axial, height):
    r"""Ks with a cancelled K3=(b/H)^2 continuation, valid at zero height.

    For b/H<=sqrt(.8) retain K3=(b/H)^2. Otherwise let v=H/b,
    u=0.2*v^2/(1-0.6*v^2); K3_eff=1-0.2*u^2/(1-u+u^2).
    This is algebraically the existing C(K3), without first dividing by H
    or squaring a huge ratio. Return Ks=(1-K3_eff)+K3_eff*Kp.
    At H=0,b>0 the value is exactly Kp, with finite zero height slope.
    At b=0,H>0 Ks=1. The joint b=H=0 limit depends on the path and remains
    invalid. Nonnegative finite lengths with at least one positive are
    required; valid inputs broadcast and retain complex-step derivatives.
    """
    kp, b, H = jnp.broadcast_arrays(jnp.asarray(profile_factor),
                                    jnp.asarray(chord_axial), jnp.asarray(height))
    valid = (jnp.isfinite(kp) & jnp.isfinite(b) & jnp.isfinite(H)
             & (jnp.real(b) >= 0) & (jnp.real(H) >= 0)
             & ((jnp.real(b) > 0) | (jnp.real(H) > 0)))
    small = valid & (jnp.real(H) > 0) & (jnp.real(b) <= jnp.sqrt(.8)*jnp.real(H))
    ratio = jnp.where(small, b, 0.)/jnp.where(small, H, 1.)
    inverse = jnp.where(valid & ~small, H, 0.)/jnp.where(valid & ~small, b, 1.)
    square = inverse*inverse
    u = .2*square/(1-.6*square)
    tail = 1-.2*u*u/(1-u+u*u)
    influence = jnp.where(small, ratio*ratio, tail)
    return jnp.where(valid, (1-influence)+influence*kp, jnp.nan)


def get_secondary_aspect_ratio_factor(aspect_ratio):
    r"""Return the capped KO aspect factor with an explicitly finite zero limit.

    For A=H/c, the raw low branch is N(A)/A,
    N=1-.25*sqrt(2-A). Retain the existing quintic Hermite join over
    1.5<A<2.5 to the upper branch1/A, then apply the common C2 factor
    continuation C: identity through0.8 and asymptoteone above it.

    On the capped low branch the implemented expression first cancels A:

    .. math::

        u=\frac{0.2A}{N(A)-0.6A},\qquad
        f_\mathrm{AR}=1-\frac{0.2u^2}{1-u+u^2}.

    This is algebraically C(N/A) at positive A, so every positive-AR
    prediction is preserved. At A=0 it evaluates directly toone with
    zero first derivative and finite curvature, without division by zero.
    The cancelled branch applies where N>0.8A; otherwise use N/A at a
    safely positive A. H=0 is a mathematical continuation endpoint, not a
    valid physical zero-height turbine. Negative/nonfinite A returnsNaN.
    Inputs broadcast and retain complex-step derivatives. The factor's
    unit limit and Hermite join are numerical modeling choices.
    """
    A = jnp.asarray(aspect_ratio)
    valid = jnp.isfinite(A) & (jnp.real(A) >= 0)
    a, b = 1.5, 2.5
    q = 2-a
    n = 1-.25*jnp.sqrt(q)
    dn, d2n = .125/jnp.sqrt(q), .0625/q**1.5
    y0, dy0 = n/a, dn/a-n/a**2
    d2y0 = d2n/a-2*dn/a**2+2*n/a**3
    patch = math.quintic_hermite(A, a, b, y0, 1/b, dy0, -1/b**2, d2y0, 2/b**3)
    low_ar = jnp.where(valid & (jnp.real(A) <= a), A, a)
    numerator = 1-.25*jnp.sqrt(2-low_ar)
    capped = jnp.real(numerator) > .8*jnp.real(low_ar)
    # u=.2/(.2+N/A-.8), cancelled before A=0 is evaluated.
    denominator = numerator-.6*low_ar
    u = .2*low_ar/jnp.where(capped, denominator, 1.)
    low_capped = 1-.2*u*u/(1-u+u*u)
    low_raw = numerator/jnp.where(capped, 1., low_ar)
    low = jnp.where(capped, low_capped, low_raw)
    high = 1/jnp.where(valid & (jnp.real(A) >= b), A, b)
    value = jnp.where(jnp.real(A) <= a, low, jnp.where(jnp.real(A) >= b, high, patch))
    return jnp.where(valid, value, jnp.nan)


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
        beta_out: finite signed exit flow angle in degrees;
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

    The Mach-dependent paper conversion is used at every Mach.
    Kacker--Okapuu, Moustapha and Benner share this production helper.

    Below 10 degrees in the fixed design outlet direction the same rational
    effective-angle coordinate as profile loss continues its existing trend.
    The z coordinate retains its sign through beta_out=0; no forced nozzle
    reset is applied. The sign of geometry metal_angle_out defines the
    direction; zero or an absent metal angle selects +1. This changes
    the low-angle extrapolation, not the ordinary beta>=10deg correlation.
    For prescribed finite-z paths use
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

    For ``|z|<0.05``, replace z*abs(z) by the odd C2 polynomial
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
    Invalid thickness/opening/angles return NaN. The common axial-angle
    continuation permits beta_out=0 without a sign switch; prescribed finite-z paths
    can instead be evaluated by
    trailing_edge.energy_loss(x, z). Inactive divisions are protected.
    """
    thickness = jnp.asarray(geometry["trailing_edge_thickness"])
    opening = jnp.asarray(geometry["opening"])
    theta = jnp.asarray(geometry["leading_edge_angle"])
    beta = jnp.asarray(flow_parameters["beta_out"])
    valid = (jnp.isfinite(thickness) & jnp.isfinite(opening)
             & jnp.isfinite(theta) & jnp.isfinite(beta)
             & (jnp.real(thickness) >= 0) & (jnp.real(opening) > 0)
)
    x = jnp.where(valid, thickness, 0.)/jnp.where(valid, opening, 1.)
    z = get_cascade_angle_ratio(theta, beta, geometry.get("metal_angle_out", 1.))
    value = trailing_edge.energy_loss(x, z)
    return jnp.where(valid, value, jnp.nan)


# KO equation (20): effective seal gap / true chord, with the published exponent.
# Below 0.1% chord, a C2 endpoint patch removes the fractional-power singularity.
TIP_CLEARANCE_EXPONENT = 0.78
TIP_CLEARANCE_TRANSITION = 1e-3
# Numerical continuations, not parameters fitted to KO measurements.
TIP_CLEARANCE_ANGLE_START = 80.0
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
    """Compatibility alias for the common bounded C2 tangent coordinate."""
    return get_blade_tangent(angle)


def get_tip_clearance_loading(beta_in, beta_out):
    """Physical-turning loading with only a C2 mean-secant continuation.

    See _blade_loading.loading for the equations and forward-flow domain.
    No individual tangent is saturated: the mean secant is identity up
    to sec80deg, then approaches sec87.5deg. Equal angles give zero loading.
    There is no separate cap on the turning factor or on final loading.
    """
    return get_blade_loading(beta_in, beta_out)


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
    :func:`get_tip_clearance_loading`: physical turning times a continued
    mean secant, identity through sec80deg and asymptotic to sec87.5deg.
    Individual flow angles are not limited. This preserves finite turning
    as both rotor relative angles approach -90deg at high reaction,
    preventing an artificial collapse of clearance loss. There is no
    separate cap on turning or final loading. The secant thresholds are
    numerical extrapolation choices, not an empirical clearance refit.
    See those helpers for the implemented equations and their rationale.

    Parameters
    ----------
    flow_parameters : dict
        Signed row-frame beta_in and beta_out strictly within (-90,+90)
        degrees from the
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
    join = math.quintic_hermite(
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


# Preserve the Figure 6 chart coordinate at and above 0.5. Below that join,
# use one positive coordinate for BOTH the hub-Mach fit and the span weight.
# The lower asymptote is a modeling choice, not a fitted chart parameter.
HUB_CONTINUATION_START = 0.5
HUB_RATIO_ASYMPTOTE = 0.3


def effective_hub_tip_ratio(hub_tip_ratio):
    r"""Return the positive loss coordinate for a signed raw hub/tip ratio.

    For r>=0.5 return r. Below 0.5 use
    0.3+0.2/(1+t+t^2), t=(0.5-r)/0.2. It matches identity value, slope
    and curvature at 0.5 and approaches 0.3 as r tends to minus infinity.
    The 0.5 join preserves the Figure 6 chart interval; the 0.3 asymptote
    bounds hub-Mach amplification and keeps the span-weighted shock
    positive for an active onset. Both shock factors use this coordinate.
    It changes only empirical loss inputs; signed radii remain in geometry.
    These limits are modeling choices, not additional chart calibration.
    Real-part branch decisions preserve complex-step differentiation.
    """
    ratio = jnp.asarray(hub_tip_ratio)
    delta = HUB_CONTINUATION_START - HUB_RATIO_ASYMPTOTE
    excess = jnp.where(jnp.real(ratio) < HUB_CONTINUATION_START,
                       HUB_CONTINUATION_START-ratio, 0.)
    inverse = delta/(delta+excess)
    continued = HUB_RATIO_ASYMPTOTE + delta*inverse**2/(1-inverse+inverse**2)
    return jnp.where(jnp.real(ratio) < HUB_CONTINUATION_START, continued, ratio)


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
    """C2 Figure 6 fit; finite hub/tip <= 1, stator=0, rotor=1.

    The fit is unchanged above 0.5. Below 0.5 the effective-coordinate map
    tends to 0.3, permitting zero and negative raw ratios. Those ratios
    remain empirical extrapolation. Apply the coordinate map exactly once.

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
    valid = (jnp.isfinite(ratio) & (jnp.real(ratio) <= 1)
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
# A value of 0.7 is an engineering safeguard for extreme extrapolation; it
# is not fitted to KO data and does not cap the complete profile loss.
SHOCK_ASYMPTOTE = 0.7


def continue_shock_loss(raw_loss):
    """C2 upper continuation of the exit-normalized shock coefficient.

    Preserve raw values through 0.25; above that join, increase smoothly
    toward 0.7 without a finite hard plateau. With d=0.45 and
    t=(raw_loss-0.25)/d, the upper branch is 0.7-d/(1+t+t*t).
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
    r"""Return the continued KO inlet shock loss and diagnostic components.

    The row-frame inlet hub Mach uses the Figure 6 fit evaluated at the
    positive effective hub/tip ratio. The same effective ratio weights the
    hub shock to obtain a mean-inlet coefficient. Using one coordinate in
    both factors permits negative raw hub radii while retaining finite,
    nonnegative losses and the signed meanline geometry. The coordinate
    equals the raw ratio at and above 0.5 and tends to 0.3 below that join.

    At hub Mach<=0.4 shock is zero. A quintic C2 onset over 0.4-0.5
    joins the KO power (M_hub-0.4)^1.75. The mean-inlet coefficient is
    0.75*onset_strength*effective_hub_tip_ratio. Multiplication by the
    actual inlet/exit row-frame pressure-head ratio converts it to the
    exit reference. This raw coefficient is retained through 0.25; a C2
    rational continuation above 0.25 tends to 0.7, preventing unbounded
    loss amplification as exit pressure head becomes small. The 0.3 and
    0.7 limits and the smooth joins are explicit extrapolation choices.
    They bound the correlation without changing pressures or clipping
    geometry. The outer 0.914*f_Re multiplier is applied by get_profile_loss;
    its high-Re nonzero limit preserves an active shock contribution.

    Parameters
    ----------
    flow_parameters : mapping
        ``Ma_rel_in`` is the nonnegative mean inlet Mach in the blade frame.
        ``p0_rel_in``, ``p_in``, ``p0_rel_out`` and ``p_out`` are the actual
        row-frame total/static pressures in consistent units. For a stator
        this frame is absolute; for a rotor it is relative to the blade.
    geometry : mapping
        ``hub_tip_ratio_in`` is the raw inlet hub/tip ratio, finite and <= 1.
        Zero and negative values are accepted for meanline extrapolation.
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
        includes the continued positive hub/tip span weight. ``inlet_head``,
        ``exit_head`` and ``head_ratio`` expose the pressure conversion.
        ``Y_shock_raw`` is the unbounded exit-normalized coefficient.
        ``head_ratio_effective`` reconstructs Y_shock from the span-weighted
        inlet loss after the upper continuation; it equals head_ratio when
        the raw loss is at most 0.25 (also when the inlet shock is zero).
        All entries have the broadcast output shape. Example integration:
        ``Y_shock = get_shock_loss(flow, geometry, row)["Y_shock"]``.

    Notes
    -----
    Every loss/component plot uses this single evaluation. Real-part
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
    loss_mean = loss_hub * effective_hub_tip_ratio(ratio)
    head_ratio = q1 / jnp.where(jnp.real(q2) > 0, q2, jnp.nan)
    raw = loss_mean * head_ratio
    value = continue_shock_loss(raw)
    effective_head_ratio = jnp.where(jnp.real(raw) > SHOCK_CONTINUATION_START,
        value / jnp.where(jnp.real(loss_mean) > 0, loss_mean, 1.), head_ratio)
    pressure_valid = (jnp.isfinite(p01) & jnp.isfinite(p1) & jnp.isfinite(p02)
                      & jnp.isfinite(p2) & (jnp.real(p01) > 0) & (jnp.real(p1) > 0) & (jnp.real(p02) > 0) & (jnp.real(p2) > 0))
    valid = (pressure_valid & jnp.isfinite(m1) & (jnp.real(m1) >= 0) & jnp.isfinite(ratio)
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
        "hub_tip_ratio_in must be finite and <= 1":
            jnp.isfinite(geometry["hub_tip_ratio_in"])
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
