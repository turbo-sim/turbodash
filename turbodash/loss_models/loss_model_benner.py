"""Benner losses using shared production Kacker--Okapuu components.

Lengths use consistent units, chord is true chord and height is full blade
height. Angles are degrees from the meridional direction. Array operations
use NumPy; the backend seam does not imply a JIT guarantee.

The model uses the shared profile, trailing-edge and clearance calculations,
including their shock, Reynolds and physical-turning loading continuations.
Benner secondary and incidence correlations retain their own documented
angle treatment. Passage-vortex penetration is smoothly limited to half
blade height and weights the primary components consistently. Numerical
transition choices are documented beside the empirical equations and in
loss_model_analysis/loss_model_audit/individual_functions/.
"""

from ._backend import jnp
from .. import math
from . import _numerics
from . import loss_model_kacker_okapuu as lm_ko
from .loss_coefficient_conversion import (
    convert_kinetic_energy_to_stagnation_pressure_loss,
)

PENETRATION_MAX_RATIO = 0.5
PENETRATION_TRANSITION_HALF_WIDTH = 0.05
_FLOW_KEYS = {
    "Re_in",
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
    "leading_edge_diameter",
    "leading_edge_wedge_angle",
    "maximum_thickness",
    "meridional_chord",
    "height",
    "opening",
    "trailing_edge_thickness",
    "tip_clearance",
    "A_throat",
    "A_out",
    "stagger_angle",
}


def compute_losses(input_parameters):
    r"""
    Evaluate loss coefficient according to :cite:`benner_influence_1997`, :cite:`benner_empirical_2006-1` and :cite:`benner_empirical_2006`.

    This model split the total loss coefficient into profile, secondary, trailing edge, tip clearance and incidence losses.
    The loss coefficient are combined through the following relation:

    .. math::

        \mathrm{Y_{tot}} = (\mathrm{Y_{p}} + \mathrm{Y_{te}} + \mathrm{Y_{inc}})(1-z_\mathrm{eff}) + \mathrm{Y_{s}} + \mathrm{Y_{cl}}

    where :math:`Z` is the penetration depth of the passage vortex separation line in spanwise direction, and :math:`H` is the mean blade height. This quantity depend on the displacement thickness of the
    inlet endwall bounary layer, which is approximated from a reference value:

    .. math::

        \frac{\delta}{H} = \frac{\delta}{H}_\mathrm{ref} \left(\frac{\mathrm{Re_{in}}}{3\cdot10^5}\right)^{-1/7}

    The reference value may be specified by the user.

    The function calls a function for each loss component:

        - :math:`\mathrm{Y_{p}}` : `get_profile_loss`
        - :math:`\mathrm{Y_{te}}` : `get_trailing_edge_loss`
        - :math:`\mathrm{Y_{s}}` : `get_secondary_loss`
        - :math:`\mathrm{Y_{cl}}` : `get_tip_clearance_loss`
        - :math:`\mathrm{Y_{inc}}` : `get_incidence_loss`
        - :math:`Z/H` : `get_penetration_depth`


    Parameters
    ----------
    input_parameters : dict
        A dictionary containing containing the keys, ``flow``, ``geometry`` and ``loss_model``.

    Returns
    -------
    dict
        A dictionary with the loss components.

    Notes
    -----
    The primary-loss area fraction uses
    :math:`z_\mathrm{eff}=\operatorname{limit}(Z_\mathrm{raw}/H)`
    from :func:`limit_penetration_depth`, with maximum 0.5 and default
    transition 0.45--0.55. It multiplies profile, trailing-edge and
    incidence losses together; secondary and clearance losses are
    outside this weight. Diagnostics retain raw and effective
    penetration and the unweighted primary components.
    The minimum-loss reference ``beta_des`` is currently the inlet
    metal angle. Inlet and outlet flow inputs are absolute-frame
    values for a stator and relative-frame values for a rotor.
    """
    flow_parameters = input_parameters["flow"]
    geometry = input_parameters["geometry"]
    options = input_parameters["loss_model"]
    beta_des = geometry["leading_edge_angle"]
    cascade_type = geometry.get("cascade_type", "stator")
    if cascade_type not in lm_ko._CASCADE_TO_ID:
        raise ValueError(
            f"Unsupported cascade_type '{cascade_type}'. Expected one of {list(lm_ko._CASCADE_TO_ID)}."
        )
    cascade_type_id = lm_ko._CASCADE_TO_ID[cascade_type]
    delta_height_ref = options["inlet_displacement_thickness_height_ratio"]
    flow_jax = {
        k: jnp.asarray(v) for k, v in flow_parameters.items() if k in _FLOW_KEYS
    }
    geom_jax = {k: jnp.asarray(v) for k, v in geometry.items() if k in _GEOM_KEYS}
    losses = _compute_losses_benner_jit(
        flow_jax,
        geom_jax,
        cascade_type_id,
        delta_height_ref,
        beta_des,
        options.get(
            "penetration_transition_half_width", PENETRATION_TRANSITION_HALF_WIDTH
        ),
    )
    return losses


def _compute_losses_benner_jit(
    flow_parameters,
    geometry,
    cascade_type_id: int,
    delta_height_ref,
    beta_des,
    penetration_transition_half_width=PENETRATION_TRANSITION_HALF_WIDTH,
):
    r"""
    Assemble Benner with Kacker–Okapuu profile, trailing and clearance helpers.
    delta/H=delta_ref*(Re_in/3e5)^(-1/7). Weight all three primary components
    identically and return their sum plus secondary/clearance. Diagnostics precede
    dispatcher tuning. Historical name retained; Python validation and NumPy errstate
    are not JAX traceable.
    """
    _require_positive_finite(flow_parameters["Re_in"], "Inlet Reynolds number")
    _require_nonnegative_finite(
        delta_height_ref, "Reference displacement-thickness ratio"
    )
    delta_height = delta_height_ref * (flow_parameters["Re_in"] / 300000.0) ** (-1 / 7)
    diagnostics = get_penetration_diagnostics(
        flow_parameters,
        geometry,
        delta_height,
        transition_half_width=penetration_transition_half_width,
    )
    Y_p = lm_ko.get_profile_loss(flow_parameters, geometry, cascade_type_id)
    Y_te = lm_ko.get_trailing_edge_loss(flow_parameters, geometry)
    Y_s = get_secondary_loss(flow_parameters, geometry, delta_height)
    Y_cl = lm_ko.get_tip_clearance_loss(flow_parameters, geometry, cascade_type_id)
    Y_inc = get_incidence_loss(flow_parameters, geometry, beta_des, cascade_type_id)
    weight = diagnostics["primary_loss_weight"]
    diagnostics.update(
        profile_loss_unweighted=Y_p,
        trailing_loss_unweighted=Y_te,
        incidence_loss_unweighted=Y_inc,
        primary_loss_unweighted=Y_p + Y_te + Y_inc,
        primary_loss_weighted=weight * (Y_p + Y_te + Y_inc),
    )
    Y_p = Y_p * weight
    Y_te = Y_te * weight
    Y_inc = Y_inc * weight
    return {
        "loss_profile": Y_p,
        "loss_incidence": Y_inc,
        "loss_trailing": Y_te,
        "loss_secondary": Y_s,
        "loss_clearance": Y_cl,
        "loss_total": Y_p + Y_te + Y_inc + Y_s + Y_cl,
        "diagnostics": diagnostics,
    }


def get_secondary_loss(flow_parameters, geometry, delta_height):
    r"""
    Calculate the Benner secondary pressure-loss coefficient.

    The correlation of :cite:`benner_empirical_2006-1` and
    :cite:`benner_empirical_2006` uses the full mean blade height :math:`H`, true chord
    :math:`c`, stagger angle :math:`\xi`, and inlet endwall displacement
    thickness ratio :math:`\delta^*/H`. For :math:`AR=H/c\leq2`, it evaluates

    .. math::

        Y_s = \frac{0.038+0.41\tanh(1.20\delta^*/H)}
          {\sqrt{\cos\xi}\,CR\,(H/c)^{0.55}
           \left(\frac{\cos\bar\beta_\mathrm{out}}{\cos\xi}\right)^{0.55}}.

    For :math:`H/c>2`, the implemented expression is

    .. math::

        Y_s = \frac{0.052+0.56\tanh(1.20\delta^*/H)}
          {\sqrt{\cos\xi}\,CR\,(H/c)
           \left(\frac{\cos\bar\beta_\mathrm{out}}{\cos\xi}\right)^{0.55}}.

    The flow angles are first clipped to :math:`[-85^\circ,85^\circ]`.
    Writing those clipped angles as :math:`\bar\beta`, the convergence ratio is

    .. math::

        CR=\operatorname{smoothmax}\left(
           \frac{\cos\bar\beta_\mathrm{in}}
                {\cos\bar\beta_\mathrm{out}},0.1\right).

    Parameters
    ----------
    flow_parameters : dict
        Inlet and outlet row-frame flow angles ``beta_in`` and ``beta_out``
        in degrees. Use absolute angles for a stator and relative angles for a rotor.
    geometry : dict
        Full mean ``height``, true ``chord`` and ``stagger_angle`` in degrees.
    delta_height : float or array_like
        Inlet displacement thickness divided by full mean blade height, after
        the inlet Reynolds-number correction in :func:`compute_losses`.

    Returns
    -------
    float or array_like
        Dimensionless secondary pressure-loss coefficient. The Benner primary
        penetration weight is not applied to this component.

    Notes
    -----
    The two displayed equations are the original empirical branches. The
    implementation replaces their value/slope mismatch by a quintic Hermite patch
    on :math:`1.8<H/c<2.2`, matching each original endpoint value, first
    derivative and curvature. For branch :math:`C a^{-p}`, the endpoint
    derivatives are :math:`-pY/a` and :math:`p(p+1)Y/a^2`; powers 0.55 and 1
    are retained exactly outside the patch. This numerical width gives a
    positive decreasing transition without overshoot for nonnegative boundary
    layer thickness; it is not an empirical calibration. All joins are C2.

    The CR floor uses the strict C2 maximum with half-width 0.02
    in the dimensionless ratio, enforcing :math:`CR\geq0.1`. Heights/chords
    must be positive, displacement-thickness ratio nonnegative, and stagger
    must have positive cosine. The pre-existing flow-angle clips at 85 degrees
    are retained, so global smoothness across those guard locations is not
    claimed. The Benner penetration cap is unaffected by this secondary patch.
    """
    beta_in = jnp.clip(flow_parameters["beta_in"], -85.0, 85.0)
    beta_out = jnp.clip(flow_parameters["beta_out"], -85.0, 85.0)
    height = geometry["height"]
    chord = geometry["chord"]
    stagger = geometry["stagger_angle"]
    AR = height / chord
    CR = math.cosd(beta_in) / math.cosd(beta_out)
    CR = _numerics.smooth_maximum(CR, 0.1, 0.02)
    shape = (
        jnp.sqrt(math.cosd(stagger))
        * CR
        * (math.cosd(beta_out) / math.cosd(stagger)) ** 0.55
    )
    coefficient_low = (0.038 + 0.41 * jnp.tanh(1.2 * delta_height)) / shape
    coefficient_high = (0.052 + 0.56 * jnp.tanh(1.2 * delta_height)) / shape
    Y_sec_low, Y_sec_high = coefficient_low / AR**0.55, coefficient_high / AR
    a, b = 1.8, 2.2
    y0, y1 = coefficient_low / a**0.55, coefficient_high / b
    patch = _numerics.quintic_hermite(
        AR,
        a,
        b,
        y0,
        y1,
        -0.55 * y0 / a,
        -y1 / b,
        0.55 * 1.55 * y0 / a**2,
        2 * y1 / b**2,
    )
    Y_sec = jnp.where(AR <= a, Y_sec_low, jnp.where(AR >= b, Y_sec_high, patch))
    return Y_sec


def get_incidence_loss(flow_parameters, geometry, beta_des, cascade_type_id: int):
    r"""
    Calculate the incidence loss coefficient according to the correlation proposed by :cite:`benner_influence_1997`.

    This model calculates the incidence loss parameter through the function `get_incidence_parameter`. The kinetic energu coefficient
    can be obtained from `get_incidence_profile_loss_increment`, which is a function of the incidence parameter. The kinetic energy loss coefficient is converted
    to the total pressure loss coefficient through `convert_kinetic_energy_to_stagnation_pressure_loss`.

    Parameters
    ----------
    flow_parameters : dict
        Dictionary containing flow-related parameters.
    geometry : dict
        Dictionary with geometric parameters.
    beta_des : float
        Reference inlet flow angle in degrees; currently the inlet metal angle.

    cascade_type_id : int
        Row identifier (0 stator, 1 rotor), controlling the incidence sign.

    Returns
    -------
    float
        Incidence loss coefficient.

    Notes
    -----
    The exit reference angle supplied to the incidence parameter is
    the unsigned gauging angle
    :math:`\theta_\mathrm{out}=\arccos(A_\mathrm{throat}/A_\mathrm{out})`.
    ``leading_edge_wedge_angle`` and all flow/metal angles are in degrees.
    The kinetic-to-pressure conversion uses its optional Mach preprocessing
    and compact pressure-loss cap. The numerical incidence join preserves
    the polynomial's zero reference value. Its small negative positive-side
    minimum is retained; this routine does not redefine the reference angle
    or impose nonnegative signed incidence increments.
    """
    beta_in = flow_parameters["beta_in"]
    gamma = flow_parameters["gamma_out"]
    Ma_rel_out = flow_parameters["Ma_rel_out"]
    le = geometry["leading_edge_diameter"]
    s = geometry["pitch"]
    We = geometry["leading_edge_wedge_angle"]
    theta_in = geometry["leading_edge_angle"]
    theta_out = math.arccosd(geometry["A_throat"] / geometry["A_out"])
    chi = get_incidence_parameter(
        le, s, We, theta_in, theta_out, beta_in, beta_des, cascade_type_id
    )
    delta_phi2 = get_incidence_profile_loss_increment(chi)
    Y_inc = convert_kinetic_energy_to_stagnation_pressure_loss(
        Ma_rel_out, gamma, delta_phi2, limit_output=True
    )
    return Y_inc


def get_incidence_profile_loss_increment(chi, chi_extrapolation=5, loss_limit=0.5):
    r"""
    Computes the increment in profile losses due to the effect of incidence according to the correlation proposed by :cite:`benner_influence_1997`.

    The increment in profile losses due to incidence is based on the incidence parameter :math:`\chi`.
    It is obtained from :func:`get_incidence_parameter`, including the
    row-dependent incidence sign and the metal/gauging-angle convention.

    Depending on the value of :math:`\chi`, two equations are used for computing the increment in profile losses:

    1. For :math:`\chi \geq 0`:

    .. math::

        \Delta \phi_{p}^2 = \sum_{i=1}^{8} a_i \, \chi^i

    with coefficients:

    .. math::

        a_1 = -6.149 \times 10^{-5}  \\
        a_2 = +1.327 \times 10^{-3}  \\
        a_3 = -2.506 \times 10^{-4}  \\
        a_4 = -1.542 \times 10^{-4}  \\
        a_5 = +9.017 \times 10^{-5}  \\
        a_6 = +1.106 \times 10^{-5}  \\
        a_7 = -5.318 \times 10^{-6}  \\
        a_8 = +3.711 \times 10^{-7}

    2. For :math:`\chi < 0`:

    .. math::

        \Delta \phi_{p}^2 = \sum_{i=1}^{2} b_i \, \chi^i

    with coefficients:

    .. math::

        b_1 = -8.720 \times 10^{-4}  \\
        b_2 = +1.358 \times 10^{-4}

    This function also implements two safeguard methods to prevent unrealistic values of the pressure loss increment
    when the value of :math:`\chi` is outside the range of validity of the correlation

    1. Linear extrapolation beyond a certain :math:`\chi`:
        Beyond a certain value of the distance parameter :math:`\hat{\chi}` the function linearly extrapolates based on the slope of the polynomial.

    2. Loss Limiting:
        The profile loss increment is limited to a maximum value :math:`\Delta \hat{\phi}_{p}^2` to prevent potential over-predictions. A compact numerical limiter enforces that upper bound exactly and retains the raw value outside its transition.

    The final expression for the loss coefficient increment can be summarized as:

    .. math::

        \begin{aligned}
        \Delta \phi_{p}^2 =
        \begin{cases}
        \sum_{i=1}^{2} b_i \, \chi^i & \text{for } \chi < 0 \\
        \sum_{i=1}^{8} a_i \, \chi^i & \text{for } 0 \leq \chi  \leq \hat{\chi} \\
        \Delta\phi_{p}^2(\hat{\chi}) + \mathrm{slope}(\hat{\chi}) \cdot(\chi - \hat{\chi}) & \text{for } \chi > \hat{\chi}
        \end{cases}
        \end{aligned}

    .. math::
        \Delta \phi_{p}^2 = \operatorname{compactmin}_{w}(\Delta \phi_{p}^2, \Delta \hat{\phi}_{p}^2)


    Parameters
    ----------
    chi : array-like
        The distance parameter, :math:`\chi`, used to compute the profile losses increment.
    chi_extrapolation : float, optional
        The value of :math:`\chi` beyond which linear extrapolation is applied. Default is 5.
    loss_limit : float or None, optional
        The upper limit to which the profile loss increment is smoothly constrained. Default is 0.5.

    Returns
    -------
    array-like
        Increment in profile losses due to the effect of incidence.

    Notes
    -----
    Two numerical modifications are introduced here, independently of the
    published polynomial coefficients. On :math:`-0.1<\chi<0`, a quintic
    Hermite polynomial matches the negative branch's value, first derivative
    and second derivative at -0.1, and the unchanged positive branch's
    corresponding values at zero. The latter are
    :math:`(0,-6.149\,10^{-5},2(1.327\,10^{-3}))`. The join is C2,
    preserves the zero reference, and remains decreasing across the patch.
    Its width 0.1 is a local numerical choice (2% of the default extrapolation
    location), not a literature calibration. Outside this interval both
    empirical polynomials are unchanged. The positive polynomial's small
    negative minimum is intentionally retained, since removing it would
    change the incidence reference convention.

    For a positive finite ``loss_limit`` the compact C2 upper limiter is
    :func:`~turbodash.loss_models._numerics.smooth_minimum`, with
    :math:`w=\min(0.02,0.04\,\Delta\hat\phi_p^2)`. At the default
    limit 0.5 its transition is raw loss 0.48--0.52. The result equals raw
    loss below 0.48 and equals 0.5 above 0.52, never exceeds the bound,
    and introduces no offset at zero. Passing ``None`` disables this limiter.
    For nonpositive limits the existing interface is retained with an exact
    minimum; a C2 physical-loss cap requires a positive limit.
    The slope-matched linear extension at ``chi_extrapolation`` remains C1,
    not generally C2. All losses and :math:`\chi` are dimensionless.
    """
    coeffs_1 = jnp.array(
        [
            +3.711e-07,
            -5.318e-06,
            +1.106e-05,
            +9.017e-05,
            -0.0001542,
            -0.0002506,
            +0.001327,
            -6.149e-05,
        ]
    )
    func_1 = lambda x: jnp.sum(
        jnp.asarray([a * x**i for i, a in enumerate(coeffs_1[::-1], start=1)]), axis=0
    )
    coeffs_2 = jnp.array([0.0001358, -0.000872])
    func_2 = lambda x: jnp.sum(
        jnp.asarray([b * x**i for i, b in enumerate(coeffs_2[::-1], start=1)]), axis=0
    )
    loss_poly = jnp.where(chi >= 0, func_1(chi), func_2(chi))
    join_lower = -0.1
    joined = _numerics.quintic_hermite(
        chi,
        join_lower,
        0.0,
        func_2(join_lower),
        0.0,
        2 * coeffs_2[0] * join_lower + coeffs_2[1],
        coeffs_1[-1],
        2 * coeffs_2[0],
        2 * coeffs_1[-2],
    )
    loss_poly = jnp.where((chi > join_lower) & (chi < 0), joined, loss_poly)
    _chi = chi_extrapolation
    loss = func_1(_chi)
    coeffs_slope = jnp.array([i * a for i, a in enumerate(coeffs_1[::-1], start=1)])
    slope = jnp.sum(
        jnp.array([a * _chi ** (i - 1) for i, a in enumerate(coeffs_slope, start=1)])
    )
    loss_extrap = loss + slope * (chi - _chi)
    loss_increment = jnp.where(chi > _chi, loss_extrap, loss_poly)
    if loss_limit is not None:
        loss_increment = _numerics.smooth_minimum(
            loss_increment,
            loss_limit,
            half_width=jnp.minimum(0.02, jnp.maximum(0.04 * loss_limit, 0.0)),
        )
    return loss_increment


def get_incidence_parameter(
    le, s, We, theta_in, theta_out, beta_in, beta_des, cascade_type_id: int
):
    r"""
    Calculate the incidence parameter according to the correlation proposed by :cite:`benner_influence_1997`.

    The incidence parameter is used to calculate the increment in profile losses due to the effect of incidence according to function `get_incidence_profile_loss_increment`.

    The quantity is calculated as:

    .. math::

        \chi = \left(\frac{d_\mathrm{le}}{s}\right)^{-0.05}\mathrm{We}_{\mathrm{le}}^{-0.2}\left(\frac{\cos\theta_\mathrm{in}}{\cos\theta_\mathrm{out}}\right)^{-1.4} i,

    .. math::

        i=\begin{cases}-(\beta_\mathrm{in}-\beta_\mathrm{des}) & \text{stator},\\+(\beta_\mathrm{in}-\beta_\mathrm{des}) & \text{rotor}.\end{cases}

    where:

        - :math:`\mathrm{d_{le}}` is the leading edge diameter.
        - :math:`s` is the pitch.
        - :math:`\mathrm{We_{le}}` is the leading edge wedge angle.
        - :math:`\theta_\mathrm{in}` and :math:`\theta_\mathrm{out}` are the inlet metal angle and the exit reference/gauging angle, respectively.
        - :math:`\beta_\mathrm{in}` and :math:`\beta_\mathrm{des}` is the inlet relative flow angle at given and design conditions respectively.

    Parameters
    ----------
    le : float
        Leading edge diameter.
    s : float
        Pitch.
    We : float
        Leading edge wedge angle (in degrees).
    theta_in : float
        Leading edge metal angle (in degrees).
    theta_out : float
        Exit reference angle (degrees); the row wrapper uses the unsigned gauging angle.
    beta_in : float
        Inlet relative flow angle (in degrees).
    beta_des : float
        Inlet relative flow angle at design condition (in degrees).

    cascade_type_id : int
        Row identifier: 0 selects stator incidence, 1 selects rotor incidence.

    Returns
    -------
    float
        Incidence parameter.
    """
    incidence = jnp.where(
        cascade_type_id == lm_ko._CASCADE_TO_ID["stator"],
        -(beta_in - beta_des),
        +(beta_in - beta_des),
    )
    cosine_ratio = math.cosd(theta_in) / math.cosd(theta_out)
    chi = (le / s) ** (-0.05) * We ** (-0.2) * cosine_ratio ** (-1.4) * incidence
    return chi


def _require_positive_finite(value, name):
    r"""
    Raise ValueError unless every supplied element is finite and strictly positive.
    Python branching retains the existing limitation on JAX tracing.
    """
    value = jnp.asarray(value)
    if not jnp.all(jnp.isfinite(value) & (value > 0)):
        raise ValueError(f"{name} must be finite and positive.")


def _require_nonnegative_finite(value, name):
    r"""
    Raise ValueError unless every supplied element is finite and nonnegative. Python
    branching retains the existing limitation on JAX tracing.
    """
    value = jnp.asarray(value)
    if not jnp.all(jnp.isfinite(value) & (value >= 0)):
        raise ValueError(f"{name} must be finite and nonnegative.")


def limit_penetration_depth(
    penetration_raw_ratio, *, transition_half_width=PENETRATION_TRANSITION_HALF_WIDTH
):
    r"""
    Smoothly limit vortex penetration to half the full blade height.

    Let :math:`z=Z_\mathrm{raw}/H`, :math:`z_\mathrm{max}=0.5`, and let
    :math:`d` denote ``transition_half_width``. With
    :math:`l=z_\mathrm{max}-d`, :math:`u=z_\mathrm{max}+d` and
    :math:`t=(z-l)/(2d)`, the implemented compact transition is

    .. math::

        z_\mathrm{eff}=\begin{cases}
           z & z\leq l,\\
           l+2d\left(t-t^3+\tfrac12t^4\right) & l<z<u,\\
           z_\mathrm{max} & z\geq u.
        \end{cases}

    Its first derivative in the transition is :math:`1-3t^2+2t^3`, providing
    continuous first and second derivatives at both joins. The default
    :math:`d=0.05` gives an identity branch through 0.45 and exact saturation
    from 0.55. At raw penetration 0.5, the effective ratio is 0.490625.

    Parameters
    ----------
    penetration_raw_ratio : float or array_like
        Finite, nonnegative raw penetration divided by full mean blade height.
    transition_half_width : float, optional
        Scalar half-width :math:`0<d<0.5`; default is 0.05.

    Returns
    -------
    float or array_like
        Effective penetration ratio used in the primary-loss area weight.

    Raises
    ------
    ValueError
        If raw inputs are negative/nonfinite, or the transition width is invalid
        or too small to represent its two endpoints.

    Notes
    -----
    This is the existing numerical regularization, not a new empirical fit.
    Inputs support NumPy scalars and arrays. Python validation and NumPy error
    contexts retain the present limitation on JAX tracing.
    """
    raw = jnp.asarray(penetration_raw_ratio, dtype=float)
    _require_nonnegative_finite(raw, "Raw penetration ratio")
    d = jnp.asarray(transition_half_width, dtype=float)
    if d.ndim != 0 or not jnp.isfinite(d) or (not 0 < d < PENETRATION_MAX_RATIO):
        raise ValueError(
            "Penetration transition half-width must be finite and between 0 and 0.5."
        )
    lower, upper = (PENETRATION_MAX_RATIO - d, PENETRATION_MAX_RATIO + d)
    if not lower < PENETRATION_MAX_RATIO < upper:
        raise ValueError("Penetration transition half-width is too small to represent.")
    t = (jnp.clip(raw, lower, upper) - lower) / (2 * d)
    transition = lower + 2 * d * (t - t**3 + 0.5 * t**4)
    return jnp.where(
        raw <= lower, raw, jnp.where(raw >= upper, PENETRATION_MAX_RATIO, transition)
    )


def get_penetration_diagnostics(
    flow_parameters,
    geometry,
    delta_height,
    *,
    transition_half_width=PENETRATION_TRANSITION_HALF_WIDTH,
):
    r"""
    Evaluate raw and effective penetration and the primary-loss area weight.

    This retains the raw prediction of
    :func:`get_penetration_depth_raw` before applying the smooth
    cap of :func:`limit_penetration_depth`. The area weighting is

    .. math::

        w_\mathrm{primary}=1-z_\mathrm{eff},\qquad
        z_\mathrm{eff}=\operatorname{limit}(Z_\mathrm{raw}/H).

    Parameters
    ----------
    flow_parameters : dict
        Inlet/exit row-frame flow angles in degrees.
    geometry : dict
        Pitch, true chord, meridional chord and full mean blade height.
    delta_height : float or array_like
        Inlet displacement thickness divided by full mean height.
    transition_half_width : float, optional
        Half-width of the smooth transition about 0.5; default is 0.05.

    Returns
    -------
    dict
        Raw and effective penetration ratios, primary area weight,
        loading parameter, convergence ratio, true-chord aspect ratio,
        displacement-thickness ratio, smoothing flag and outside-domain
        flag. These metadata are not additional loss terms.

    Notes
    -----
    Raw predictions above half height remain visible as extrapolation
    diagnostics. The flag does not validate overlapping secondary-flow
    regions or modify the secondary-loss correlation.
    """
    diagnostics = _get_raw_penetration_diagnostics(
        flow_parameters, geometry, delta_height
    )
    raw = diagnostics["penetration_raw_ratio"]
    effective = limit_penetration_depth(
        raw, transition_half_width=transition_half_width
    )
    diagnostics.update(
        penetration_effective_ratio=effective,
        primary_loss_weight=1 - effective,
        penetration_smoothing_active=(
            raw > PENETRATION_MAX_RATIO - transition_half_width
        )
        & (raw < PENETRATION_MAX_RATIO + transition_half_width),
        penetration_outside_model_domain=raw > PENETRATION_MAX_RATIO,
    )
    return diagnostics


def get_penetration_depth(
    flow_parameters,
    geometry,
    delta_height,
    *,
    transition_half_width=PENETRATION_TRANSITION_HALF_WIDTH,
):
    r"""
    Calculate effective vortex penetration divided by full blade height.

    The raw empirical prediction is evaluated by
    :func:`get_penetration_depth_raw`; the existing smooth half-height
    bound is then applied by :func:`limit_penetration_depth`:

    .. math::

        z_\mathrm{raw}=\frac{0.10F_t^{0.79}}
          {\sqrt{CR}(H/c)^{0.55}}
          +32.70\left(\frac{\delta^*}{H}\right)^2,
        \qquad z_\mathrm{eff}=\operatorname{limit}(z_\mathrm{raw}).

    Parameters
    ----------
    flow_parameters : dict
        Inlet and exit row-frame flow angles in degrees.
    geometry : dict
        Pitch, true chord, meridional chord and full mean blade height.
    delta_height : float or array_like
        Inlet displacement thickness divided by full mean height.
    transition_half_width : float, optional
        Half-width of the smooth transition about 0.5; default is 0.05.

    Returns
    -------
    float or array_like
        Effective penetration ratio in [0, 0.5], used for weighting
        primary losses. Raw values remain available separately.

    See Also
    --------
    get_penetration_depth_raw, get_penetration_diagnostics
    """
    raw = get_penetration_depth_raw(flow_parameters, geometry, delta_height)
    return limit_penetration_depth(raw, transition_half_width=transition_half_width)


def get_penetration_depth_raw(flow_parameters, geometry, delta_height):
    r"""
    Calculated the penetration depth of the passage vortex separation line relative to the blade span.

    The endwall inlet boundary layer generate a vortex which propagtes through the cascade section,
    and the spanwise penetration of this vortex affect the magnutude of the secondary loss coefficient.
    This function approximate the vortex penetration depth to the blade span by the correlation developed by :cite:`benner_empirical_2006-1`.

    The quantity is calculated as:

    .. math::

        \frac{\mathrm{Z_{te}}}{H} = \frac{0.10F_t^{0.79}}{\sqrt{CR}\left(\frac{H}{c}\right)^{0.55}} + 32.70\left(\frac{\delta^*}{H}\right)^2

    where:

        - :math:`CR = \frac{\cos(\beta_\mathrm{in})}{\cos(\beta_\mathrm{out})}` is the convergence ratio.
        - :math:`H` is the mean height.
        - :math:`c` is the chord.
        - :math:`\delta^*` is the inlet endwall boundary layer displacement thickness.
        - :math:`\beta_\mathrm{out}` is the exit relative flow angle.
        - :math:`F_t` is the tangential loading coefficient.

    The tangiential loading coefficient is calculated by a separate function (`F_t`).

    Parameters
    ----------
    flow_parameters : dict
        Dictionary containing flow-related parameters.
    geometry : dict
        Dictionary with geometric parameters.
    delta_height : float
        The inlet endwall boundary layer displacement thickness relative to the mean blade height.

    Returns
    -------
    float
        Unbounded penetration relative to full mean blade height. Values above
        0.5 are outside the non-overlapping-endwall construction.
    """
    return _get_raw_penetration_diagnostics(flow_parameters, geometry, delta_height)[
        "penetration_raw_ratio"
    ]


def _get_raw_penetration_diagnostics(flow_parameters, geometry, delta_height):
    r"""
    Validate strict flow-angle domain (-90,90), positive dimensions and nonnegative
    delta_height. Return raw=.10*F_t^.79/sqrt(CR)/AR^.55+32.70*delta_height^2,
    CR=cos(beta_in)/cos(beta_out), AR=height/true chord, BSx=meridional_chord/pitch.
    Report CR, AR, F_t and delta_height. Raw CR is not the secondary helper
    smoothed/clipped CR.
    """
    beta_in = jnp.asarray(flow_parameters["beta_in"])
    beta_out = jnp.asarray(flow_parameters["beta_out"])
    for angle in (beta_in, beta_out):
        if not jnp.all(jnp.isfinite(angle) & (jnp.abs(angle) < 90)):
            raise ValueError(
                "Benner flow angles must be finite and strictly between -90 and 90 degrees."
            )
    for key in ("meridional_chord", "pitch", "chord", "height"):
        _require_positive_finite(geometry[key], f"Benner {key}")
    _require_nonnegative_finite(delta_height, "Inlet displacement-thickness ratio")
    meridional_chord = jnp.asarray(geometry["meridional_chord"])
    pitch = jnp.asarray(geometry["pitch"])
    chord = jnp.asarray(geometry["chord"])
    height = jnp.asarray(geometry["height"])
    CR = math.cosd(beta_in) / math.cosd(beta_out)
    with jnp.errstate(over="ignore", divide="ignore", invalid="ignore"):
        BSx = meridional_chord / pitch
        AR = height / chord
        Ft = F_t(BSx, beta_in, beta_out)
        Z_TE = (
            0.1 * Ft**0.79 / jnp.sqrt(CR) / AR**0.55
            + 32.7 * jnp.asarray(delta_height) ** 2
        )
    _require_positive_finite(CR, "Benner convergence ratio")
    _require_positive_finite(AR, "Benner true-chord aspect ratio")
    _require_nonnegative_finite(Ft, "Benner tangential loading parameter")
    _require_nonnegative_finite(Z_TE, "Raw penetration ratio")
    return {
        "penetration_raw_ratio": Z_TE,
        "tangential_loading_parameter": Ft,
        "convergence_ratio": CR,
        "aspect_ratio_true_chord": AR,
        "inlet_displacement_thickness_height_ratio": jnp.asarray(delta_height),
    }


def F_t(BSx, beta_in, beta_out):
    r"""
    Calculate the tangential loading coefficient according to the correlation proposed by :cite:`benner_empirical_2006-1`.

    .. math::

        F_t = 2\frac{s}{c_\mathrm{ax}}\cos^2(\beta_m)
              (|\tan(\beta_\mathrm{in})| + |\tan(\beta_\mathrm{out})|)

    where:

        - :math:`s` is the pitch.
        - :math:`c_\mathrm{ax}` is the axial chord.
        - :math:`\beta_m = \tan^{-1}(0.5(\tan(\beta_\mathrm{in}) + \tan(\beta_\mathrm{out})))` is the mean vector angle.
        - :math:`\beta_\mathrm{in}` and :math:`\beta_\mathrm{out}` is the inlet and outlet relative flow angle.

    Tangents use the profile-reference bounded C2 coordinates (exact through
    +/-80 degrees, asymptotic to +/-tan87.5); the mean cosine is consistent
    with those coordinates. This is Benner's own penetration input, not
    the KO/clearance loading Z, which preserves physical turning and
    continues only its mean secant. Absolute-tangent cusps at zero remain part of
    this correlation. The signed-tangent convention
    in the reference differs for same-sign angles and requires a separate
    correlation audit; it is not changed by the penetration limiter.

    Parameters
    ----------
    BSx : float
        Blade solidity based on axial chord.
    beta_in : float
        Inlet relative flow angle (in degrees).
    beta_out : float
        Exit relative flow angle (in degrees).

    Returns
    -------
    float
        Tangetnial loading parameter.
    """
    # Benner's penetration input retains its own bounded tangent convention.
    tangent_in = lm_ko.get_blade_tangent(beta_in)
    tangent_out = lm_ko.get_blade_tangent(beta_out)
    mean_tangent = .5*(tangent_in+tangent_out)
    F_t = (
        2.0
        * 1.0
        / BSx
        / (1+mean_tangent**2)
        * (jnp.abs(tangent_in) + jnp.abs(tangent_out))
    )
    return F_t
