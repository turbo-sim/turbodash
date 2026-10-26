"""Moustapha incidence corrections with production Kacker--Okapuu losses.

Lengths use consistent units, chord is true chord and height is full blade
height. Angles are degrees from the meridional direction. Array operations
use NumPy; the backend seam does not imply a JIT guarantee.

The empirical incidence and secondary-correction equations use the shared
KO components and coefficient conversions, including their documented
numerical continuations. The turbine solver supplies the unsigned exit
gauging angle from acos(A_throat/A_out), including radial opening geometry.
Prescribed row inputs must supply this angle explicitly.
"""

from ._backend import jnp
from .. import math
from . import loss_model_kacker_okapuu as lm_ko
from .loss_coefficient_conversion import (
    convert_kinetic_energy_to_stagnation_pressure_loss,
)

_FLOW_KEYS = {
    "Re_out",
    "Re_in",
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
    "maximum_thickness",
    "meridional_chord",
    "height",
    "opening",
    "trailing_edge_thickness",
    "tip_clearance",
    "A_throat",
    "A_out",
    "gauging_angle",
}


def compute_losses(input_parameters):
    r"""
    Evaluate loss coefficient according to :cite:`kacker_mean_1982` and :cite:`moustapha_improved_1990`.

    This model split the total loss coefficient into profile, secondary, trailing edge, tip clearance and incidence.
    The model adopt the incidence loss coefficient proposed by  :cite:`moustapha_improved_1990`, but is otherwise similar to the
    model proposed by :cite:`kacker_mean_1982`.
    The loss coefficient are combined through the following relation:

    .. math::

        \mathrm{Y_{tot}} = \mathrm{Y_{p}} + \mathrm{Y_{te}} + \mathrm{Y_{corr}}\mathrm{Y_{s}} + \mathrm{Y_{cl}} + \mathrm{Y_{inc}}

    The function calls a function for each loss component:

        - :math:`\mathrm{Y_{p}}` : `get_profile_loss`
        - :math:`\mathrm{Y_{te}}` : `get_trailing_edge_loss`
        - :math:`\mathrm{Y_{s}}` : `get_secondary_loss`
        - :math:`\mathrm{Y_{cl}}` : `get_tip_clearance_loss`
        - :math:`\mathrm{Y_{inc}}` : `get_incidence_loss`
        - :math:`\mathrm{Y_{corr}}` : `get_secondary_loss_correction_factor`

    Parameters
    ----------
    input_parameters : dict
        A dictionary containing the keys ``flow``, ``geometry`` and ``loss_model``.
        In ``loss_model``, optional boolean ``exit_mach_correction`` enables
        the shared KO supersonic exit profile-loss multiplier (default False).

    Returns
    -------
    dict
        A dictionary with the loss components.
    """
    flow_parameters = input_parameters["flow"]
    geometry = input_parameters["geometry"]
    beta_des = geometry["leading_edge_angle"]
    cascade_type = geometry.get("cascade_type", "stator")
    if cascade_type not in lm_ko._CASCADE_TO_ID:
        raise ValueError(
            f"Unsupported cascade_type '{cascade_type}'. Expected one of {list(lm_ko._CASCADE_TO_ID)}."
        )
    cascade_type_id = lm_ko._CASCADE_TO_ID[cascade_type]
    flow_jax = {
        k: jnp.asarray(v) for k, v in flow_parameters.items() if k in _FLOW_KEYS
    }
    geom_jax = {k: jnp.asarray(v) for k, v in geometry.items() if k in _GEOM_KEYS}
    losses = _compute_losses_moustapha_jit(
        flow_jax, geom_jax, cascade_type_id, beta_des,
        exit_mach_correction=input_parameters.get("loss_model", {}).get(
            "exit_mach_correction", False
        ),
    )
    return losses


def _compute_losses_moustapha_jit(
    flow_parameters, geometry, cascade_type_id: int, beta_des, *,
    exit_mach_correction=False,
):
    r"""
    Assemble Kacker–Okapuu profile, trailing, clearance and secondary components;
    multiply secondary by the Moustapha incidence correction and add own incidence.
    Historical name retained; no active JIT decoration.
    """
    Y_p = lm_ko.get_profile_loss(
        flow_parameters, geometry, cascade_type_id,
        exit_mach_correction=exit_mach_correction,
    )
    Y_s = lm_ko.get_secondary_loss(flow_parameters, geometry, cascade_type_id)
    Y_cl = lm_ko.get_tip_clearance_loss(flow_parameters, geometry, cascade_type_id)
    Y_te = lm_ko.get_trailing_edge_loss(flow_parameters, geometry)
    Y_inc = get_incidence_loss(flow_parameters, geometry, beta_des)
    Y_corr = get_secondary_loss_correction_factor(flow_parameters, geometry)
    Y_s = Y_s * Y_corr
    return {
        "loss_profile": Y_p,
        "loss_incidence": Y_inc,
        "loss_trailing": Y_te,
        "loss_secondary": Y_s,
        "loss_clearance": Y_cl,
        "loss_total": Y_p + Y_te + Y_inc + Y_s + Y_cl,
    }


def get_incidence_loss(flow_parameters, geometry, beta_des):
    r"""
    Calculate the incidence loss coefficient according to the correlation proposed by :cite:`moustapha_improved_1990`.

    The model first computes the incidence parameter, :math:`\chi`. Based on this parameter, the kinetic-energy incidence
    loss coefficient is calculated to match experimental data.

    .. math::

        \Delta\phi^2_p = \begin{cases}
                        -5.1734\times10^{-6}\chi + 7.6902\times10^{-9}\chi^2 & \text{if } \chi < 0 \\
                        0.778\times10^{-5}\chi + 0.56\times10^{-7}\chi^2 + 0.4\times10^{-10}\chi^3 + 2.054\times10^{-19}\chi^6 & \text{if } \chi \geq 0
                        \end{cases}

    The kinetic-energy coefficient is converted to total pressure loss coefficient through the `convert_kinetic_energy_to_stagnation_pressure_loss` function.

    Parameters
    ----------
    flow_parameters : dict
        Dictionary containing flow-related parameters.
    geometry : dict
        Dictionary with geometric parameters.
    beta_des : float
        Reference inlet flow angle in degrees; currently the inlet metal angle.

    Returns
    -------
    float
        Incidence loss coefficient.

    Notes
    -----
    The polynomials are evaluated beyond the legacy documented
    :math:`[-800,800]` interval without a range warning. The incidence
    parameter uses the difference of absolute inlet/reference angles,
    not their signed difference. The existing kinetic-to-pressure
    conversion applies its Mach preprocessing and output cap.
    """
    Ma_rel_out = flow_parameters["Ma_rel_out"]
    beta_in = flow_parameters["beta_in"]
    gamma = flow_parameters["gamma_out"]
    s = geometry["pitch"]
    le = geometry["leading_edge_diameter"]
    theta_in = geometry["leading_edge_angle"]
    theta_out = geometry["gauging_angle"]
    chi = get_incidence_parameter(le, s, theta_in, theta_out, beta_in, beta_des)
    dPhi_pos = 7.78e-06 * chi + 5.6e-08 * chi**2 + 4e-11 * chi**3 + 2.054e-19 * chi**6
    dPhi_neg = -5.1734e-06 * chi + 7.6902e-09 * chi**2
    dPhi = jnp.where(chi >= 0, dPhi_pos, dPhi_neg)
    Y_inc = convert_kinetic_energy_to_stagnation_pressure_loss(
        Ma_rel_out, gamma, dPhi, limit_output=True
    )
    return Y_inc


def get_secondary_loss_correction_factor(flow_parameters, geometry):
    r"""
    Calculate the correction factor for the secondary loss coefficient to account for incidence losses.

    The correction factor is calculated based on the secondary flow incidence parameter (:math:`\chi`), which is determined using
    geometrical and flow parameters.

    The correction factor (:math:`Y_\mathrm{corr}`) is given by:

    .. math::

        Y_\mathrm{corr} = \begin{cases}
        e^{0.9\chi} + 13\chi^2 + 400\chi^4 & \text{if } \chi \geq 0 \\
        e^{0.9\chi} & \text{if } \chi < 0
        \end{cases}

    and the incidence paramter is calculated as:

    .. math::

        \chi = \frac{|\beta_\mathrm{in}| - |\theta_\mathrm{in}|}{180 - (\theta_\mathrm{in} + \theta_\mathrm{out})}\left(\frac{\cos(\theta_\mathrm{in})}{\cos(\theta_\mathrm{out})}\right)^{-1.5}\left(\frac{\mathrm{d_{le}}}{c}\right)^{-0.3}

    where:

        - :math:`\chi` is the secondary flow incidence parameter.
        - :math:`c` is the chord length.
        - :math:`\mathrm{d_{le}}` is the leading edge diameter.
        - :math:`\theta_\mathrm{in}` is the inlet metal angle.
        - :math:`\theta_\mathrm{out}` is the exit gauging angle.
        - :math:`\beta_\mathrm{in}` is the inlet relative flow angle.

    The implementation evaluates this expression without a calibration-range warning or extrapolation guard.

    Parameters
    ----------
    flow_parameters : dict
        Dictionary containing flow-related parameters.
    geometry : dict
        Dictionary with geometric parameters.

    Returns
    -------
    float
        Secondary loss correction factor.
    """
    beta_in = flow_parameters["beta_in"]
    c = geometry["chord"]
    le = geometry["leading_edge_diameter"]
    theta_in = geometry["leading_edge_angle"]
    theta_out = geometry["gauging_angle"]
    chi = (
        (jnp.abs(beta_in) - jnp.abs(theta_in))
        / (180 - (theta_in + theta_out))
        * (math.cosd(theta_in) / math.cosd(theta_out)) ** (-1.5)
        * (le / c) ** (-0.3)
    )
    Y_corr_pos = jnp.exp(0.9 * chi) + 13 * chi**2 + 400 * chi**4
    Y_corr_neg = jnp.exp(0.9 * chi)
    Y_corr = jnp.where(chi >= 0, Y_corr_pos, Y_corr_neg)
    return Y_corr


def get_incidence_parameter(le, s, theta_in, theta_out, beta_in, beta_des):
    r"""
    Calculate the incidence parameter according to the correlation proposed by :cite:`moustapha_improved_1990`.

    The incidence parameter is used to calculate the increment in profile losses due to the effect of incidence according to function `get_incidence_loss`.

    The quantity is calculated as:

    .. math::

        \chi = \left(\frac{\mathrm{d_{le}}}{s}\right)^{-1.6}\left(\frac{\cos{\theta_\mathrm{in}}}{\cos{\theta_\mathrm{out}}}\right)^{-2}(|\beta_\mathrm{in}| - |\beta_\mathrm{des}|)

    where:

        - :math:`\mathrm{d_{le}}` is the leading edge diameter.
        - :math:`s` is the pitch.
        - :math:`\theta_\mathrm{in}` and :math:`\theta_\mathrm{out}` are the inlet metal angle and the exit reference/gauging angle, respectively.
        - :math:`\beta_\mathrm{in}` and :math:`\beta_\mathrm{des}` is the inlet relative flow angle at given and design conditions respectively.

    Parameters
    ----------
    le : float
        Leading edge diameter.
    s : float
        Pitch.
    theta_in : float
        Leading edge metal angle (in degrees).
    theta_out : float
        Exit gauging/reference angle (in degrees).
    beta_in : float
        Inlet relative flow angle (in degrees).
    beta_des : float
        Inlet relative flow angle at design condition (in degrees).

    Returns
    -------
    float
        Incidence parameter.
    """
    chi = (
        (le / s) ** (-1.6)
        * (math.cosd(theta_in) / math.cosd(theta_out)) ** (-2)
        * (abs(beta_in) - abs(beta_des))
    )
    return chi
