"""Aungier reference-profile equations used to benchmark the analytical AM fit.

These two reference curves have no Reynolds, Mach, shock or row-loss assembly.
They are the published profile-shape comparator for the measured AM chart.
"""
import numpy as np

def nozzle_blades(r_sc, angle_out):
    """
    Use Aungier correlation to compute the pressure loss coefficient for nozzle blades :cite:`aungier_turbine_2006`.

    This correlation is a formula that reproduces the figures from the Ainley and Mathieson original figures :cite:`ainley_method_1951`,
    and is a function of the pitch-to-chord ratio and exit relative flow angle. 

    The correlation uses the following equations:

    .. math::

        & \\beta_\\mathrm{tan} = 90 - \\beta_\\mathrm{ax} \\\\
        & \\left(\\frac{s}{c}\\right)_\\mathrm{min} = \\begin{cases}
                                                    0.46 + \\frac{\\beta_\\mathrm{tan}}{77} && \\text{if } \\beta_\\mathrm{tan} < 30 \\\\
                                                    0.614 + \\frac{\\beta_\\mathrm{tan}}{130} && \\text{if } \\beta_\\mathrm{tan} \\geq 30
                                                \\end{cases} \\\\
        & X = \\left(\\frac{s}{c}\\right) - \\left(\\frac{s}{c}\\right)_\\mathrm{min} \\\\
        & A = \\begin{cases}
                0.025 + \\frac{27 - \\beta_\\mathrm{tan}}{530} && \\text{if } \\beta_\\mathrm{tan} < 27 \\\\
                0.025 + \\frac{27 - \\beta_\\mathrm{tan}}{3085} && \\text{if } \\beta_\\mathrm{tan} \\geq 27
              \\end{cases} \\\\
        & B = 0.1583 - \\frac{\\beta_\\mathrm{tan}}{1640} \\\\
        & C = 0.08\\left(\\frac{\\beta_\\mathrm{tan}}{30}\\right)^2 -1 \\\\
        & n = 1 + \\frac{\\beta_\\mathrm{tan}}{30} \\\\
        & \\mathrm{Y_{p,reaction}} = \\begin{cases}
                A + BX^2 + CX^3 && \\text{if } \\beta_\\mathrm{tan} < 30 \\\\
                A + B |X|^n && \\text{if } \\beta_\\mathrm{tan} \\geq 30
              \\end{cases}   

    where:

        - :math:`s` is the pitch
        - :math:`c` is the chord
        - :math:`\\beta_\\mathrm{tan}` and :math:`\\beta_\\mathrm{ax}` is the exit relative flow angle with respect to tangential and axial direction. 
        
    Parameters
    ----------
    r_sc : float
        Pitch-to-chord ratio.
    angle_out : float
        Exit relative flow angle (in degrees).

    Returns
    -------
    float
        Pressure loss coefficient for impulse blades.

    """
    phi = 90 - angle_out
    r_sc_min = (0.46 + phi / 77) * (phi < 30) + (0.614 + phi / 130) * (phi >= 30)
    X = r_sc - r_sc_min
    A = (0.025 + (27 - phi) / 530) * (phi < 27) + (0.025 + (27 - phi) / 3085) * (phi >= 27)
    B = 0.1583 - phi / 1640
    C = 0.08 * ((phi / 30) ** 2 - 1)
    n = 1 + phi / 30
    Yp_reaction = (A + B * X ** 2 + C * X ** 3) * (phi < 30) + (A + B * abs(X) ** n) * (phi >= 30)
    return Yp_reaction

def impulse_blades(r_sc, angle_out):
    """
    Use Aungier correlation to compute the pressure loss coefficient for impulse blades :cite:`aungier_turbine_2006`.

    This correlation is a formula that reproduces the figures from the Ainley and Mathieson original figures :cite:`ainley_method_1951`,
    and is a function of the pitch-to-chord ratio and exit relative flow angle. 

    The correlation uses the following equations:

    .. math::

        & \\beta_\\mathrm{tan} = 90 - \\beta_\\mathrm{ax} \\\\
        & \\left(\\frac{s}{c}\\right)_\\mathrm{min} = 0.224 + 1.575\\left(\\frac{\\beta_\\mathrm{tan}}{90}\\right) - \\left(\\frac{\\beta_\\mathrm{tan}}{90}\\right)^2 \\\\
        & X = \\left(\\frac{s}{c}\\right) - \\left(\\frac{s}{c}\\right)_\\mathrm{min} \\\\
        & A = 0.242 - \\frac{\\beta_\\mathrm{tan}}{151} + \\left(\\frac{\\beta_\\mathrm{tan}}{127}\\right)^2 \\\\
        & B = \\begin{cases}
                0.3 + \\frac{30 - \\beta_\\mathrm{tan}}{50} && \\text{if } \\beta_\\mathrm{tan} < 30 \\\\
                0.3 + \\frac{30 - \\beta_\\mathrm{tan}}{275} && \\text{if } \\beta_\\mathrm{tan} \\geq 30
              \\end{cases} \\\\
        & C = 0.88 - \\frac{\\beta_\\mathrm{tan}}{42.4} + \\left(\\frac{\\beta_\\mathrm{tan}}{72.8}\\right)^2 \\\\
        & \\mathrm{Y_{p,impulse}} = A + BX^2 - CX^3 

    where:

        - :math:`s` is the pitch
        - :math:`c` is the chord
        - :math:`\\beta_\\mathrm{tan}` and :math:`\\beta_\\mathrm{ax}` is the exit relative flow angle with respect to tangential and axial direction. 
        
    Parameters
    ----------
    r_sc : float
        Pitch-to-chord ratio.
    angle_out : float
        Exit relative flow angle (in degrees).

    Returns
    -------
    float
        Pressure loss coefficient for impulse blades.

    """
    phi = 90 - angle_out
    r_sc_min = 0.224 + 1.575 * (phi / 90) - (phi / 90) ** 2
    X = r_sc - r_sc_min
    A = 0.242 - phi / 151 + (phi / 127) ** 2
    B = (0.3 + (30 - phi) / 50) * (phi < 30) + (0.3 + (30 - phi) / 275) * (phi >= 30)
    C = 0.88 - phi / 42.4 + (phi / 72.8) ** 2
    Yp_impulse = A + B * X ** 2 - C * X ** 3
    return Yp_impulse
