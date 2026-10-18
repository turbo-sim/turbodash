"""KO Figure 14 calibration, bounded angle blend and equation (18) conversion.

Integrated from loss_model_analysis/kacker_okapuu_trailing_edge. All fit
constants are literals; production imports no study scripts or data files.
Angles enter here only through the prescribed coordinate z. Inputs broadcast
and valid complex-step perturbations are retained by the NumPy backend.
"""
from ._backend import jnp

COEFFICIENTS = {'nozzle': (0.02564380076855907, 1.2332771855078724, -1.0332647877407999), 'impulse': (0.015128580684873045, 0.6394153340696378, -0.5515740653135338)}
REFERENCE_CEILINGS = {'nozzle': 0.25, 'impulse': 0.13}

RATIO_JOIN = .4
Z_TRANSITION = .05
Z_UPPER_GAIN = .2
Z_LOWER_REDUCTION = .2
MACH_TAIL_START = .9
MACH_ASYMPTOTE = 1.
GAMMA_MAX = 5/3


def reference_loss(thickness_opening, family):
    r"""Return one Figure 14 energy-loss reference with a C2 bounded tail.

    For x=t/o<=0.4, f=a*x+b*x**2+c*x**3. For h=x-0.4>0 use
    f=L-D/(1+A*h+B*h**2), where D=L-f0, A=f0'/D and
    B=A**2+f0''/(2*D). This matches value, slope and curvature.
    The fitted constants have D,A,B>0, so the tail is increasing and tends
    to L without a finite plateau. L=0.25 (nozzle), 0.13 (impulse) is an
    explicit numerical ceiling, not an empirical fit beyond the chart.
    Negative ratios are invalid. At x=0 the value is exactly zero.
    """
    x = jnp.asarray(thickness_opening)
    valid = jnp.isfinite(x) & (jnp.real(x) >= 0)
    x = jnp.where(valid, x, 0.)
    a, b, c = COEFFICIENTS[family]
    end, ceiling = RATIO_JOIN, REFERENCE_CEILINGS[family]
    y0 = end*(a+end*(b+c*end))
    slope = a+end*(2*b+3*c*end)
    curvature = 2*b+6*c*end
    delta = ceiling-y0
    linear = slope/delta
    quadratic = linear**2+curvature/(2*delta)
    inner = jnp.where((jnp.real(x) >= 0) & (jnp.real(x) <= end), x, end)
    polynomial = inner*(a+inner*(b+c*inner))
    h = jnp.where(jnp.real(x) > end, x-end, 0.)
    # Reciprocal form avoids overflowing h**2 far outside the plotted domain.
    inverse = 1/(1+h)
    fraction = h/(1+h)
    tail = ceiling-delta*inverse**2/(inverse**2+linear*fraction*inverse+quadratic*fraction**2)
    result = jnp.where(jnp.real(x) <= end, polynomial, tail)
    return jnp.where(valid, result, jnp.nan)


def _saturate_nonnegative(t):
    """t/sqrt(1+t^2), with stable small/large branches and complex derivatives."""
    small = jnp.where(jnp.real(t) <= 1, t, 1.)
    inverse = 1/jnp.where(jnp.real(t) > 1, t, 1.)
    return jnp.where(jnp.real(t) <= 1, small/jnp.sqrt(1+small*small), 1/jnp.sqrt(1+inverse*inverse))


def _signed_square(z):
    r"""C2 smoothing of z*abs(z) over |z|<w, preserving both outer branches.

    With q=z/w, use w^2*(3*q/8+3*q^3/4-q^5/8). This odd polynomial
    matches value, first and second derivatives at +/-w; its value is zero
    at z=0. The small nonzero slope at zero is deliberate. The unsmoothed
    signed square is only C1 there. Outer inputs are limited here to |z|<=1;
    the remote tails in energy_loss use reciprocal angle ratios.
    """
    w = Z_TRANSITION
    q = jnp.where(abs(jnp.real(z)) < w, z/w, 0.)
    polynomial = w*w*q*(3/8+q*q*(3/4-q*q/8))
    square = jnp.where(jnp.real(z) >= 0, z*z, -z*z)
    return jnp.where(abs(jnp.real(z)) < w, polynomial, square)


def energy_loss(thickness_opening, z):
    r"""Blend nozzle R and impulse I with bounded angle-ratio extrapolation.

    z=-theta_in/alpha2, using signed production angles. On w<=z<=1,
    d=(1-z^2)*R+z^2*I, exactly KO equation (17). Let W be the C2 signed
    square. Inside |z|<w the same equation uses W instead of z*abs(z).

    For W<0, d=R+0.2*R*S((R-I)*(-W)/(0.2*R)). For W>1,
    d=I-0.2*I*S((R-I)*(W-1)/(0.2*I)), with S(t)=t/sqrt(1+t^2).
    Values, slopes and curvature match at W=0 and W=1. Thus
    0.8*I<=d<=1.2*R<=0.3; zero thickness gives zero exactly. This preserves
    the paper's direction for negative z (increasing loss) and prevents
    negative losses for z>1, where the lower impulse curve would otherwise
    be overshot. Reference ratios at x=0 use the reduced cubics so that
    thickness derivatives there are preserved without division by zero.
    """
    x, z = jnp.broadcast_arrays(jnp.asarray(thickness_opening), jnp.asarray(z))
    reaction, impulse = reference_loss(x, 'nozzle'), reference_loss(x, 'impulse')
    local = jnp.where((jnp.real(x) >= 0) & (jnp.real(x) <= RATIO_JOIN), x, 0.)
    ar, br, cr = COEFFICIENTS['nozzle']
    ai, bi, ci = COEFFICIENTS['impulse']
    ratio_inner = (ai+local*(bi+ci*local))/(ar+local*(br+cr*local))
    ratio = jnp.where(jnp.real(x) <= RATIO_JOIN, ratio_inner,
                     impulse/jnp.where(jnp.real(x) > RATIO_JOIN, reaction, 1.))
    # Bounded inputs in the local polynomial avoid squaring huge angle ratios.
    local_z = jnp.where(abs(jnp.real(z)) <= 1, z, jnp.where(jnp.real(z) > 0, 1., -1.))
    weight = _signed_square(local_z)
    inside = 1+(ratio-1)*weight
    negative_t = (1-ratio)*jnp.where(jnp.real(weight) < 0, -weight, 0.)/Z_UPPER_GAIN
    negative = 1+Z_UPPER_GAIN*_saturate_nonnegative(negative_t)
    remote_z = jnp.where(abs(jnp.real(z)) > 1, z, 2.)
    inv_square = (1/remote_z)**2
    negative_inverse_t = Z_UPPER_GAIN*inv_square/(1-ratio)
    negative_remote = 1+Z_UPPER_GAIN/jnp.sqrt(1+negative_inverse_t**2)
    # For z>1, evaluate saturation using 1/t; protect the inactive z=1 branch.
    positive_inverse_t = Z_LOWER_REDUCTION*ratio*inv_square/((1-ratio)*(1-inv_square))
    positive_remote = ratio*(1-Z_LOWER_REDUCTION/jnp.sqrt(1+positive_inverse_t**2))
    factor = jnp.where(jnp.real(z) < 0, negative, inside)
    factor = jnp.where(jnp.real(z) < -1, negative_remote, factor)
    factor = jnp.where(jnp.real(z) > 1, positive_remote, factor)
    return jnp.where(jnp.isfinite(z), reaction*factor, jnp.nan)


def effective_mach(mach):
    r"""C2 asymptotic continuation: M unchanged through .9, tending to 1.

    For t=(M-.9)/.1>0, M_eff=.9+.1*t/sqrt(1+t^2). The continuation
    removes the conversion singularity at large Mach without an exact
    plateau. It is a modeling choice consistent with assigning additional
    supersonic drag to profile loss; the paper does not prescribe this tail.
    It modifies the upper tenth of the subsonic Mach range as well.
    """
    mach = jnp.asarray(mach)
    width = MACH_ASYMPTOTE-MACH_TAIL_START
    t = jnp.where(jnp.real(mach) > MACH_TAIL_START, mach-MACH_TAIL_START, 0.)
    # Saturate before rescaling the tail coordinate to avoid M/.1 overflow.
    small = jnp.where(jnp.real(t) <= width, t, width)
    inv = width/jnp.where(jnp.real(t) > width, t, width)
    saturation = jnp.where(jnp.real(t) <= width, small/jnp.sqrt(width**2+small*small), 1/jnp.sqrt(1+inv*inv))
    value = jnp.where(jnp.real(mach) <= MACH_TAIL_START, mach, MACH_TAIL_START+width*saturation)
    return jnp.where(jnp.isfinite(mach) & (jnp.real(mach) >= 0), value, jnp.nan)


def _log1p_over_x(x):
    small = abs(jnp.real(x)) < 1e-4
    u = jnp.where(small, x, 0.)
    series = 1+u*(-.5+u*(1/3+u*(-.25+u*(.2-u/6))))
    safe = jnp.where(small, 1., x)
    return jnp.where(small, series, jnp.log1p(safe)/safe)


def _expm1_over_x(x):
    small = abs(jnp.real(x)) < 1e-4
    u = jnp.where(small, x, 0.)
    series = 1+u*(.5+u*(1/6+u*(1/24+u*(1/120+u/720))))
    safe = jnp.where(small, 1., x)
    return jnp.where(small, series, jnp.expm1(safe)/safe)


def pressure_loss(increment, mach, gamma=1.4, *, continue_mach=True):
    r"""KO equation (18), with an analytic zero-Mach evaluation.

    q=(gamma-1)*M_eff^2/2, n=gamma/(gamma-1), A=d/(1-d),
    Y=[(1-q*A)^(-n)-1]/[1-(1+q)^(-n)]. The limit M=0 is A.
    Logarithmic relative functions preserve that limit and complex-step
    derivatives without an arbitrary Mach cutoff. Valid gamma is 1<gamma<=5/3
    (the specified ideal-gas domain), d>=0, M>=0 and q*A<1. Without Mach
    continuation the raw gas identity can diverge; invalid points return NaN.
    With the bounded d<=.3 and M_eff<=1, the domain margin is at least 6/7
    and Y is bounded. No independent clipping of the pressure loss is used.
    """
    d, mach, gamma = jnp.broadcast_arrays(increment, mach, gamma)
    valid = (jnp.isfinite(d) & jnp.isfinite(mach) & jnp.isfinite(gamma)
             & (jnp.real(d) >= 0) & (jnp.real(d) < 1) & (jnp.real(mach) >= 0)
             & (jnp.real(gamma) > 1) & (jnp.real(gamma) <= GAMMA_MAX))
    gas = jnp.where(valid, gamma, 1.4)
    m = effective_mach(mach) if continue_mach else jnp.asarray(mach)
    m = jnp.where(valid, m, 0.)
    loss = jnp.where(valid, d, 0.)
    a = loss/(1-loss)
    q = (gas-1)*m*m/2
    valid = valid & (jnp.real(q*a) < 1)
    q = jnp.where(valid, q, 0.)
    # n*q evaluated directly remains accurate as gamma approaches one.
    nq = jnp.where(valid, gas*m*m/2, 0.)
    ln = _log1p_over_x(-q*a)
    ld = _log1p_over_x(q)
    y = a*ln/ld*_expm1_over_x(nq*a*ln)/_expm1_over_x(-nq*ld)
    return jnp.where(valid, y, jnp.nan)


def trailing_loss(thickness_opening, z, mach=0., gamma=1.4):
    """Full trailing-edge pressure loss: references, z blend, gas law.

    Pass x=t_TE/o>=0 with o>0. z=-theta_in/alpha2 is a prescribed angle
    coordinate; at alpha2=0 it must be specified as a path limit, not obtained
    by dividing zero by zero. No separate inlet Mach or Reynolds factor is
    present. Inputs outside the documented physical domain yield NaN.
    """
    return pressure_loss(energy_loss(thickness_opening, z), mach, gamma)
