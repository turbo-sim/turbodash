"""Physical loading and separate reference-angle continuations.

Loading preserves physical turning and continues only the mean secant.
The bounded tangent remains for profile references and Benner's own
tangential-loading correlation; it is not used to form KO loading.
"""
from ._backend import jnp
from .. import math

ANGLE_START = 80.
ANGLE_ASYMPTOTE = 87.5
AXIAL_ANGLE_START = 10.
MEAN_SECANT_START = 1/math.cosd(80.)
MEAN_SECANT_ASYMPTOTE = 1/math.cosd(87.5)


def tangent(angle):
    r"""C2 odd tangent: exact through +/-80 deg, tending to +/-tan(87.5 deg).

    For h=|beta|-80>0, let t0=tan80, L=tan87.5, D=L-t0,
    a=T'(80)/D, b=a^2+T''(80)/(2D), with derivatives per degree.
    The outer branch is sign(beta)*(L-D/(1+a*h+b*h^2)).
    All coefficients are positive. Values, slopes and curvatures match the
    original tangent at +/-80; the outer coordinate is monotone and bounded.
    The 80/87.5 degree settings are explicit numerical choices. Inactive
    tangents are never evaluated near their 90-degree poles.
    """
    beta = jnp.asarray(angle)
    valid = jnp.isfinite(beta)
    beta = jnp.where(valid, beta, 0.)
    sign = jnp.where(jnp.real(beta) < 0, -1., 1.)
    magnitude = sign*beta
    outer = jnp.real(magnitude) > ANGLE_START
    h = jnp.where(outer, magnitude-ANGLE_START, 0.)
    t0, limit = math.tand(ANGLE_START), math.tand(ANGLE_ASYMPTOTE)
    derivative = (jnp.pi/180)*(1+t0*t0)
    curvature = 2*(jnp.pi/180)**2*t0*(1+t0*t0)
    d = limit-t0
    a, b = derivative/d, (derivative/d)**2+curvature/(2*d)
    inv, fraction = 1/(1+h), h/(1+h)
    tail = sign*(limit-d*inv**2/(inv**2+a*fraction*inv+b*fraction**2))
    original = math.tand(jnp.where(outer, 0., beta))
    return jnp.where(valid, jnp.where(outer, tail, original), jnp.nan)


def cosine(angle):
    """Positive cosine consistent with tangent continuation; >=cos(87.5 deg)."""
    t = tangent(angle)
    return 1/jnp.sqrt(1+t*t)


def effective_angle(angle):
    """Signed angle represented by the continued tangent; limit +/-87.5 deg."""
    return math.arctand(tangent(angle))


def effective_outlet_angle(beta_out, reference=1.):
    r"""Continue the low-angle operating trend without reversing its direction.

    Let d=sign(reference), with zero selecting +1, and b=d*beta_out.
    For b>=a=10 degrees return beta_out exactly. Below a set
    t=(a-b)/a and beta_eff=d*a/(1+t+t^2). This C2 rational coordinate
    matches value, slope and curvature of the original angle at b=a,
    remains in the fixed design direction through physical beta_out=0,
    and approaches zero only asymptotically for remote reversed trial flow.
    No minimum-angle plateau or nozzle reset is introduced.

    The reference is fixed by the cascade geometry, never inferred from
    the changing trial flow angle. Callers pass the signed metal_angle_out;
    its magnitude does not affect this coordinate. Zero or an absent metal
    angle selects +1 for standalone calls. The 10 degree
    threshold is a numerical extrapolation choice, not KO calibration.
    """
    beta, ref = jnp.broadcast_arrays(jnp.asarray(beta_out), jnp.asarray(reference))
    valid = jnp.isfinite(beta) & jnp.isfinite(ref)
    direction = jnp.where(jnp.real(ref) < 0, -1., 1.)
    b = direction*jnp.where(valid, beta, 0.)
    active = jnp.real(b) < AXIAL_ANGLE_START
    h = jnp.where(active, AXIAL_ANGLE_START-b, 0.)
    inverse = AXIAL_ANGLE_START/(AXIAL_ANGLE_START+h)
    value = direction*AXIAL_ANGLE_START*inverse**2/(1-inverse+inverse**2)
    return jnp.where(valid, jnp.where(active, value, beta), jnp.nan)


def angle_ratio(theta_in, beta_out, reference=1.):
    r"""z=-theta_in/beta_eff in the fixed design outlet direction.

    See effective_outlet_angle. Below10deg the original low-angle trend
    continues smoothly, rather than switching z sign or returning to a
    nozzle at physical beta_out=0. Reciprocal evaluation avoids dividing by
    an underflowed beta_eff for remote trial flow. At ordinary finite
    angles the ratio is exactly the original -theta/beta.
    """
    theta, beta, ref = jnp.broadcast_arrays(jnp.asarray(theta_in),
                                            jnp.asarray(beta_out), jnp.asarray(reference))
    valid = jnp.isfinite(theta) & jnp.isfinite(beta) & jnp.isfinite(ref)
    direction = jnp.where(jnp.real(ref) < 0, -1., 1.)
    b = direction*jnp.where(valid, beta, 0.)
    active = jnp.real(b) < AXIAL_ANGLE_START
    t = jnp.where(active, (AXIAL_ANGLE_START-b)/AXIAL_ANGLE_START, 0.)
    inside = -jnp.where(valid, theta, 0.)*direction/AXIAL_ANGLE_START*(1+t+t*t)
    outside = -jnp.where(valid & ~active, theta, 0.)/jnp.where(valid & ~active,beta,1.)
    return jnp.where(valid,jnp.where(active,inside,outside),jnp.nan)


def mean_secant(meridional_component, mean_tangential_component):
    r"""C2 continuation of the physical mean-vector secant only.

    For m>=0, q finite, let s=sqrt(m^2+q^2)/m, a=sec80deg,
    L=sec87.5deg, d=L-a and t=(s-a)/d. Return s through a and
    L-d/(1+t+t^2) above a. Value, slope and curvature match identity.
    The positive tail increases toward L. The thresholds are numerical
    extrapolation choices, not KO fit data. Continuing only this factor
    avoids erasing finite physical turning near zero throughflow.

    Components may share any positive scale. Reciprocal evaluation avoids
    an infinite intermediate at m=0,q!=0 (return L). The zero vector and
    negative meridional components are undefined and return NaN.
    """
    m, q = jnp.broadcast_arrays(jnp.asarray(meridional_component),
                                 jnp.asarray(mean_tangential_component))
    valid = jnp.isfinite(m) & jnp.isfinite(q) & (jnp.real(m) >= 0)
    scale = jnp.maximum(jnp.abs(jnp.real(m)), jnp.abs(jnp.real(q)))
    valid = valid & (scale > 0)
    scale = jnp.where(valid, scale, 1.)
    m, q = jnp.where(valid, m/scale, 1.), jnp.where(valid, q/scale, 0.)
    magnitude = jnp.sqrt(m*m+q*q)
    a, limit = MEAN_SECANT_START, MEAN_SECANT_ASYMPTOTE
    outer = jnp.real(magnitude) > a*jnp.real(m)
    span = limit-a
    inverse = span*m/(magnitude+(limit-2*a)*m)
    tail = limit-span*inverse**2/(1-inverse+inverse**2)
    original = magnitude/jnp.where(outer, 1., m)
    return jnp.where(valid, jnp.where(outer, tail, original), jnp.nan)


def physical_cosine_ratio(beta_out, theta_in):
    """Physical cos(beta_out)/cos(theta_in), without angle saturation.

    Signed angles must lie strictly within (-90,+90) degrees. For zero
    incidence and equal meridional velocities the ratio is w_in/w_out,
    retaining its physical finite limit near grazing flow. Exactly grazing
    and reversed-flow angles are outside this correlation's domain (NaN).
    """
    bo, theta = jnp.broadcast_arrays(jnp.asarray(beta_out), jnp.asarray(theta_in))
    valid = (jnp.isfinite(bo) & jnp.isfinite(theta)
             & (jnp.abs(jnp.real(bo)) < 90.) & (jnp.abs(jnp.real(theta)) < 90.))
    value = math.cosd(jnp.where(valid, bo, 0.))/math.cosd(jnp.where(valid, theta, 0.))
    return jnp.where(valid, value, jnp.nan)


def loading(beta_in, beta_out):
    r"""KO loading preserving physical turning, with a mean-secant tail.

    For signed row-frame angles strictly within (-90,+90) degrees,

    .. math::

        Z=4\left[\frac{\sin(\beta_{in}-\beta_{out})}
                           {\cos\beta_{in}}\right]^2 S(\sec\beta_m),
        \qquad \tan\beta_m=\frac{\tan\beta_{in}+\tan\beta_{out}}2.

    S is mean_secant: identity through sec80deg, C2 rational tail toward
    sec87.5deg. No individual tangent, turning factor or final Z is capped.
    The original correlation is retained whenever the mean angle is
    within +/-80deg, even if either individual angle is larger.

    At equal meridional velocities, the first factor is exactly
    4*((w_theta,in-w_theta,out)/w_out)^2. It stays finite and nonzero along
    the high-reaction turbine family. Separately saturating two tangents
    would erase that turning when both angles approach -90deg. Only the
    mean secant diverges along that family; its continuation gives finite
    limiting loading. These thresholds specify numerical extrapolation.

    The angle form retains the original KO definition at unequal
    meridional velocities too; its first factor then need not equal the
    physical velocity difference. Scaled sine/cosine products avoid both
    tangent poles. Equal angles give exactly zero and angle reflection
    preserves loading. Exactly grazing or reversed-flow angles return NaN:
    zero throughflow has no unique angle-only loading endpoint, and no
    periodic or reversed-flow continuation is invented.
    """
    bi, bo = jnp.broadcast_arrays(jnp.asarray(beta_in), jnp.asarray(beta_out))
    valid = (jnp.isfinite(bi) & jnp.isfinite(bo)
             & (jnp.abs(jnp.real(bi)) < 90.) & (jnp.abs(jnp.real(bo)) < 90.))
    bi, bo = jnp.where(valid, bi, 0.), jnp.where(valid, bo, 0.)
    ci, co = math.cosd(bi), math.cosd(bo)
    turning = 4*(math.sind(bi-bo)/ci)**2
    mean = mean_secant(ci*co, .5*math.sind(bi+bo))
    return jnp.where(valid, turning*mean, jnp.nan)
