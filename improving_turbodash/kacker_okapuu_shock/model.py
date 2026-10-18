"""Digitization-fit helpers and prescribed perfect-gas states for the study.

The fit uses the chart interval, where the low-radius continuation is inactive.
All shock-loss evaluations are performed by shock_loss.get_shock_loss.
"""
import numpy as np
import jax.numpy as jnp

ROUNDING_WIDTH = .01


def positive_part(value):
    value=np.asarray(value)
    q=np.where(np.abs(np.real(value))<ROUNDING_WIDTH,(value+ROUNDING_WIDTH)/(2*ROUNDING_WIDTH),0.)
    middle=ROUNDING_WIDTH*(2*q**3-q**4)
    p=np.where(np.real(value)<=-ROUNDING_WIDTH,0,np.where(np.real(value)>=ROUNDING_WIDTH,value,middle))
    dp=np.where(np.real(value)<=-ROUNDING_WIDTH,0,np.where(np.real(value)>=ROUNDING_WIDTH,1,3*q*q-2*q**3))
    ddp=np.where(np.abs(np.real(value))<ROUNDING_WIDTH,3*q*(1-q)/ROUNDING_WIDTH,0)
    return p,dp,ddp


def hub_factor(ratio, amplitude, end_ratio):
    p,dp,ddp=positive_part(end_ratio-np.asarray(ratio))
    return 1+amplitude*p*p


def hub_derivatives(ratio, amplitude, end_ratio):
    p,dp,ddp=positive_part(end_ratio-np.asarray(ratio))
    return -2*amplitude*p*dp,2*amplitude*(dp*dp+p*ddp)


def head_fraction(mach,gamma=1.4):
    """(p0-p)/p0 for a calorically perfect gas, stable at small Mach."""
    mach=np.asarray(mach)
    return -np.expm1(-gamma/(gamma-1)*np.log1p((gamma-1)/2*mach*mach))


def head_fraction_derivative(mach,gamma=1.4):
    mach=np.asarray(mach)
    return gamma*mach*(1+(gamma-1)/2*mach*mach)**(-gamma/(gamma-1)-1)


def ideal_flow(mach_in, mach_out=.6, p0=2e5, gamma=1.4):
    """Equal row-frame total pressures; static pressures follow ideal-gas Mach.

    This defines a sensitivity experiment, not a solved rotor or stator state.
    JAX operations allow differentiation through the prescribed pressures.
    """
    def fraction(mach):
        return -jnp.expm1(-gamma/(gamma-1)*jnp.log1p((gamma-1)/2*jnp.asarray(mach)**2))
    return dict(Ma_rel_in=mach_in, Ma_rel_out=mach_out, p0_rel_in=p0, p0_rel_out=p0,
                p_in=p0*(1-fraction(mach_in)), p_out=p0*(1-fraction(mach_out)))
