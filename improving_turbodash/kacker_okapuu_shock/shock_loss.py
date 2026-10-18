"""Standalone KO shock loss with a smooth onset and all inspection components.

No production imports or runtime data files are required. The source profile
model now calls an equivalent NumPy implementation in its own source module.
This independent JAX version is retained for the study and regression checks.
"""
import jax.numpy as jnp
import numpy as np


# Digitization of Figure 6; independently reproduced by run.py.
# Rows: stator (0), rotor (1). No runtime fitting or data-file access.
HUB_FIT = ((4.348048474772493, 0.8084516270970612),
           (5.94625398189089, 0.9468539887756984))
HUB_ROUNDING_WIDTH = 0.01
HUB_FLATTENING_RATIO = 0.3
ONSET_CENTER = 0.45
ONSET_HALF_WIDTH = 0.05
ONSET_START = 0.4
ONSET_END = 0.5
SHOCK_CONTINUATION_START = 0.25
SHOCK_ASYMPTOTE = 0.5


def effective_hub_tip_ratio(hub_tip_ratio):
    """C2 coordinate map: unchanged above 0.3, flat at the zero-ratio limit."""
    ratio = jnp.asarray(hub_tip_ratio)
    u = jnp.where(ratio < HUB_FLATTENING_RATIO, ratio / HUB_FLATTENING_RATIO, 1.)
    continued = HUB_FLATTENING_RATIO * (.5 + u**3 - .5*u**4)
    return jnp.where(ratio < HUB_FLATTENING_RATIO, continued, ratio)


def get_hub_to_mean_mach_ratio(hub_tip_ratio, cascade_type_id: int):
    """C2 Figure 6 fit; 0 <= hub/tip <= 1, stator=0, rotor=1.

    The fit is unchanged above 0.3. Below 0.3 the effective-coordinate map
    flattens the correction toward a finite limit with zero slope and
    curvature. Ratios below the chart limit 0.5 remain extrapolation.
    Zero ratio is accepted here only to inspect the mathematical limit.
    """
    ratio = jnp.asarray(hub_tip_ratio)
    row = jnp.asarray(cascade_type_id)
    amplitude = jnp.where(row == 0, HUB_FIT[0][0], HUB_FIT[1][0])
    end = jnp.where(row == 0, HUB_FIT[0][1], HUB_FIT[1][1])
    v = end - effective_hub_tip_ratio(ratio)
    # Evaluate the transition polynomial only on its bounded active argument.
    u = jnp.where((v > -HUB_ROUNDING_WIDTH) & (v < HUB_ROUNDING_WIDTH),
                  (v + HUB_ROUNDING_WIDTH) / (2 * HUB_ROUNDING_WIDTH), 0.)
    rounded = HUB_ROUNDING_WIDTH * (2 * u**3 - u**4)
    positive = jnp.where(v <= -HUB_ROUNDING_WIDTH, 0.,
                         jnp.where(v >= HUB_ROUNDING_WIDTH, v, rounded))
    value = 1 + amplitude * positive**2
    valid = (jnp.isfinite(ratio) & (ratio >= 0) & (ratio <= 1)
             & ((row == 0) | (row == 1)))
    return jnp.where(valid, value, jnp.nan)


def shock_onset(hub_mach):
    """Return (smoothstep weight, weighted KO power) for the hub Mach number.

    Both are exactly zero at and below 0.4. A quintic smoothstep centered
    at 0.45 with half-width 0.05 blends to the original power at 0.5.
    Values and first two derivatives match at both joins. The weighted
    power increases monotonically and needs no maximum or output floor.
    """
    mach = jnp.asarray(hub_mach)
    active = mach > ONSET_START
    in_transition = active & (mach < ONSET_END)
    u = jnp.where(in_transition, (mach - ONSET_START) / (ONSET_END - ONSET_START), 0.)
    smoothstep = u**3 * (10. - 15.*u + 6.*u**2)
    weight = jnp.where(mach >= ONSET_END, 1., jnp.where(active, smoothstep, 0.))
    # JAX evaluates inactive branches too. A positive dummy argument prevents
    # fractional-power NaNs and singular inactive Hessians below onset.
    power_argument = jnp.where(active, mach - ONSET_START, 1.)
    strength = jnp.where(active, weight * power_argument**1.75, 0.)
    return weight, strength


def continue_shock_loss(raw_loss):
    """C2 upper continuation of the exit-normalized shock coefficient.

    Preserve raw values through 0.25; above that join, increase smoothly
    toward 0.5 without a finite hard plateau. With d=0.25 and
    t=(raw_loss-0.25)/d, the upper branch is 0.5-d/(1+t+t*t).
    Its slope is (1+2*t)/(1+t+t*t)**2: positive, at most one, and
    equal to one at the join, where curvature is zero. These are chosen
    safeguards, not parameters fitted to the KO paper. The reciprocal
    form below avoids squaring large raw values and is autodiff safe.
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
    dict of JAX arrays
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
    Every loss/component plot can use this single evaluation; JAX JVP also
    obtains component derivatives in the same evaluation pass.

    Finite states must have positive pressures, inlet head >= 0 and exit
    head > 0. Invalid states return NaN for Y_shock; the component outputs
    remain available for diagnosis. Call validate_shock_inputs outside JIT
    for descriptive errors. No pressure epsilon, EOS or Reynolds correction
    is imposed. The caller supplies thermodynamically consistent states.
    This candidate has no production imports or runtime file dependencies.
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
    head_ratio = q1 / jnp.where(q2 > 0, q2, jnp.nan)
    raw = loss_mean * head_ratio
    value = continue_shock_loss(raw)
    effective_head_ratio = jnp.where(jnp.real(raw) > SHOCK_CONTINUATION_START,
        value / jnp.where(jnp.real(loss_mean) > 0, loss_mean, 1.), head_ratio)
    pressure_valid = (jnp.isfinite(p01) & jnp.isfinite(p1) & jnp.isfinite(p02)
                      & jnp.isfinite(p2) & (p01 > 0) & (p1 > 0) & (p02 > 0) & (p2 > 0))
    valid = (pressure_valid & jnp.isfinite(m1) & (m1 >= 0) & (ratio > 0)
             & (ratio <= 1) & jnp.isfinite(factor) & (q1 >= 0) & (q2 > 0))
    result = dict(Y_shock=jnp.where(valid, value, jnp.nan), f_hub=factor,
        hub_tip_ratio=ratio, effective_hub_tip_ratio=effective_hub_tip_ratio(ratio),
        Ma_hub=mach_hub, onset_weight=weight, onset_strength=strength,
        Y_shock_hub=loss_hub, Y_shock_mean_inlet=loss_mean,
        inlet_head=q1, exit_head=q2, head_ratio=head_ratio,
        Y_shock_raw=raw, head_ratio_effective=effective_head_ratio)
    arrays = jnp.broadcast_arrays(*result.values())
    return dict(zip(result, arrays))


def validate_shock_inputs(flow_parameters, geometry, cascade_type_id: int):
    """Eager domain validation, separate from JIT/autodiff numerical work.

    Does not impose an ideal-gas equation of state. The caller must supply
    thermodynamically consistent states. Do not call on JAX tracers.
    """
    checks = {
        "cascade_type_id must be 0 (stator) or 1 (rotor)":
            np.isin(np.asarray(cascade_type_id), (0, 1)),
        "Ma_rel_in must be finite and nonnegative":
            np.isfinite(flow_parameters["Ma_rel_in"]) & (np.asarray(flow_parameters["Ma_rel_in"]) >= 0),
        "hub_tip_ratio_in must be finite and in (0, 1]":
            np.isfinite(geometry["hub_tip_ratio_in"]) & (np.asarray(geometry["hub_tip_ratio_in"]) > 0)
            & (np.asarray(geometry["hub_tip_ratio_in"]) <= 1),
    }
    for key in ("p0_rel_in", "p_in", "p0_rel_out", "p_out"):
        value = np.asarray(flow_parameters[key])
        checks[f"{key} must be finite and positive"] = np.isfinite(value) & (value > 0)
    checks["inlet pressure head p0_rel_in - p_in must be nonnegative"] = (
        np.asarray(flow_parameters["p0_rel_in"]) >= np.asarray(flow_parameters["p_in"]))
    checks["exit pressure head p0_rel_out - p_out must be positive"] = (
        np.asarray(flow_parameters["p0_rel_out"]) > np.asarray(flow_parameters["p_out"]))
    for message, valid in checks.items():
        if not np.all(valid):
            raise ValueError(message)
