# Benner penetration bound

The Benner loss model now uses a maximum effective penetration ratio of **0.5**
relative to full mean blade height. The transition spans raw ratios **0.45 to
0.55**, with continuous first and second derivatives. Below 0.45 the existing
correlation and loss weighting are unchanged; above 0.55 the effective ratio is
exactly 0.5.

This bounds the model's area-partition weight. It retains at least half of each
positive unweighted profile, trailing-edge and incidence coefficient. Secondary
and clearance loss formulas are unchanged. It does not predict performance of
merged secondary flows or correct extrapolated profile correlations.

For raw penetration `x`, maximum `M=0.5`, half-width `d=0.05`, lower join
`a=M-d`, upper join `b=M+d` and `t=(x-a)/(2d)`, the effective ratio is:

```text
g(x) = x                              for x <= a
       a + 2*d*(t - t**3 + 0.5*t**4)   for a < x < b
       M                              for x >= b

primary_weight = 1 - g(x)
Y_total = primary_weight*(Y_profile_2D + Y_trailing_2D + Y_incidence_2D)
          + Y_secondary + Y_clearance
```

The transition is a numerical choice, not a refitted empirical correlation.
At raw penetration 0.5, the effective value is 0.490625; it reaches 0.5 at raw
penetration 0.55. The maximum penetration bias relative to a hard 0.5 cap
is 0.009375 for the default half-width.

## API and diagnostics

Normal `core_turbine.compute_turbine_performance()` calls use the new default
automatically. No configuration changes or app callbacks are required.

```python
from turbodash.loss_models import loss_model_benner as benner

raw = benner.get_penetration_depth_raw(flow, geometry, delta_height)
used = benner.get_penetration_depth(flow, geometry, delta_height)
info = benner.get_penetration_diagnostics(flow, geometry, delta_height)
```

The loss-model dictionary option `penetration_transition_half_width` can set a
different positive half-width below 0.5 when calling the Benner evaluator or
loss dispatcher directly. It leaves the maximum fixed at 0.5. The turbine API
uses the default.

Each Benner row loss dictionary contains a separate `diagnostics` dictionary:

| Diagnostic | Meaning |
| --- | --- |
| `penetration_raw_ratio` | Uncapped correlation, relative to full mean height |
| `penetration_effective_ratio` | Ratio actually used for the area weight |
| `primary_loss_weight` | One minus effective penetration |
| `penetration_smoothing_active` | Raw value strictly inside the transition |
| `penetration_outside_model_domain` | Raw value above 0.5; bounded extrapolation |
| `profile_loss_unweighted`, `trailing_loss_unweighted`, `incidence_loss_unweighted` | Components before area weighting |
| `primary_loss_unweighted`, `primary_loss_weighted` | Corresponding component-group sums |
| `tangential_loading_parameter`, `convergence_ratio` | Existing Benner correlation inputs |
| `aspect_ratio_true_chord` | Full mean height divided by true chord |
| `inlet_displacement_thickness_height_ratio` | Displacement thickness after the Reynolds correction |

Diagnostics are **metadata**, not loss terms. Sum the named loss coefficients,
not every dictionary value. They precede any optional dispatcher tuning factors.
Existing component keys and their meaning remain unchanged.

Nonfinite or negative raw penetration, nonpositive geometric lengths or inlet
Reynolds number, negative displacement thickness and unsuitable flow angles
are rejected with `ValueError`. Saturation cannot conceal these invalid inputs.
Finite raw penetration above 0.5 is retained and flagged rather than rejected.

## Scope and validation

The true-chord/full-height convention and common primary weighting are kept.
The `F_t()` docstring now describes the implemented absolute-tangent expression;
its same-sign-angle discrepancy with the signed reference expression remains a
separate investigation. Zweifel sizing, secondary loss, coefficient conversion,
and existing piecewise profile correlations are unchanged.

The model's half-height construction is described in the original
[Benner Part I](https://doi.org/10.1115/1.2162593) and explicitly reproduced in
[Gronman et al., ETC2017-039, page 4](https://www.euroturbo.eu/paper/ETC2017-039.pdf).
The bounded extension and smoothing should be assessed separately from those
source correlations.

`tests/test_benner_penetration.py` checks bounds, unchanged low penetration,
monotonicity, join derivatives, extreme finite inputs, invalid inputs and loss
bookkeeping. The regression fixtures retain the pre-change continuous
reference results and confirm unchanged selected low-penetration designs.
