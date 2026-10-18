# Rotor meridional velocity ratio

[Back to stage/turbine index](../readme.md) · [Method and coefficient caveat](09_method.md)

The lines compare the preserved `benner` functions (previous) with `benner_new` using the current analytical AM references, bounded reaction/impulse extrapolation, no minimum outlet angle and the smooth upper shock continuation. These are freshly recalculated Benner totals. All samples call the full turbine API, rebuilding the real-fluid states and continuous geometry. Efficiencies are the existing driver's estimates; the pressure-to-energy coefficient normalization remains unresolved.

Figures use the repository plotting style. Beige shading marks coordinates where at least one row is outside the AM reference angle/spacing envelope; thickness and trailing-edge reference limits are also shaded in their dedicated sweeps. Shading indicates extrapolation, not failed evaluations or a validated design boundary. The individual-function report defines these reference ranges.

The rotor meridional velocity ratio input `meridional_velocity_ratio_34` varies from 0.30 to 1.20. Other saved inputs remain fixed. Continuous blade heights, angles, chord lengths and flow states are rebuilt at each point.

| Recuperator case | Sweep | Finite pairs | Largest efficiency change, total-static [pp] | Largest efficiency change, total-total [pp] |
|---|---|---:|---:|---:|
| 0% | meridional_ratio | 201/201 | 0.0918946 | 0.0931754 |
| 80% | meridional_ratio | 201/201 | 0.0787613 | 0.0796034 |

Changes are absolute differences; positive values in the difference panels mean the revised implementation predicts higher efficiency. Gaps preserve failed/nonfinite evaluations. Review flags, warnings and raw values remain in each CSV.


## Recuperator effectiveness 0% — meridional_ratio

Maximum sampled revised total-static efficiency: 89.7412% at x=1.02362. This is a sampled diagnostic maximum over the plotted interval, not an optimized or validated design.

Across the finite interval x=0.3 to 1.2, revised total-static efficiency goes from 75.7323% to 89.6544%; the driver row-loss fraction goes from 0.0629136 to 0.0928785. Stator Y changes from 0.0639497 to 0.0639497, and rotor Y from 0.060064 to 0.118246.

![Efficiency, loss coefficients and differences](../plots/eps0_meridional_ratio.png)

![Separate stator and rotor loss components](../plots/eps0_meridional_ratio_components.png)

[Download all points, statuses and diagnostics](../data/eps0_meridional_ratio.csv)


## Recuperator effectiveness 80% — meridional_ratio

Maximum sampled revised total-static efficiency: 89.3536% at x=1.02362. This is a sampled diagnostic maximum over the plotted interval, not an optimized or validated design.

Across the finite interval x=0.3 to 1.2, revised total-static efficiency goes from 75.3778% to 89.2631%; the driver row-loss fraction goes from 0.0663931 to 0.0967907. Stator Y changes from 0.0699002 to 0.0699002, and rotor Y from 0.0610398 to 0.120075.

![Efficiency, loss coefficients and differences](../plots/eps80_meridional_ratio.png)

![Separate stator and rotor loss components](../plots/eps80_meridional_ratio_components.png)

[Download all points, statuses and diagnostics](../data/eps80_meridional_ratio.csv)

