# Blade solidity

[Back to stage/turbine index](../readme.md) · [Method and coefficient caveat](09_method.md)

The lines compare the preserved `benner` functions (previous) with `benner_new` using the current analytical AM references, bounded reaction/impulse extrapolation, no minimum outlet angle and the smooth upper shock continuation. These are freshly recalculated Benner totals. All samples call the full turbine API, rebuilding the real-fluid states and continuous geometry. Efficiencies are the existing driver's estimates; the pressure-to-energy coefficient normalization remains unresolved.

Figures use the repository plotting style. Beige shading marks coordinates where at least one row is outside the AM reference angle/spacing envelope; thickness and trailing-edge reference limits are also shaded in their dedicated sweeps. Shading indicates extrapolation, not failed evaluations or a validated design boundary. The individual-function report defines these reference ranges.

The solidity sweep multiplies both saved row values of true chord/pitch by 0.60 to 1.60, preserving their ratio; it does not independently optimize each row. All other saved design inputs stay fixed.

| Recuperator case | Sweep | Finite pairs | Largest efficiency change, total-static [pp] | Largest efficiency change, total-total [pp] |
|---|---|---:|---:|---:|
| 0% | solidity | 201/201 | 2.32541 | 2.47002 |
| 80% | solidity | 201/201 | 2.47583 | 2.64393 |

Changes are absolute differences; positive values in the difference panels mean the revised implementation predicts higher efficiency. Gaps preserve failed/nonfinite evaluations. Review flags, warnings and raw values remain in each CSV.


## Recuperator effectiveness 0% — solidity

Maximum sampled revised total-static efficiency: 87.8489% at x=1.54472. This is a sampled diagnostic maximum over the plotted interval, not an optimized or validated design.

Across the finite interval x=0.6 to 1.6, revised total-static efficiency goes from 81.5366% to 87.8453%; the driver row-loss fraction goes from 0.126086 to 0.0629987. Stator Y changes from 0.127448 to 0.0518093, and rotor Y from 0.121069 to 0.0720174.

![Efficiency, loss coefficients and differences](../plots/eps0_solidity.png)

![Separate stator and rotor loss components](../plots/eps0_solidity_components.png)

[Download all points, statuses and diagnostics](../data/eps0_solidity.csv)


## Recuperator effectiveness 80% — solidity

Maximum sampled revised total-static efficiency: 87.0925% at x=1.5196. This is a sampled diagnostic maximum over the plotted interval, not an optimized or validated design.

Across the finite interval x=0.6 to 1.6, revised total-static efficiency goes from 80.3326% to 87.0838%; the driver row-loss fraction goes from 0.133093 to 0.0655809. Stator Y changes from 0.142863 to 0.0559344, and rotor Y from 0.119701 to 0.0730291.

![Efficiency, loss coefficients and differences](../plots/eps80_solidity.png)

![Separate stator and rotor loss components](../plots/eps80_solidity_components.png)

[Download all points, statuses and diagnostics](../data/eps80_solidity.csv)

