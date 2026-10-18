# Trailing-edge thickness / opening

[Back to stage/turbine index](../readme.md) · [Method and coefficient caveat](09_method.md)

The lines compare the preserved `benner` functions (previous) with `benner_new` using the current analytical AM references, bounded reaction/impulse extrapolation, no minimum outlet angle and the smooth upper shock continuation. These are freshly recalculated Benner totals. All samples call the full turbine API, rebuilding the real-fluid states and continuous geometry. Efficiencies are the existing driver's estimates; the pressure-to-energy coefficient normalization remains unresolved.

Figures use the repository plotting style. Beige shading marks coordinates where at least one row is outside the AM reference angle/spacing envelope; thickness and trailing-edge reference limits are also shaded in their dedicated sweeps. Shading indicates extrapolation, not failed evaluations or a validated design boundary. The individual-function report defines these reference ranges.

Trailing-edge thickness / opening $t_{\mathrm{TE}}/o$ varies from 0 to 0.60, independently for the stator and rotor. Opening, chord, row solidity, flow states and all other geometry inputs stay at their saved values. Zero thickness is evaluated explicitly: the revised trailing term is exactly zero, while the previous implementation retains its historical smoothing offset. The target is supplied directly to `trailing_edge_thickness_to_opening`, and the actual output ratio is checked at every finite point. The other row's geometry and all loss components remain fixed and are checked numerically.

This geometry sensitivity changes only the selected row's trailing-edge thickness and trailing loss term; all other loss contributions are checked against the saved reference. It includes the full turbine efficiency response to the changed term.

The source trailing-edge table ends at $t_{\mathrm{TE}}/o=0.4$, marked by a vertical line with the region beyond shaded. Exact table coordinates 0.2 and 0.4 are retained. The revised implementation continues the table as a constant above 0.4; the previous implementation uses a smooth minimum before linear interpolation. The resulting plateau over 0.4 to 0.6 shows endpoint continuation, not calibration or physical validation of thick trailing edges.

| Recuperator case | Sweep | Finite pairs | Largest efficiency change, total-static [pp] | Largest efficiency change, total-total [pp] |
|---|---|---:|---:|---:|
| 0% | stator_trailing_edge_ratio | 203/203 | 1.99014 | 2.1139 |
| 0% | rotor_trailing_edge_ratio | 203/203 | 1.91413 | 2.03317 |
| 80% | stator_trailing_edge_ratio | 203/203 | 2.00696 | 2.14322 |
| 80% | rotor_trailing_edge_ratio | 203/203 | 1.93044 | 2.06151 |

Changes are absolute differences; positive values in the difference panels mean the revised implementation predicts higher efficiency. Gaps preserve failed/nonfinite evaluations. Review flags, warnings and raw values remain in each CSV.


## Recuperator effectiveness 0% — stator_trailing_edge_ratio

Maximum sampled revised total-static efficiency: 87.4734% at x=0. This is a sampled diagnostic maximum over the plotted interval, not an optimized or validated design.

Across the finite interval x=0 to 0.6, revised total-static efficiency goes from 87.4734% to 79.1272%; the driver row-loss fraction goes from 0.0667184 to 0.15018. Stator Y changes from 0.0545119 to 0.22141, and rotor Y from 0.0766156 to 0.0766156.

![Efficiency, loss coefficients and differences](../plots/eps0_stator_trailing_edge_ratio.png)

![Separate stator and rotor loss components](../plots/eps0_stator_trailing_edge_ratio_components.png)

[Download all points, statuses and diagnostics](../data/eps0_stator_trailing_edge_ratio.csv)


## Recuperator effectiveness 0% — rotor_trailing_edge_ratio

Maximum sampled revised total-static efficiency: 87.4559% at x=0. This is a sampled diagnostic maximum over the plotted interval, not an optimized or validated design.

Across the finite interval x=0 to 0.6, revised total-static efficiency goes from 87.4559% to 79.4171%; the driver row-loss fraction goes from 0.0668927 to 0.147281. Stator Y changes from 0.0639497 to 0.0639497, and rotor Y from 0.0677902 to 0.223879.

![Efficiency, loss coefficients and differences](../plots/eps0_rotor_trailing_edge_ratio.png)

![Separate stator and rotor loss components](../plots/eps0_rotor_trailing_edge_ratio_components.png)

[Download all points, statuses and diagnostics](../data/eps0_rotor_trailing_edge_ratio.csv)


## Recuperator effectiveness 80% — stator_trailing_edge_ratio

Maximum sampled revised total-static efficiency: 86.6925% at x=0. This is a sampled diagnostic maximum over the plotted interval, not an optimized or validated design.

Across the finite interval x=0 to 0.6, revised total-static efficiency goes from 86.6925% to 78.3530%; the driver row-loss fraction goes from 0.0694946 to 0.152889. Stator Y changes from 0.0604721 to 0.227199, and rotor Y from 0.076222 to 0.076222.

![Efficiency, loss coefficients and differences](../plots/eps80_stator_trailing_edge_ratio.png)

![Separate stator and rotor loss components](../plots/eps80_stator_trailing_edge_ratio_components.png)

[Download all points, statuses and diagnostics](../data/eps80_stator_trailing_edge_ratio.csv)


## Recuperator effectiveness 80% — rotor_trailing_edge_ratio

Maximum sampled revised total-static efficiency: 86.6749% at x=0. This is a sampled diagnostic maximum over the plotted interval, not an optimized or validated design.

Across the finite interval x=0 to 0.6, revised total-static efficiency goes from 86.6749% to 78.6449%; the driver row-loss fraction goes from 0.0696702 to 0.149971. Stator Y changes from 0.0699002 to 0.0699002, and rotor Y from 0.0674043 to 0.223356.

![Efficiency, loss coefficients and differences](../plots/eps80_rotor_trailing_edge_ratio.png)

![Separate stator and rotor loss components](../plots/eps80_rotor_trailing_edge_ratio_components.png)

[Download all points, statuses and diagnostics](../data/eps80_rotor_trailing_edge_ratio.csv)

