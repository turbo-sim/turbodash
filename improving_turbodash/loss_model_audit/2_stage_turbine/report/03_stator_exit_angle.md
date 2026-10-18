# Stator exit angle

[Back to stage/turbine index](../readme.md) · [Method and coefficient caveat](09_method.md)

The lines compare the preserved `benner` functions (previous) with `benner_new` using the current analytical AM references, bounded reaction/impulse extrapolation, no minimum outlet angle and the smooth upper shock continuation. These are freshly recalculated Benner totals. All samples call the full turbine API, rebuilding the real-fluid states and continuous geometry. Efficiencies are the existing driver's estimates; the pressure-to-energy coefficient normalization remains unresolved.

Figures use the repository plotting style. Beige shading marks coordinates where at least one row is outside the AM reference angle/spacing envelope; thickness and trailing-edge reference limits are also shaded in their dedicated sweeps. Shading indicates extrapolation, not failed evaluations or a validated design boundary. The individual-function report defines these reference ranges.

The stator exit angle varies from 30 to 80 degrees. All other saved design inputs stay fixed; flow states and continuous geometry are recomputed at each angle.

| Recuperator case | Sweep | Finite pairs | Largest efficiency change, total-static [pp] | Largest efficiency change, total-total [pp] |
|---|---|---:|---:|---:|
| 0% | exit_angle | 108/200 | 4.95598 | 59.4446 |
| 80% | exit_angle | 98/200 | 2.21672 | 62.304 |

Changes are absolute differences; positive values in the difference panels mean the revised implementation predicts higher efficiency. Gaps preserve failed/nonfinite evaluations. Review flags, warnings and raw values remain in each CSV.


## Recuperator effectiveness 0% — exit_angle

Maximum sampled revised total-static efficiency: 87.0014% at x=80. This is a sampled diagnostic maximum over the plotted interval, not an optimized or validated design.

Across the finite interval x=53.1156 to 80, revised total-static efficiency goes from -25.9391% to 87.0014%; the driver row-loss fraction goes from 0.157279 to 0.071438. Stator Y changes from 0.0457719 to 0.0639497, and rotor Y from 0.192625 to 0.0766156.

184 evaluations across the two versions failed or were nonfinite, spanning requested x=30 to 52.8643. The curves retain those gaps. First recorded error: `AssertionError: Degree of reaction check failed: Computed = nan, Target = 5.000000e-01`.

![Efficiency, loss coefficients and differences](../plots/eps0_exit_angle.png)

The following display zoom shows the 0–100% efficiency range. Extreme and negative outputs remain visible in the full-range figure above and remain unchanged in the CSV; this view does not make the underlying designs physically admissible.

![Efficiency display zoom](../plots/eps0_exit_angle_efficiency_detail.png)

![Separate stator and rotor loss components](../plots/eps0_exit_angle_components.png)

[Download all points, statuses and diagnostics](../data/eps0_exit_angle.csv)


## Recuperator effectiveness 80% — exit_angle

Maximum sampled revised total-static efficiency: 86.2209% at x=80. This is a sampled diagnostic maximum over the plotted interval, not an optimized or validated design.

Across the finite interval x=55.6281 to 80, revised total-static efficiency goes from -20.8305% to 86.2209%; the driver row-loss fraction goes from 0.144823 to 0.0742105. Stator Y changes from 0.0470062 to 0.0699002, and rotor Y from 0.180591 to 0.076222.

204 evaluations across the two versions failed or were nonfinite, spanning requested x=30 to 55.3769. The curves retain those gaps. First recorded error: `AssertionError: Degree of reaction check failed: Computed = nan, Target = 5.000000e-01`.

![Efficiency, loss coefficients and differences](../plots/eps80_exit_angle.png)

The following display zoom shows the 0–100% efficiency range. Extreme and negative outputs remain visible in the full-range figure above and remain unchanged in the CSV; this view does not make the underlying designs physically admissible.

![Efficiency display zoom](../plots/eps80_exit_angle_efficiency_detail.png)

![Separate stator and rotor loss components](../plots/eps80_exit_angle_components.png)

[Download all points, statuses and diagnostics](../data/eps80_exit_angle.csv)

