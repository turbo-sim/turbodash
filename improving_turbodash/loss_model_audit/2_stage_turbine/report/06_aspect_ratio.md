# Blade aspect ratio: height / true chord

[Back to stage/turbine index](../readme.md) · [Method and coefficient caveat](09_method.md)

The lines compare the preserved `benner` functions (previous) with `benner_new` using the current analytical AM references, bounded reaction/impulse extrapolation, no minimum outlet angle and the smooth upper shock continuation. These are freshly recalculated Benner totals. All samples call the full turbine API, rebuilding the real-fluid states and continuous geometry. Efficiencies are the existing driver's estimates; the pressure-to-energy coefficient normalization remains unresolved.

Figures use the repository plotting style. Beige shading marks coordinates where at least one row is outside the AM reference angle/spacing envelope; thickness and trailing-edge reference limits are also shaded in their dedicated sweeps. Shading indicates extrapolation, not failed evaluations or a validated design boundary. The individual-function report defines these reference ranges.

Height / true chord $H/c$ varies from 0.50 to 5.00, first for the stator with the rotor fixed and then for the rotor with the stator fixed. The horizontal axis is the actual loss-model aspect ratio. The axial core input named `aspect_ratio` is $H/C_x$, where $C_x=c\cos(\text{stagger})$, so the runner supplies $\frac{H/c}{\cos(\text{saved stagger})}$ and checks the resulting $H/c$ at every finite point. The exact saved geometry is included.

At fixed velocity triangles and true-chord solidity, blade height stays constant while chord, pitch, opening, thicknesses and leading-edge diameter change together. The continuous blade count and chord-based Reynolds numbers therefore change. The maximum-thickness/chord and trailing-edge/opening ratios stay fixed. This sensitivity includes the resulting profile, secondary, penetration and rotor-clearance changes; it does not isolate a single algebraic aspect-ratio factor. The other row's geometry and all loss components remain fixed and are checked numerically.

| Recuperator case | Sweep | Finite pairs | Largest efficiency change, total-static [pp] | Largest efficiency change, total-total [pp] |
|---|---|---:|---:|---:|
| 0% | stator_aspect_ratio | 206/206 | 0.0981338 | 0.104237 |
| 0% | rotor_aspect_ratio | 206/206 | 0.0904961 | 0.096124 |
| 80% | stator_aspect_ratio | 206/206 | 0.0850475 | 0.090822 |
| 80% | rotor_aspect_ratio | 206/206 | 0.0718414 | 0.0767192 |

Changes are absolute differences; positive values in the difference panels mean the revised implementation predicts higher efficiency. Gaps preserve failed/nonfinite evaluations. Review flags, warnings and raw values remain in each CSV.


## Recuperator effectiveness 0% — stator_aspect_ratio

Maximum sampled revised total-static efficiency: 87.0108% at x=2.73869. This is a sampled diagnostic maximum over the plotted interval, not an optimized or validated design.

Across the finite interval x=0.5 to 5, revised total-static efficiency goes from 86.8386% to 86.9682%; the driver row-loss fraction goes from 0.0730661 to 0.0717703. Stator Y changes from 0.0672055 to 0.0646141, and rotor Y from 0.0766156 to 0.0766156.

![Efficiency, loss coefficients and differences](../plots/eps0_stator_aspect_ratio.png)

![Separate stator and rotor loss components](../plots/eps0_stator_aspect_ratio_components.png)

[Download all points, statuses and diagnostics](../data/eps0_stator_aspect_ratio.csv)


## Recuperator effectiveness 0% — rotor_aspect_ratio

Maximum sampled revised total-static efficiency: 88.3647% at x=5. This is a sampled diagnostic maximum over the plotted interval, not an optimized or validated design.

Across the finite interval x=0.5 to 5, revised total-static efficiency goes from 85.4061% to 88.3647%; the driver row-loss fraction goes from 0.0873908 to 0.0578048. Stator Y changes from 0.0639497 to 0.0639497, and rotor Y from 0.107591 to 0.0501442.

![Efficiency, loss coefficients and differences](../plots/eps0_rotor_aspect_ratio.png)

![Separate stator and rotor loss components](../plots/eps0_rotor_aspect_ratio_components.png)

[Download all points, statuses and diagnostics](../data/eps0_rotor_aspect_ratio.csv)


## Recuperator effectiveness 80% — stator_aspect_ratio

Maximum sampled revised total-static efficiency: 86.4429% at x=5. This is a sampled diagnostic maximum over the plotted interval, not an optimized or validated design.

Across the finite interval x=0.5 to 5, revised total-static efficiency goes from 86.1195% to 86.4429%; the driver row-loss fraction goes from 0.0752246 to 0.0719902. Stator Y changes from 0.0719277 to 0.0654614, and rotor Y from 0.076222 to 0.076222.

![Efficiency, loss coefficients and differences](../plots/eps80_stator_aspect_ratio.png)

![Separate stator and rotor loss components](../plots/eps80_stator_aspect_ratio_components.png)

[Download all points, statuses and diagnostics](../data/eps80_stator_aspect_ratio.csv)


## Recuperator effectiveness 80% — rotor_aspect_ratio

Maximum sampled revised total-static efficiency: 87.7282% at x=5. This is a sampled diagnostic maximum over the plotted interval, not an optimized or validated design.

Across the finite interval x=0.5 to 5, revised total-static efficiency goes from 84.6482% to 87.7282%; the driver row-loss fraction goes from 0.0899369 to 0.059137. Stator Y changes from 0.0699002 to 0.0699002, and rotor Y from 0.106764 to 0.0469478.

![Efficiency, loss coefficients and differences](../plots/eps80_rotor_aspect_ratio.png)

![Separate stator and rotor loss components](../plots/eps80_rotor_aspect_ratio_components.png)

[Download all points, statuses and diagnostics](../data/eps80_rotor_aspect_ratio.csv)

