# Maximum blade thickness / true chord

[Back to stage/turbine index](../readme.md) · [Method and coefficient caveat](09_method.md)

The lines compare the preserved `benner` functions (previous) with `benner_new` using the current analytical AM references, bounded reaction/impulse extrapolation, no minimum outlet angle and the smooth upper shock continuation. These are freshly recalculated Benner totals. All samples call the full turbine API, rebuilding the real-fluid states and continuous geometry. Efficiencies are the existing driver's estimates; the pressure-to-energy coefficient normalization remains unresolved.

Figures use the repository plotting style. Beige shading marks coordinates where at least one row is outside the AM reference angle/spacing envelope; thickness and trailing-edge reference limits are also shaded in their dedicated sweeps. Shading indicates extrapolation, not failed evaluations or a validated design boundary. The individual-function report defines these reference ranges.

Maximum blade thickness / true chord $t_{\max}/c$ varies from 0.05 to 0.40, independently for the stator and rotor. The core input named `maximum_thickness_to_chord` currently uses meridional chord `Cx`, so the runner supplies $\frac{t_{\max}/c}{\cos(\text{saved stagger})}$ and checks the actual true-chord ratio at every finite point. The exact saved geometry is included.

The saved leading-edge-radius / maximum-thickness ratio stays fixed, so maximum thickness and leading-edge radius/diameter change together. Chord, pitch, opening, height, Reynolds numbers and the other row remain fixed. These are sensitivities of the complete row geometry specification, including any leading-edge dependence, rather than independent changes to maximum thickness alone. Both stator and rotor responses are flat in both saved cases because their KO thickness factor is exactly one at nonpositive `z = -theta_in/beta_out`. The new factor blends smoothly into the original power law over `0 < z < 0.05`. The design metal and flow inlet angles coincide, retaining zero-incidence reference behavior. The physical thicknesses and leading-edge diameters do change; these saved operating conditions give no loss or efficiency sensitivity to that change.

| Recuperator case | Sweep | Finite pairs | Largest efficiency change, total-static [pp] | Largest efficiency change, total-total [pp] |
|---|---|---:|---:|---:|
| 0% | stator_maximum_thickness_ratio | 201/201 | 0.0665978 | 0.0707394 |
| 0% | rotor_maximum_thickness_ratio | 201/201 | 0.0665978 | 0.0707394 |
| 80% | stator_maximum_thickness_ratio | 201/201 | 0.0481251 | 0.0513927 |
| 80% | rotor_maximum_thickness_ratio | 201/201 | 0.0481251 | 0.0513927 |

Changes are absolute differences; positive values in the difference panels mean the revised implementation predicts higher efficiency. Gaps preserve failed/nonfinite evaluations. Review flags, warnings and raw values remain in each CSV.


## Recuperator effectiveness 0% — stator_maximum_thickness_ratio

Maximum sampled revised total-static efficiency: 87.0014% at x=0.05. This is a sampled diagnostic maximum over the plotted interval, not an optimized or validated design.

Across the finite interval x=0.05 to 0.4, revised total-static efficiency goes from 87.0014% to 87.0014%; the driver row-loss fraction goes from 0.071438 to 0.071438. Stator Y changes from 0.0639497 to 0.0639497, and rotor Y from 0.0766156 to 0.0766156.

![Efficiency, loss coefficients and differences](../plots/eps0_stator_maximum_thickness_ratio.png)

![Separate stator and rotor loss components](../plots/eps0_stator_maximum_thickness_ratio_components.png)

[Download all points, statuses and diagnostics](../data/eps0_stator_maximum_thickness_ratio.csv)


## Recuperator effectiveness 0% — rotor_maximum_thickness_ratio

Maximum sampled revised total-static efficiency: 87.0014% at x=0.05. This is a sampled diagnostic maximum over the plotted interval, not an optimized or validated design.

Across the finite interval x=0.05 to 0.4, revised total-static efficiency goes from 87.0014% to 87.0014%; the driver row-loss fraction goes from 0.071438 to 0.071438. Stator Y changes from 0.0639497 to 0.0639497, and rotor Y from 0.0766156 to 0.0766156.

![Efficiency, loss coefficients and differences](../plots/eps0_rotor_maximum_thickness_ratio.png)

![Separate stator and rotor loss components](../plots/eps0_rotor_maximum_thickness_ratio_components.png)

[Download all points, statuses and diagnostics](../data/eps0_rotor_maximum_thickness_ratio.csv)


## Recuperator effectiveness 80% — stator_maximum_thickness_ratio

Maximum sampled revised total-static efficiency: 86.2209% at x=0.05. This is a sampled diagnostic maximum over the plotted interval, not an optimized or validated design.

Across the finite interval x=0.05 to 0.4, revised total-static efficiency goes from 86.2209% to 86.2209%; the driver row-loss fraction goes from 0.0742105 to 0.0742105. Stator Y changes from 0.0699002 to 0.0699002, and rotor Y from 0.076222 to 0.076222.

![Efficiency, loss coefficients and differences](../plots/eps80_stator_maximum_thickness_ratio.png)

![Separate stator and rotor loss components](../plots/eps80_stator_maximum_thickness_ratio_components.png)

[Download all points, statuses and diagnostics](../data/eps80_stator_maximum_thickness_ratio.csv)


## Recuperator effectiveness 80% — rotor_maximum_thickness_ratio

Maximum sampled revised total-static efficiency: 86.2209% at x=0.05. This is a sampled diagnostic maximum over the plotted interval, not an optimized or validated design.

Across the finite interval x=0.05 to 0.4, revised total-static efficiency goes from 86.2209% to 86.2209%; the driver row-loss fraction goes from 0.0742105 to 0.0742105. Stator Y changes from 0.0699002 to 0.0699002, and rotor Y from 0.076222 to 0.076222.

![Efficiency, loss coefficients and differences](../plots/eps80_rotor_maximum_thickness_ratio.png)

![Separate stator and rotor loss components](../plots/eps80_rotor_maximum_thickness_ratio_components.png)

[Download all points, statuses and diagnostics](../data/eps80_rotor_maximum_thickness_ratio.csv)

