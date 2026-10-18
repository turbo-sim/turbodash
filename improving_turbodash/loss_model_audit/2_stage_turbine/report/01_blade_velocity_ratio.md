# Blade velocity ratio and reaction families

[Back to stage/turbine index](../readme.md) · [Method and coefficient caveat](09_method.md)

The lines compare the preserved `benner` functions (previous) with `benner_new` using the current analytical AM references, bounded reaction/impulse extrapolation, no minimum outlet angle and the smooth upper shock continuation. These are freshly recalculated Benner totals. All samples call the full turbine API, rebuilding the real-fluid states and continuous geometry. Efficiencies are the existing driver's estimates; the pressure-to-energy coefficient normalization remains unresolved.

Figures use the repository plotting style. Beige shading marks coordinates where at least one row is outside the AM reference angle/spacing envelope; thickness and trailing-edge reference limits are also shaded in their dedicated sweeps. Shading indicates extrapolation, not failed evaluations or a validated design boundary. The individual-function report defines these reference ranges.

For each saved recuperator case, blade velocity ratio varies from 0.30 to 1.00 at reaction 0.25, 0.50 and 0.75. The saved case has reaction 0.50 and blade velocity ratio 0.70. These are redesign sweeps, with radius and blade geometry recomputed, rather than fixed-hardware off-design maps.

| Recuperator case | Sweep | Finite pairs | Largest efficiency change, total-static [pp] | Largest efficiency change, total-total [pp] |
|---|---|---:|---:|---:|
| 0% | speed_r025 | 201/201 | 3.03232 | 3.95319 |
| 0% | speed_r050 | 201/201 | 0.450122 | 0.621586 |
| 0% | speed_r075 | 201/201 | 0.946755 | 1.46818 |
| 80% | speed_r025 | 201/201 | 3.24733 | 4.22549 |
| 80% | speed_r050 | 201/201 | 0.646405 | 0.896255 |
| 80% | speed_r075 | 201/201 | 55.0159 | 56.8043 |

Changes are absolute differences; positive values in the difference panels mean the revised implementation predicts higher efficiency. Gaps preserve failed/nonfinite evaluations. Review flags, warnings and raw values remain in each CSV.


## Recuperator effectiveness 0% — speed_r025

Maximum sampled revised total-static efficiency: 81.2901% at x=0.570854. This is a sampled diagnostic maximum over the plotted interval, not an optimized or validated design.

Across the finite interval x=0.3 to 1, revised total-static efficiency goes from 50.9886% to 54.5832%; the driver row-loss fraction goes from 0.242413 to 0.0704504. Stator Y changes from 0.0576247 to 0.0668219, and rotor Y from 0.344467 to 0.0691036.

![Efficiency, loss coefficients and differences](../plots/eps0_speed_r025.png)

![Separate stator and rotor loss components](../plots/eps0_speed_r025_components.png)

[Download all points, statuses and diagnostics](../data/eps0_speed_r025.csv)


## Recuperator effectiveness 0% — speed_r050

Maximum sampled revised total-static efficiency: 87.0319% at x=0.690452. This is a sampled diagnostic maximum over the plotted interval, not an optimized or validated design.

Across the finite interval x=0.3 to 1, revised total-static efficiency goes from 52.7193% to 76.5929%; the driver row-loss fraction goes from 0.180873 to 0.109054. Stator Y changes from 0.0576985 to 0.0669044, and rotor Y from 0.226169 to 0.124508.

![Efficiency, loss coefficients and differences](../plots/eps0_speed_r050.png)

![Separate stator and rotor loss components](../plots/eps0_speed_r050_components.png)

[Download all points, statuses and diagnostics](../data/eps0_speed_r050.csv)


## Recuperator effectiveness 0% — speed_r075

Maximum sampled revised total-static efficiency: 80.2420% at x=0.697487. This is a sampled diagnostic maximum over the plotted interval, not an optimized or validated design.

Across the finite interval x=0.3 to 1, revised total-static efficiency goes from 47.2827% to 61.3807%; the driver row-loss fraction goes from 0.167618 to 0.357155. Stator Y changes from 0.0577217 to 0.066921, and rotor Y from 0.1928 to 0.335363.

![Efficiency, loss coefficients and differences](../plots/eps0_speed_r075.png)

![Separate stator and rotor loss components](../plots/eps0_speed_r075_components.png)

[Download all points, statuses and diagnostics](../data/eps0_speed_r075.csv)


## Recuperator effectiveness 80% — speed_r025

Maximum sampled revised total-static efficiency: 79.8902% at x=0.567337. This is a sampled diagnostic maximum over the plotted interval, not an optimized or validated design.

Across the finite interval x=0.3 to 1, revised total-static efficiency goes from 48.8423% to 52.5438%; the driver row-loss fraction goes from 0.260777 to 0.0746485. Stator Y changes from 0.0619703 to 0.0722337, and rotor Y from 0.370597 to 0.0695779.

![Efficiency, loss coefficients and differences](../plots/eps80_speed_r025.png)

![Separate stator and rotor loss components](../plots/eps80_speed_r025_components.png)

[Download all points, statuses and diagnostics](../data/eps80_speed_r025.csv)


## Recuperator effectiveness 80% — speed_r050

Maximum sampled revised total-static efficiency: 86.2564% at x=0.690452. This is a sampled diagnostic maximum over the plotted interval, not an optimized or validated design.

Across the finite interval x=0.3 to 1, revised total-static efficiency goes from 50.0870% to 75.2510%; the driver row-loss fraction goes from 0.205368 to 0.115867. Stator Y changes from 0.0627444 to 0.0730183, and rotor Y from 0.258865 to 0.130713.

![Efficiency, loss coefficients and differences](../plots/eps80_speed_r050.png)

![Separate stator and rotor loss components](../plots/eps80_speed_r050_components.png)

[Download all points, statuses and diagnostics](../data/eps80_speed_r050.csv)


## Recuperator effectiveness 80% — speed_r075

Maximum sampled revised total-static efficiency: 79.0178% at x=0.616583. This is a sampled diagnostic maximum over the plotted interval, not an optimized or validated design.

Across the finite interval x=0.3 to 1, revised total-static efficiency goes from 45.9939% to 54.5366%; the driver row-loss fraction goes from 0.179706 to 0.423151. Stator Y changes from 0.0642282 to 0.0741032, and rotor Y from 0.205973 to 0.398669.

![Efficiency, loss coefficients and differences](../plots/eps80_speed_r075.png)

![Separate stator and rotor loss components](../plots/eps80_speed_r075_components.png)

[Download all points, statuses and diagnostics](../data/eps80_speed_r075.csv)

