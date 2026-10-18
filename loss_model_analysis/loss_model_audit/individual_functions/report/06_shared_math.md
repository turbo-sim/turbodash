# Shared numerical helpers

[Previous: Current coefficient conversions](05_conversions.md) · [Section index](../readme.md) · [Next: Moustapha: incidence loss](07_moustapha_incidence.md)

These plots evaluate the current production implementation. [Audit overview](../../readme.md).

## Functions and dependencies

`smooth_maximum, smooth_minimum, smooth_abs (single shared turbodash.math module)`.

## Observed behavior

The shared Boltzmann min/max are smooth weighted averages rather than strict bounds, and they can shift reference values. Changing them globally would affect the other correlations that use them.

## Implemented formulation

`turbodash/math.py` contains the shared mathematical utilities. The production loss models use `turbodash/loss_models/_numerics.py` for a compact C2 positive-part envelope, strict min/max, quintic Hermite joins and C1 shape-preserving cubic tables. Each caller documents its transition parameters and purpose.

## Current values and derivatives

[smooth_bounds: values and d/dx (PDF)](../figures/smooth_bounds.pdf)

[smooth_absolute: values and d/dx (PDF)](../figures/smooth_absolute.pdf)

## Quantitative checks

2 fundamental sweeps for this mechanism. [Value/gradient data and manifest](../data/sweep_manifest.json), [first-derivative convergence](../data/provenance/fundamental_join_intervals.csv), [second-derivative convergence](../data/provenance/fundamental_curvature_intervals.csv). C2 claims additionally use endpoint regression tests; C1 table interpolation and retained C1 branches are explicitly distinguished.

## Remaining limitations

Array operations retain the limited backend seam, with no new JIT/AD guarantee. The shared `smooth_abs` variants and their domains remain unchanged. Numerical widths are documented in each production function rather than assigned a literature justification.

## All sweeps for this mechanism

Every figure contains current-source values and derivatives; links below cover all saved sweeps, including those not embedded above. CSV files are gzip-compressed.

| Parameter / case | Range | Comparison figure | Numerical data |
|---|---|---|---|
| Helper argument · smooth bounds | -0.5 to 0.8 | [Open](../figures/smooth_bounds.pdf) | [prescribed reference](../data/sweeps/fundamental_smooth_bounds.csv.gz) |
| Helper argument · smooth absolute | -0.02 to 0.02 | [Open](../figures/smooth_absolute.pdf) | [prescribed reference](../data/sweeps/fundamental_smooth_absolute.csv.gz) |

[All individual functions](../readme.md) · [Methods, validation and reproduction](../methodology.md)
