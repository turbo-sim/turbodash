# Benner: secondary loss

[Previous: Moustapha: secondary incidence correction](08_moustapha_secondary.md) · [Section index](../readme.md) · [Next: Benner: incidence loss](10_benner_incidence.md)

Comparison recorded on 2026-10-17: **previous implementation (baseline)** versus **revised implementation (candidate)**. These are individual-function results; [return to the audit overview](../../readme.md) for stage-level evidence.

## Functions and dependencies

`get_secondary_loss; inlet boundary-layer correction delta/H=delta_ref*(Re_in/3e5)^(-1/7)`.

## Observed behavior

The baseline branches differ in both value (about 4.27e-5 in the manufactured component) and slope at H/c=2.

## Implemented changes

A full-factor quintic Hermite transition over H/c = 1.8–2.2 matches original endpoint values, first derivatives and second derivatives. Outer branches and the boundary-layer dependence are preserved. Dense checks confirm positive decreasing values with no transition overshoot over the tested displacement-thickness ratios.

## Baseline versus candidate

![benner_secondary_aspect_ratio: values and d/dx](../figures/benner_secondary_aspect_ratio.png)

![benner_secondary_delta_ref: values and d/dx](../figures/benner_secondary_delta_ref.png)

Solid: preserved baseline; dashed: revised candidate. Both use the same inputs, 200 base points and shared adaptive refinement. Right panels are d/dx, including logarithmic x axes. NaN value gaps remain explicit; omitted close derivative support does not delete value records.

## Quantitative checks

Shrinking one-sided probes below use h=1e-5×max(1,|join|). The full h sequence is retained; the smallest h alone is not a continuity test.

| Feature | Version | Extrapolated value jump | Right minus left slope |
|---|---|---:|---:|
| benner_secondary_AR1.8 | baseline | 1.1796e-16 | 4.2629e-07 |
| benner_secondary_AR1.8 | candidate | 1.4398e-15 | 4.262e-07 |
| benner_secondary_AR2.0 | baseline | 4.2712e-05 | -0.0063912 |
| benner_secondary_AR2.0 | candidate | -2.0123e-16 | -8.5193e-07 |
| benner_secondary_AR2.2 | baseline | 3.0878e-16 | 7.0304e-07 |
| benner_secondary_AR2.2 | candidate | -2.3245e-15 | 7.029e-07 |

The cosine-ratio floor also uses the candidate strict C2 envelope: bound 0.1, half-width 0.02. The patch is checked over displacement-thickness/full-height ratios 0, 0.01, 0.05, 0.2 and 1.

5 fundamental sweeps for this mechanism. [Value/gradient data and manifest](../data/sweep_manifest.json), [first-derivative convergence](../data/provenance/fundamental_join_intervals.csv), [second-derivative convergence](../data/provenance/fundamental_curvature_intervals.csv). C2 claims additionally use endpoint regression tests; C1 table interpolation and retained C1 branches are explicitly distinguished.

## Remaining limitations

The transition width is a numerical design choice. This removes a branch defect without validating the empirical low-AR extrapolation or changing the inlet boundary-layer model.

## All sweeps for this mechanism

Every figure contains the previous and revised curves; links below cover all saved sweeps, including those not embedded above. CSV files are gzip-compressed.

| Parameter / case | Range | Comparison figure | Numerical data |
|---|---|---|---|
| Aspect ratio H/c | 0.05 to 10 (log axis) | [Open](../figures/benner_secondary_aspect_ratio.png) | [stator](../data/sweeps/fundamental_benner_secondary_aspect_ratio__stator.csv.gz) |
| Incidence | -30 to 30 | [Open](../figures/benner_secondary_incidence.png) | [stator](../data/sweeps/fundamental_benner_secondary_incidence__stator.csv.gz) |
| Exit flow angle | 30 to 80 | [Open](../figures/benner_secondary_exit_angle.png) | [stator](../data/sweeps/fundamental_benner_secondary_exit_angle__stator.csv.gz) |
| Reference boundary-layer thickness | 0 to 0.15 | [Open](../figures/benner_secondary_delta_ref.png) | [stator](../data/sweeps/fundamental_benner_secondary_delta_ref__stator.csv.gz) |
| Stagger angle | -89.9 to 89.9 | [Open](../figures/benner_secondary_stagger.png) | [stator](../data/sweeps/fundamental_benner_secondary_stagger__stator.csv.gz) |

[All individual functions](../readme.md) · [Methods, validation and reproduction](../methodology.md)
