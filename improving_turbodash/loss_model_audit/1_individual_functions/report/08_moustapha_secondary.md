# Moustapha: secondary incidence correction

[Previous: Moustapha: incidence loss](07_moustapha_incidence.md) · [Section index](../readme.md) · [Next: Benner: secondary loss](09_benner_secondary.md)

Comparison recorded on 2026-10-17: **previous implementation (baseline)** versus **revised implementation (candidate)**. These are individual-function results; [return to the audit overview](../../readme.md) for stage-level evidence.

## Functions and dependencies

`get_secondary_loss_correction_factor × KO.get_secondary_loss`.

## Observed behavior

The correction factor's extra positive-branch terms start quadratically, so its chi 0 join already has a continuous first derivative. Its second derivative changes there.

## Implemented changes

The correction factor is unchanged. The separate KO secondary component on which the complete Moustapha model depends receives the aspect-ratio repair described in its report. These fundamental plots show the factor alone.

## Baseline versus candidate

![moustapha_secondary_incidence: values and d/dx](../figures/moustapha_secondary_incidence.png)

![moustapha_secondary_leading_edge_ratio: values and d/dx](../figures/moustapha_secondary_leading_edge_ratio.png)

Solid: preserved baseline; dashed: revised candidate. Both use the same inputs, 200 base points and shared adaptive refinement. Right panels are d/dx, including logarithmic x axes. NaN value gaps remain explicit; omitted close derivative support does not delete value records.

## Quantitative checks

Shrinking one-sided probes below use h=1e-5×max(1,|join|). The full h sequence is retained; the smallest h alone is not a continuity test.

| Feature | Version | Extrapolated value jump | Right minus left slope |
|---|---|---:|---:|
| moustapha_secondary_reference | baseline | -1.1805e-12 | 1.8819e-07 |
| moustapha_secondary_reference | candidate | -1.1805e-12 | 1.8819e-07 |

2 fundamental sweeps for this mechanism. [Value/gradient data and manifest](../data/sweep_manifest.json), [first-derivative convergence](../data/provenance/fundamental_join_intervals.csv), [second-derivative convergence](../data/provenance/fundamental_curvature_intervals.csv). C2 claims additionally use endpoint regression tests; C1 table interpolation and retained C1 branches are explicitly distinguished.

## Remaining limitations

High-incidence growth and absolute-angle conventions remain unresolved. No saturation is introduced without physical evidence; no global C2 claim is made.

## All sweeps for this mechanism

Every figure contains the previous and revised curves; links below cover all saved sweeps, including those not embedded above. CSV files are gzip-compressed.

| Parameter / case | Range | Comparison figure | Numerical data |
|---|---|---|---|
| Incidence | -30 to 30 | [Open](../figures/moustapha_secondary_incidence.png) | [stator](../data/sweeps/fundamental_moustapha_secondary_incidence__stator.csv.gz) |
| Leading-edge ratio | 0.001 to 0.2 (log axis) | [Open](../figures/moustapha_secondary_leading_edge_ratio.png) | [stator, inlet offset=+10 deg](../data/sweeps/fundamental_moustapha_secondary_leading_edge_ratio__stator.csv.gz) |

[All individual functions](../readme.md) · [Methods, validation and reproduction](../methodology.md)

The complete candidate model also inherits the new analytical AM references through KO profile loss. The mechanism reviewed here is unchanged by that replacement. See the [profile integration and model-propagation review](01_ko_profile.md#shared-profile-loss-in-moustapha-and-benner) for isolated profile/total differences and independent regression checks.
