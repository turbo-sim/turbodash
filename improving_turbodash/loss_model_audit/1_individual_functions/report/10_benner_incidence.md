# Benner: incidence loss

[Previous: Benner: secondary loss](09_benner_secondary.md) · [Section index](../readme.md) · [Next: Benner: penetration and primary-loss weighting](11_benner_penetration.md)

Comparison recorded on 2026-10-17: **previous implementation (baseline)** versus **revised implementation (candidate)**. These are individual-function results; [return to the audit overview](../../readme.md) for stage-level evidence.

## Functions and dependencies

`get_incidence_parameter → get_incidence_profile_loss_increment → get_incidence_loss → coefficient conversion`.

## Observed behavior

The raw branches at chi = 0 meet in value but have slopes about -8.720e-4 and -6.149e-5. The original positive-side polynomial has a small negative pocket; the former limiter also introduced a negative offset at exactly zero.

## Implemented changes

A quintic Hermite patch on chi in [-0.1, 0] matches the negative polynomial at -.1 and the unchanged positive polynomial's value, slope and curvature at 0. It is C2, decreases to exactly 0 and introduces no new extrema or negative values inside the patch. The original positive-side negative pocket remains. The upper limiter is now exact away from raw loss 0.5 ± 0.02 and C2 across that band; raw and limited values remain separate. Candidate conversions also expose the unrestricted identity.

## Baseline versus candidate

Beige shading marks the continuation beyond the polynomial's upper join at $\chi=5$.

![incidence_chi_detail: values and d/dx](../figures/incidence_chi_detail.png)

![incidence_chi: values and d/dx](../figures/incidence_chi.png)

![benner_incidence_incidence: values and d/dx](../figures/benner_incidence_incidence.png)

![benner_incidence_Ma_rel_out: values and d/dx](../figures/benner_incidence_ma_rel_out.png)

Solid: preserved baseline; dashed: revised candidate. Both use the same inputs, 200 base points and shared adaptive refinement. Right panels are d/dx, including logarithmic x axes. NaN value gaps remain explicit; omitted close derivative support does not delete value records.

## Quantitative checks

Shrinking one-sided probes below use h=1e-5×max(1,|join|). The full h sequence is retained; the smallest h alone is not a continuity test.

| Feature | Version | Extrapolated value jump | Right minus left slope |
|---|---|---:|---:|
| incidence_chi_-0.1 | baseline | 5.421e-20 | 8.148e-09 |
| incidence_chi_-0.1 | candidate | 1.8731e-15 | 7.9295e-09 |
| incidence_chi_0.0 | baseline | -2.3824e-13 | 0.00081055 |
| incidence_chi_0.0 | candidate | 2.7039e-15 | 7.9935e-08 |
| incidence_chi_5.0 | baseline | 2.1006e-11 | 6.3015e-07 |
| incidence_chi_5.0 | candidate | 2.1006e-11 | 6.3015e-07 |

Reference check `incidence_at_chi0`: baseline **-1.49065849e-07**, candidate **0**.

12 fundamental sweeps for this mechanism. [Value/gradient data and manifest](../data/sweep_manifest.json), [first-derivative convergence](../data/provenance/fundamental_join_intervals.csv), [second-derivative convergence](../data/provenance/fundamental_curvature_intervals.csv). C2 claims additionally use endpoint regression tests; C1 table interpolation and retained C1 branches are explicitly distinguished.

## Remaining limitations

This preserves the existing reference convention; it does not declare design incidence to be the loss minimum. Signed negative increments are algebraic corrections, not negative total dissipation. The extrapolation at chi = 5 remains C1, not C2. Stator and rotor directions are both swept.

## All sweeps for this mechanism

Every figure contains the previous and revised curves; links below cover all saved sweeps, including those not embedded above. CSV files are gzip-compressed.

| Parameter / case | Range | Comparison figure | Numerical data |
|---|---|---|---|
| Incidence | -30 to 30 | [Open](../figures/benner_incidence_incidence.png) | [stator](../data/sweeps/fundamental_benner_incidence_incidence__stator.csv.gz) · [rotor](../data/sweeps/fundamental_benner_incidence_incidence__rotor.csv.gz) |
| Leading-edge ratio | 0.001 to 0.2 (log axis) | [Open](../figures/benner_incidence_leading_edge_ratio.png) | [stator, inlet offset=+10 deg](../data/sweeps/fundamental_benner_incidence_leading_edge_ratio__stator.csv.gz) · [rotor, inlet offset=+10 deg](../data/sweeps/fundamental_benner_incidence_leading_edge_ratio__rotor.csv.gz) |
| Leading-edge wedge angle | 0.1 to 90 | [Open](../figures/benner_incidence_wedge.png) | [stator, inlet offset=+10 deg](../data/sweeps/fundamental_benner_incidence_wedge__stator.csv.gz) · [rotor, inlet offset=+10 deg](../data/sweeps/fundamental_benner_incidence_wedge__rotor.csv.gz) |
| Pitch/chord ratio | 0.05 to 5 | [Open](../figures/benner_incidence_pitch_chord.png) | [stator, inlet offset=+10 deg](../data/sweeps/fundamental_benner_incidence_pitch_chord__stator.csv.gz) · [rotor, inlet offset=+10 deg](../data/sweeps/fundamental_benner_incidence_pitch_chord__rotor.csv.gz) |
| Exit relative Mach number | 0 to 3 | [Open](../figures/benner_incidence_ma_rel_out.png) | [stator, inlet offset=+10 deg](../data/sweeps/fundamental_benner_incidence_ma_rel_out__stator.csv.gz) · [rotor, inlet offset=+10 deg](../data/sweeps/fundamental_benner_incidence_ma_rel_out__rotor.csv.gz) |
| Incidence parameter chi | -10 to 20 | [Open](../figures/incidence_chi.png) | [prescribed reference](../data/sweeps/fundamental_incidence_chi.csv.gz) |
| Incidence parameter chi · incidence chi detail | -0.15 to 0.1 | [Open](../figures/incidence_chi_detail.png) | [prescribed reference](../data/sweeps/fundamental_incidence_chi_detail.csv.gz) |

[All individual functions](../readme.md) · [Methods, validation and reproduction](../methodology.md)
