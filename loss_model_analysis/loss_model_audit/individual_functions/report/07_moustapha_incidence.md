# Moustapha: incidence loss

[Previous: Shared numerical helpers](06_shared_math.md) · [Section index](../readme.md) · [Next: Moustapha: secondary incidence correction](08_moustapha_secondary.md)

These plots evaluate the current production implementation. [Audit overview](../../readme.md).

## Functions and dependencies

`get_incidence_parameter → get_incidence_loss → kinetic-to-pressure conversion`.

## Observed behavior

The asymmetric raw incidence polynomials have a reference slope kink. Absolute-angle conventions may create cusps at inlet angle 0. Some prescribed leading-edge/spacing inputs produce energy increments above 1 despite admissible flow angles.

## Implemented formulation

The empirical incidence function is unchanged. Its production conversion dependency now honors raw/limited modes and accurately handles small Mach. Plots expose the raw energy increment, thermodynamic domain margin and unrestricted pressure identity so bounded output cannot conceal an invalid state.

## Current values and derivatives

[moustapha_incidence_incidence: values and d/dx (PDF)](../figures/moustapha_incidence_incidence.pdf)

[moustapha_incidence_leading_edge_ratio: values and d/dx (PDF)](../figures/moustapha_incidence_leading_edge_ratio.pdf)

[moustapha_incidence_pitch_chord: values and d/dx (PDF)](../figures/moustapha_incidence_pitch_chord.pdf)

## Quantitative checks

Shrinking one-sided probes below use h=1e-5×max(1,|join|). The full h sequence is retained; the smallest h alone is not a continuity test.

| Feature | Version | Extrapolated value jump | Right minus left slope |
|---|---|---:|---:|
| moustapha_incidence_reference | current | -2.347e-13 | 0.0020189 |

4 fundamental sweeps for this mechanism. [Value/gradient data and manifest](../data/sweep_manifest.json), [first-derivative convergence](../data/provenance/fundamental_join_intervals.csv), [second-derivative convergence](../data/provenance/fundamental_curvature_intervals.csv). C2 claims additionally use endpoint regression tests; C1 table interpolation and retained C1 branches are explicitly distinguished.

## Remaining limitations

A smooth nonnegative minimum at the exact reference would impose a reference convention. That decision, signed-angle interpretation and high-incidence validity/continuation remain deferred. No cap was added to conceal energy >= 1. This mechanism therefore still has known C1 limitations.

## All sweeps for this mechanism

Every figure contains current-source values and derivatives; links below cover all saved sweeps, including those not embedded above. CSV files are gzip-compressed.

| Parameter / case | Range | Comparison figure | Numerical data |
|---|---|---|---|
| Incidence | -30 to 30 | [Open](../figures/moustapha_incidence_incidence.pdf) | [stator](../data/sweeps/fundamental_moustapha_incidence_incidence__stator.csv.gz) |
| Leading-edge ratio | 0.001 to 0.2 (log axis) | [Open](../figures/moustapha_incidence_leading_edge_ratio.pdf) | [stator, inlet offset=+10 deg](../data/sweeps/fundamental_moustapha_incidence_leading_edge_ratio__stator.csv.gz) |
| Pitch/chord ratio | 0.05 to 5 | [Open](../figures/moustapha_incidence_pitch_chord.pdf) | [stator, inlet offset=+10 deg](../data/sweeps/fundamental_moustapha_incidence_pitch_chord__stator.csv.gz) |
| Exit relative Mach number | 0 to 3 | [Open](../figures/moustapha_incidence_ma_rel_out.pdf) | [stator, inlet offset=+10 deg](../data/sweeps/fundamental_moustapha_incidence_ma_rel_out__stator.csv.gz) |

[All individual functions](../readme.md) · [Methods, validation and reproduction](../methodology.md)

The complete production model also inherits the new analytical AM references through KO profile loss. The mechanism reviewed here is unchanged by that replacement. See the [profile integration and model-propagation review](01_ko_profile.md#shared-profile-loss-in-moustapha-and-benner) for isolated profile/total differences and independent regression checks.
