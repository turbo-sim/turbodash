# Current coefficient conversions

[Previous: Kacker–Okapuu: tip-clearance loss](04_ko_clearance.md) · [Section index](../readme.md) · [Next: Shared numerical helpers](06_shared_math.md)

These plots evaluate the current production implementation. [Audit overview](../../readme.md).

## Functions and dependencies

`convert_kinetic_energy_to_stagnation_pressure_loss / convert_stagnation_pressure_to_kinetic_energy_loss`.

## Observed behavior


## Implemented formulation

Current gas identities use `log1p`/`expm1` and analytic Mach 0 limits: inverse d/(1-d), forward Y/(1+Y). `limit_output=False` now bypasses both Mach and Y guards. `limit_output=True` retains the intentional 0.95 singular-Mach guard with a compact relative 2% transition and a C2 Y cap over 0.98–1.02. The artificial 1e-4 loss offset in the Mach guard is removed using safe inactive operands. Nonfinite/invalid inputs return NaN. Raw identity, guarded value before cap, effective Mach and domain margin are recorded. Round trips use the unrestricted mode.

## Current values and derivatives

[conversion_loss_g1.33_M0.7: values and d/dx (PDF)](../figures/conversion_loss_g1.33_m0.7.pdf)

[conversion_mach_d0.05: values and d/dx (PDF)](../figures/conversion_mach_d0.05.pdf)

[conversion_mach_d0.8: values and d/dx (PDF)](../figures/conversion_mach_d0.8.pdf)

[conversion_gamma: values and d/dx (PDF)](../figures/conversion_gamma.pdf)

## Quantitative checks

Unrestricted round trips over Mach 0, 1e-12, 1e-8, 0.2, 0.7 and 1.2, with loss increments 1e-12, 0.001, 0.05 and 0.2 at gamma 1.33: maximum absolute error **5.55e-17**. At zero Mach and increment 0.05, Y = **0.05263157894736843**. The cap endpoint tests match first/second derivatives at raw Y 0.98 and 1.02; detailed values and domain checks are in the [conversion details](../methodology.md#coefficient-conversion-details).

Unrestricted/guarded intermediates can grow very large near thermodynamic singularities; these are retained, including in the maximum-difference summary. They are not a bounded final loss. Constant round-trip panels use an explicitly labelled minimum display span; the recorded cancellation errors are unchanged.

17 fundamental sweeps for this mechanism. [Value/gradient data and manifest](../data/sweep_manifest.json), [first-derivative convergence](../data/provenance/fundamental_join_intervals.csv), [second-derivative convergence](../data/provenance/fundamental_curvature_intervals.csv). C2 claims additionally use endpoint regression tests; C1 table interpolation and retained C1 branches are explicitly distinguished.

## Remaining limitations

Physical total losses require nonnegative coefficients, gamma > 1 and admissible thermodynamic states. The signed negative-increment continuation is explicit. Optional limiting deliberately breaks round trips when active. Core coefficient normalization and real-fluid applicability remain separate physical decisions; no normalization change was made.

## All sweeps for this mechanism

Every figure contains current-source values and derivatives; links below cover all saved sweeps, including those not embedded above. CSV files are gzip-compressed.

| Parameter / case | Range | Comparison figure | Numerical data |
|---|---|---|---|
| Kinetic-energy loss increment; gamma = 1.1, Mach = 0.2 | -0.01 to 1.05 | [Open](../figures/conversion_loss_g1.1_m0.2.pdf) | [prescribed reference](../data/sweeps/fundamental_conversion_loss_g1.1_m0.2.csv.gz) |
| Kinetic-energy loss increment; gamma = 1.1, Mach = 0.7 | -0.01 to 1.05 | [Open](../figures/conversion_loss_g1.1_m0.7.pdf) | [prescribed reference](../data/sweeps/fundamental_conversion_loss_g1.1_m0.7.csv.gz) |
| Kinetic-energy loss increment; gamma = 1.1, Mach = 1.2 | -0.01 to 1.05 | [Open](../figures/conversion_loss_g1.1_m1.2.pdf) | [prescribed reference](../data/sweeps/fundamental_conversion_loss_g1.1_m1.2.csv.gz) |
| Kinetic-energy loss increment; gamma = 1.1, Mach = 2.0 | -0.01 to 1.05 | [Open](../figures/conversion_loss_g1.1_m2.0.pdf) | [prescribed reference](../data/sweeps/fundamental_conversion_loss_g1.1_m2.0.csv.gz) |
| Kinetic-energy loss increment; gamma = 1.33, Mach = 0.2 | -0.01 to 1.05 | [Open](../figures/conversion_loss_g1.33_m0.2.pdf) | [prescribed reference](../data/sweeps/fundamental_conversion_loss_g1.33_m0.2.csv.gz) |
| Kinetic-energy loss increment; gamma = 1.33, Mach = 0.7 | -0.01 to 1.05 | [Open](../figures/conversion_loss_g1.33_m0.7.pdf) | [prescribed reference](../data/sweeps/fundamental_conversion_loss_g1.33_m0.7.csv.gz) |
| Kinetic-energy loss increment; gamma = 1.33, Mach = 1.2 | -0.01 to 1.05 | [Open](../figures/conversion_loss_g1.33_m1.2.pdf) | [prescribed reference](../data/sweeps/fundamental_conversion_loss_g1.33_m1.2.csv.gz) |
| Kinetic-energy loss increment; gamma = 1.33, Mach = 2.0 | -0.01 to 1.05 | [Open](../figures/conversion_loss_g1.33_m2.0.pdf) | [prescribed reference](../data/sweeps/fundamental_conversion_loss_g1.33_m2.0.csv.gz) |
| Kinetic-energy loss increment; gamma = 1.4, Mach = 0.2 | -0.01 to 1.05 | [Open](../figures/conversion_loss_g1.4_m0.2.pdf) | [prescribed reference](../data/sweeps/fundamental_conversion_loss_g1.4_m0.2.csv.gz) |
| Kinetic-energy loss increment; gamma = 1.4, Mach = 0.7 | -0.01 to 1.05 | [Open](../figures/conversion_loss_g1.4_m0.7.pdf) | [prescribed reference](../data/sweeps/fundamental_conversion_loss_g1.4_m0.7.csv.gz) |
| Kinetic-energy loss increment; gamma = 1.4, Mach = 1.2 | -0.01 to 1.05 | [Open](../figures/conversion_loss_g1.4_m1.2.pdf) | [prescribed reference](../data/sweeps/fundamental_conversion_loss_g1.4_m1.2.csv.gz) |
| Kinetic-energy loss increment; gamma = 1.4, Mach = 2.0 | -0.01 to 1.05 | [Open](../figures/conversion_loss_g1.4_m2.0.pdf) | [prescribed reference](../data/sweeps/fundamental_conversion_loss_g1.4_m2.0.csv.gz) |
| Mach number; kinetic-energy loss increment = 0.001 | 0 to 3 | [Open](../figures/conversion_mach_d0.001.pdf) | [prescribed reference](../data/sweeps/fundamental_conversion_mach_d0.001.csv.gz) |
| Mach number; kinetic-energy loss increment = 0.05 | 0 to 3 | [Open](../figures/conversion_mach_d0.05.pdf) | [prescribed reference](../data/sweeps/fundamental_conversion_mach_d0.05.csv.gz) |
| Mach number; kinetic-energy loss increment = 0.3 | 0 to 3 | [Open](../figures/conversion_mach_d0.3.pdf) | [prescribed reference](../data/sweeps/fundamental_conversion_mach_d0.3.csv.gz) |
| Mach number; kinetic-energy loss increment = 0.8 | 0 to 3 | [Open](../figures/conversion_mach_d0.8.pdf) | [prescribed reference](../data/sweeps/fundamental_conversion_mach_d0.8.csv.gz) |
| Specific heat ratio gamma | 0.98 to 1.8 | [Open](../figures/conversion_gamma.pdf) | [prescribed reference](../data/sweeps/fundamental_conversion_gamma.csv.gz) |

[All individual functions](../readme.md) · [Methods, validation and reproduction](../methodology.md)
