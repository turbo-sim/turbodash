# Method, interpretation and reproduction

[Back to stage/turbine index](../readme.md)

The study calls `turbodash.core_turbine.compute_turbine_performance` at every grid point, with only the selected design variable and loss-model selector changed. Previous means `benner` in the repository's [previous functions](../../../../turbodash/loss_models/); revised means `benner_new` in the repository's [revised functions](../../../../turbodash/loss_models_new/). Both use the same [turbine core](../../../../turbodash/core_turbine.py), continuous blade geometry and shared math. This isolates the loss-function changes; it is not a comparison against Arianna's original software revision or pre-solidity geometry. The calculation and plotting scripts, inputs and results live in v3; project functions are imported from the existing repository without copying their implementation here.

The two copied YAML inputs describe Arianna's final R245fa single-stage designs at recuperator effectiveness 0% and 80%. The suffixes do not denote inlet angle: both turbine inlet flow angles are zero. Stage and whole-turbine results coincide for these single-stage cases; no unsupported multistage design is invented.

The reconstructed states include two-phase R245fa and supersonic stations. The source gas-cascade correlations are not validated here for these conditions. Saved station vapor qualities and Mach numbers make those limitations directly inspectable.

## What is recalculated

Every evaluation rebuilds velocity triangles, HEOS real-fluid states, continuous geometry, Reynolds and Mach numbers, row losses and overall efficiencies. Geometry follows design flow angles, so this is a family of redesigned turbines, not a fixed-blade off-design experiment. Fixed boundary inputs are the saved inlet state, backpressure, mass flow, specific shaft speed, radius ratios and all nonswept design inputs. The entry mass-flow and inlet-state differences between the two recuperator cases are retained.

Each one-dimensional sweep has 200 evenly spaced base points, plus the saved coordinate when it is absent. Aspect-ratio sweeps also include exact H/c values 1.8, 1.9, 2.0, 2.1 and 2.2 to capture the KO and Benner secondary transitions. Trailing-edge sweeps include the exact table coordinates 0.2 and 0.4. Both versions receive exactly the same coordinates, including failed points. Exceptions, warnings, nonfinite results, negative loss components, efficiency values outside [0,1], supersonic stations and penetration-domain flags are preserved. `status=ok` means a finite numerical result, not a validated physical design. No clipping, exception replacement or interpolation across invalid points is applied.

## Coefficients and efficiency caveat

Benner returns pressure-loss coefficients Y, including each row's component sum. The dispatcher option `loss_coefficient='kinetic_energy'` selects only the comparison definition; it does not convert the returned empirical Y. The current turbine driver nevertheless multiplies this returned coefficient by the row exit kinetic energy to estimate enthalpy loss. This known normalization issue is unchanged for both versions. The efficiency results are useful for tracing software changes, and must not be treated as physically validated efficiencies until the convention is resolved.

The plots therefore keep stator and rotor Y separate. The combined driver diagnostic is $\frac{Y_{\mathrm{stator}}V_2^2+Y_{\mathrm{rotor}}W_4^2}{2\Delta h_{\mathrm{is}}}$, using each row's own velocity and a common whole-turbine enthalpy denominator. It is not the sum of the two pressure coefficients. The exit kinetic fraction is $V_4^2/(2\Delta h_{\mathrm{is}})$.

The driver uses isentropic states to estimate the losses after computing its flow triangles. Losses do not feed back through an iterative non-isentropic solution. Both candidate and baseline retain the existing Benner half-height penetration cap. Negative tiny baseline incidence increments and larger extrapolation artifacts, when present, stay visible. No new physical laws or coefficient conversions were introduced by this audit.

## Numerical checks

There are 11,972 full API evaluations across 26 sweeps. Status counts are `{'previous': {'exception': 298, 'ok': 5688}, 'revised': {'exception': 298, 'ok': 5688}}`. There are 596 failed/nonfinite evaluations. Maximum finite single-stage energy accounting residual `eta_ts + driver_row_loss_fraction + exit_kinetic_fraction - 1` is 8.82e-12; maximum row component-sum residual is 2.33e-10. Reproduction assertions also compare stage and turbine efficiencies, identical old/new flow/geometry diagnostics, identical coordinate grids, saved design snapshots and unchanged input configurations. For the geometry sensitivities, they verify achieved true-chord/opening ratios, saved-coordinate reproduction, fixed velocity triangles and untouched-row geometry and losses. Zero trailing-edge thickness must produce exactly zero revised trailing loss; the previous implementation retains its smoothing offset. These verify audit plumbing and arithmetic, not the physical validity of the coefficients. For enormous finite values near the triangle boundary, arithmetic residuals are checked against the magnitude of the terms being summed; all absolute residuals remain in the CSVs.

Counts below refer to finite evaluations only and can overlap. Exceptions/nonfinite results are counted separately; error types: `{'AssertionError': 596}`. Every sweep preserves the requested x coordinates and the original error messages in its CSV.

| Review condition | Previous | Revised |
|---|---:|---:|
| Supersonic station | 5688 | 5688 |
| At least one two-phase station, 0 <= quality <= 1 | 5688 | 5688 |
| Negative component (including tiny incidence offsets) | 5688 | 0 |
| Penetration outside model domain | 325 | 325 |
| Efficiency outside [0,1] | 39 | 38 |


All original values and review flags are in the CSVs; aggregate statistics and model/input source hashes are in [summary.json](../data/summary.json) and [source_manifest.json](../data/source_manifest.json). [Saved design snapshots](../data/reference_results.json) record the complete full API outputs. The two-phase review count is derived from the saved station-quality columns, not added retroactively to the original CSV flag strings.

The v3 simulations were freshly calculated using the repository source. [Calculation provenance](../data/calculation_provenance.json) records the runner hash and all project-source and YAML-input hashes captured before imports and calculations. The runner checks that those files remain unchanged until the calculation ends, and records the repository origin of every imported TurboDash module. [Render manifest](../data/render_manifest.json) identifies the renderer separately. Paths in these manifests are relative to the repository root. Keep this audit at `improving_turbodash/loss_model_audit` inside the repository when rerunning calculations.

The complete preceding stage results are preserved in the [snapshot before extending the reaction sweep](../data/history/before_reaction_range_extension.zip), with a [SHA-256 preservation manifest](../data/history/before_reaction_range_extension.json). The [current revision comparison](../data/profile_revision_comparison.json) checks all 26 grids, requires every preceding baseline and candidate coordinate to be retained with unchanged results, and records the added reaction samples. The production model code is unchanged; the seven baseline/shared-math hashes are checked separately. Earlier rational-continuation, analytical-fit, Reynolds and v2 records remain historical provenance. No production-function source is copied into the stage archive.

The main comparison explicitly selects **Benner** (`benner` / `benner_new`), including KO profile loss plus Benner penetration weighting, secondary and incidence treatment. Its totals are not pure KO. A separate [direct-KO full-API check](../data/direct_ko_checks.json) selects `kacker_okapuu` / `kacker_okapuu_new`: two saved cases, three sensitivities (blade velocity ratio, stator exit angle and solidity), eleven coordinates each, 132 full-API evaluations. Of these, 116 are finite and 16 fail; every sensitivity retains finite pairs. [All direct-KO samples](../data/direct_ko_samples.csv) retain statuses, component values, efficiencies and diagnostics. These checks retain the same core normalization and real-fluid limitations.

The nozzle/impulse reference surfaces are bounded by 0.5. The exit-normalized inlet shock term now has its own smooth upper continuation: raw coefficients through 0.25 are unchanged; larger coefficients approach 0.5 with matching value, slope and curvature at the join, before the outer KO multiplier. This engineering safeguard controls large pressure-head amplification; it is not an original KO calibration. The [profile-loss audit](../../1_individual_functions/report/01_ko_profile.md#6-hub-mach-ratio-and-inlet-shock-term) gives its equations and component plots. The corrected KO component uses the bounded reaction/impulse interpolation with gradual negative-side extrapolation, the smooth magnitude of the outlet angle without a minimum angle, and the thickness, Reynolds, Mach and inlet-shock corrections. The thickness factor uses a quintic smoothstep over `0 < z < 0.05`, centered at 0.025; it is exactly one below this interval and recovers the original power law above it. Its negative-side reference blend approaches 80% of the reaction reference; this is an asymptote, not an applied floor. Neither corrected profile loss nor total row loss is capped at 0.5. Current coefficients are those in the standalone study's `current_fit`; no production data-file reads or fitting-study imports are used.

The revised compressibility helper is shared by KO profile and secondary loss; the main Benner comparison uses its profile correction. The existing wider joins spread the K1 lower join over outlet Mach 0.15 to 0.25, connecting the weak low-Mach variation to the original straight line. Its positive decaying tail now begins at outlet Mach 0.9: $K_1=\frac{0.125}{1+t+t^2}$, with $t=\frac{M_2-0.9}{0.1}$. K2 follows the squared Mach ratio up to 0.8, then approaches one from below without a finite-ratio plateau; the continuation uses $a=0.64$ and $d=0.36$. These wider joins retain continuous values and first two derivatives.

The existing rational Kp continuation is unchanged. The raw combined factor is $x=1-K_2(1-K_1)$, and the final Kp equals x at and above 0.6. Below 0.6, it follows $K_p=0.4+\frac{0.2}{1+t+t^2}$, with $t=\frac{0.6-x}{0.2}$. This rational continuation matches the value and first two derivatives at 0.6, so it is C2, and gradually resists further loss reduction without a flat region. Although 0.4 is the formal asymptote for arbitrarily negative x, the bounded K1 and K2 give $0\le x\le1$; the smallest reachable Kp is therefore approximately 0.415385, at x = 0. Its slope with respect to x remains at least $7/169$, approximately 0.04142, throughout this physical interval. This is a modeling continuation for optimization, not Aungier's exact Mach-input formulation, an original KO requirement or an additional aerodynamic calibration. The saved Benner design changes are listed below. Every preceding coordinate, state, geometry field, status and component remains unchanged in this audit-only extension; very large efficiency changes at extreme stress-test points remain visible in the CSVs and are not physical performance gains.

| Recuperator case | Current candidate eta_ts [%] | Change from preceding candidate [pp] |
|---|---:|---:|
| 0% | 87.001404 | +0.000000 |
| 80% | 86.220876 | +0.000000 |


The six added geometry studies per case vary stator and rotor separately. They use physical $H/c$ from 0.5 to 5, $t_{\mathrm{TE}}/o$ from 0 to 0.60, and $t_{\max}/c$ from 0.05 to 0.40. Both chord ratios use true chord. The core uses meridional-chord inputs for aspect ratio and maximum thickness, so these inputs are divided by the cosine of the saved row stagger. The saved triangles keep this stagger fixed. Each CSV retains the requested coordinate, actual dimensions and ratios, stagger in degrees, and raw core input ratios. The input conversion is independently tested against full API geometry, and each sweep verifies the untouched row. The individual reports describe the coupled dimension changes. Trailing-edge plots mark the table endpoint 0.4; the requested range up to 0.6 demonstrates endpoint continuation outside the tabulated range. These parameter ranges explore the correlations, not mesh-validated blade shapes.

## Reproduce

From the `loss_model_audit` directory, use the existing project Python environment with the dependencies in [pyproject.toml](../../../../pyproject.toml):

```powershell
python -B 2_stage_turbine/scripts/run_sweeps.py --points 200
python -B 2_stage_turbine/scripts/run_sweeps.py --render-only
python -B 2_stage_turbine/scripts/test_geometry_sweeps.py
python -B 2_stage_turbine/scripts/compare_profile_revision.py
python -B 2_stage_turbine/scripts/direct_ko_check.py
```

Use `--render-only` to redraw the reports from the saved CSVs without rerunning the turbine calculations. The script resolves its own location, so an absolute script path also works from any working directory. All plots, data, runtime records and Matplotlib cache stay under this section. Reproduction uses the repository functions directly and does not read v1 or v2.

Measured import time: 3.02 s; calculation and CSV writing: 290.84 s; plotting: 24.02 s. The saved [timings](../data/timings.json) separate API seconds by version. Python 3.13.2 on Windows.
