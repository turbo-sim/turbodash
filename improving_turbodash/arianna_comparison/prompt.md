# Reproduce Arianna's meanline analyses and compare the loss models

Use this prompt for a future analysis run. This file specifies work to perform; it does not claim that the reproduction or comparison has been completed.

Work in `C:/Users/roagr/git_projects/turbodash`. Reproduce Arianna Colleoni's original meanline results, then quantify how the revised loss functions change the same designs and parameter sweeps. Use the full turbine calculation API and save numerical results before rendering figures. Produce a concise report with figures that a human can compare directly.

## Sources and output location

Read Arianna's repository without modifying it:

```text
C:/Users/roagr/OneDrive - Danmarks Tekniske Universitet/Roberto/University/Students/Arianna Colleoni/Turbomachinery-Special-Course
```

The authoritative chapter is `docs/source/design/meanline_calculations.md`; its boundary-condition context is `docs/source/design/cycle_calculations.md`. The scripts, inputs and original outputs are under `simulations/2_meanline_calculations/` in that repository. Read the chapter and derive the required figure inventory from its image references. The archived chapter contained 12 figure references; detect and explain any changes rather than silently assuming the inventory is unchanged.

Frozen reference material is retained in this Turbodash checkout under `improving_turbodash/loss_audit_v1/inputs/arianna/reference/`; its inventory and source hashes are in `manifest.json`. The bundle contains the chapter, original scripts, YAML inputs, original figures and supporting CSVs. Prepared before/current input cases and their `conversion.json` are in `improving_turbodash/loss_audit_v1/inputs/arianna/cases/`; the historical engine identity and hashes are in `improving_turbodash/loss_audit_v1/inputs/arianna/historical_model_provenance.json`. Inspect these before using them. Prefer frozen inputs for a controlled historical reproduction. If an asset is missing, copy it from the external source into the run's reference directory and record its path and SHA-256. Preserve the distinction between the frozen reference and any newer source version. The earlier reference recorded Arianna revision `2c516f916ada80d12eb949a1a185754bf6424745`.

Keep this `arianna_comparison/` folder as a single prompt file. Put new calculations in the stage/turbine section of `improving_turbodash/loss_audit_v2/`, inside an `arianna/` subfolder. First inspect the existing v2 index to identify its stage/turbine directory. Within that destination use `reference/`, `scripts/`, `data/`, `plots/` and `report/01_arianna_comparison.md`. Keep provenance and verification in JSON, and link the report from the existing stage/turbine index. Keep report filenames lowercase and numbered in reading order; organize reproduced figures in original chapter order. If individual mechanism reports are needed later, order Kacker-Okapuu first and Benner last, consistently with v2. Avoid extra capitalized instruction/checklist Markdown files. Preserve existing runs by selecting a separate run directory when source hashes change.

## Keep three model versions distinct

| Label | Purpose | Required implementation |
| --- | --- | --- |
| Arianna historical | Reproduce the original reported figures and numerical tables | Historical Turbodash commit `eea5aa93b3adefc2b5d0ef0fb1baba704e626bde`: integer blade counts during performance calculation, prescribed Zweifel sizing, old Benner penetration cap of 0.99 |
| Current baseline | Reference for the isolated loss-function comparison | Preserved `turbodash/loss_models/`, selected through `loss_model: benner` in the current turbine API |
| Revised candidate | Measure the effect of the new loss implementations | `turbodash/loss_models_new/`, selected through `loss_model: benner_new` in the same current turbine API |

Extract the historical committed package into an isolated source directory without resetting, checking out over, or modifying the working tree. Verify that the commit exists before extraction; if it does not, use a retained historical source snapshot only after checking its manifest. Run historical and current packages in separate Python processes with explicit import paths, and record the imported module filenames and hashes. Never assume the installed package is the intended source.

The current baseline already includes continuous meanline geometry, prescribed true-chord solidity and the smooth Benner half-height penetration cap. Those earlier changes must not be attributed to the new loss functions. Compare historical versus current baseline separately from current baseline versus revised candidate. If the candidate evolves before this prompt is executed, freeze and describe the actual candidate files used; do not reuse older result labels for changed code.

For the principal current comparison, change only `inputs.loss_model` between `benner` and `benner_new`. Keep the current `core_turbine`, equation of state, geometry, solver settings and shared math identical. The dispatcher also supports paired `kacker_okapuu`/`kacker_okapuu_new` and `moustapha`/`moustapha_new`; use these as clearly labelled supplementary attribution studies if useful, not as substitutes for Arianna's Benner cases. Do not overwrite baseline functions or alter the production default.

## Original cases and geometry controls

Load only the `inputs` section of each original YAML for calculation; saved result sections are reference evidence:

| Setting | `R245fa_turbine_0_final.yaml` | `R245fa_turbine_80_final.yaml` |
| --- | --- | --- |
| Case label | `eps0`: recuperator effectiveness 0% | `eps80`: recuperator effectiveness 80% |
| Fluid / turbine | R245fa / single-stage axial | R245fa / single-stage axial |
| Inlet state | `PQ_INPUTS`, 1,930,000 Pa, quality 0 | `PT_INPUTS`, 1,930,000 Pa, 362.65 K |
| Exit static pressure | 400,000 Pa | 400,000 Pa |
| Mass flow | 15.735 kg/s | 10.432 kg/s |
| Inlet angle / stator exit angle | 0 degrees / 80 degrees | 0 degrees / 80 degrees |
| Selected velocity ratio / reaction | 0.7 / 0.5 | 0.7 / 0.5 |
| Specific speed | 0.5 | 0.5 |
| Meridional ratios `m12`, `m23`, `m34` | 0.1, 1, 0.51 | 0.158, 1, 0.49 |
| Legacy stator / rotor Zweifel inputs | 0.45 / 0.8 | 0.45 / 0.8 |

The 0 and 80 labels are recuperator effectiveness values, not inlet flow angles. Retain all remaining source YAML values, including radius ratios, aspect ratios and blade-shape definitions. Do not replace the prescribed inlet states with a guessed gas state.

For historical reproduction use the original inputs and original spacing convention. For the controlled current comparison, convert each selected original design once with `core_turbine.convert_turbine_config_to_solidity()`, then freeze that converted configuration. The previously recorded true-chord solidities were approximately:

| Case | Stator `c/s` | Rotor `c/s` |
| --- | --- | --- |
| `eps0` | 0.9921679737684417 | 0.9669562485509555 |
| `eps80` | 0.9921679737684418 | 0.9955236194309941 |

Check these against the preserved conversion record and explain mismatches before proceeding. Hold these solidities fixed in both current-model branches while varying other design parameters. Do not reconvert at every sweep point: that would restore a fixed-Zweifel design path. A historical fixed-Zweifel sweep and a current fixed-solidity sweep are different design specifications, even if their selected point matches.

Use true chord `c`, mean circumferential pitch `s`, and full mean blade height consistently. The legacy output `solidity` may mean meridional chord divided by pitch; use `solidity_chord` explicitly. Keep continuous geometry for current performance and a separate integer realization for manufacture/export plots. Never recompute losses from a mixture of these two geometries. For axial source cases the inlet and exit radius conventions coincide; document the actual velocity-ratio definition if adding radial or unequal-radius cases.

Preserve the existing current Benner limiter in both current branches: effective penetration relative to full mean height is capped at 0.5 with its C2 transition over raw values 0.45 to 0.55. Record raw penetration, effective penetration and primary weight separately. An intentional later change to that limiter needs its own ablation comparison; do not combine it silently with loss-function refinements.

## Reproduce the original analyses first

Copy and adapt the following original scripts only to redirect paths, select the isolated source package, run headlessly and record inputs/results. Preserve their original numerical grids and plotting logic in this reproduction pass. Run the final-design exporter once per case.

| Original script | Required analysis and original sampling |
| --- | --- |
| `plot_efficiency_vs_nu.py` | Full redesign for both cases; reaction 0, 0.25, 0.5 and 0.75; velocity ratio 0.30 to 1.00 in 0.05 steps plus 0.56; total-to-static and total-to-total efficiency |
| `plot_statorMach_vs_nu.py` | Same redesign grid; stator-exit absolute Mach `v2/a2` |
| `plot_rotorMach_vs_nu.py` | Same redesign grid; rotor-inlet relative Mach `w3/a3` |
| `plot_stator_flaring.py` | At reaction 0.5 and velocity ratio 0.7, `m12 = geomspace(0.04, 1.0, 40)`; preserve its additional comparison-table points and zero-flaring calculation |
| `plot_zweifel.py` | Independently vary stator and rotor Zweifel from 0.30 to 1.20 in 0.05 steps, holding the other row at its selected value; preserve blade-profile sampling and passage-width calculation |
| `comparison_cases.py` | Selected designs: velocity triangles, meridional channels and blade-to-blade geometry |
| `run_turbodash_from_code.py` | Each selected design: performance/geometry/station CSVs and figures; efficiency trends from `compute_turbine_efficiency_trends()` on `linspace(0.001, 1.0, 100)` |

The last efficiency-trend calculation holds selected-design loss coefficients fixed. It is not a full redesign sweep. Label both types of analysis explicitly and preserve both when reproducing the report.

Build a figure-by-figure comparison in the chapter's order, showing the actual original image beside the historical reproduction. Compare original CSVs and saved YAML values numerically wherever available. Use the chapter source as the reference if a documentation server or rendered report is unavailable; do not depend on an assumed local server URL. A screenshot alone is not numerical validation. Record any missing original arrays rather than inventing them from plot pixels.

## Compare current baseline and revised candidate

Run both current loss variants through `core_turbine.compute_turbine_performance()` at every full-redesign point. Existing individual-function or isolated-row audit sweeps cannot substitute for this full API calculation. Reuse the same inputs, parameter grids, valid-data rules and plotting axes for each pair. Use at least 201 evenly spaced points per linear sweep and the union with every original sample; use geometric spacing for `m12`. Preserve original grids as separately identifiable samples. Refine locally around joins, extrema or failure boundaries when necessary, without replacing calculated values by fitted curves.

Cover the original velocity-ratio/reaction and `m12` studies, plus these direct design studies:

- Reaction at fixed selected velocity ratio over 0 to 0.75, including 0, 0.25, 0.5 and 0.75.
- True-chord solidity 0.5 to 2.5 for each row independently, with 201 common points plus the selected solidity. Plot prescribed and realized solidity separately if historical integer geometry is included.
- Stator exit angle 30 to 80 degrees, including 30, 50, 65, 75 and 80; retain the selected case settings otherwise.
- Rotor meridional velocity ratio `m34` over 0.3 to 1.0, including both source values and 1.0; hold `m12` at its selected case value.
- Stator and rotor aspect ratio over 0.5 to 3.0, one row at a time, retaining the selected values 2.0 and 1.3. State the code's aspect-ratio chord convention and report the resulting true-chord aspect ratio.

Label the added sweeps as extensions to Arianna's original analyses. Keep pressure, mass flow, rotational-speed specification, stage count and other geometry inputs fixed unless a plot explicitly identifies them as the varied parameter. State how actual shaft speed and radius respond to a fixed specific-speed specification.

For each successful point retain overall and stage `eta_ts`, `eta_tt`, power, torque and loss coefficients; stator and rotor total/profile/secondary/trailing-edge/clearance/incidence coefficients; the coefficient definitions and normalizing quantities; weighted and unweighted primary losses; penetration diagnostics; Mach, Reynolds number, flow and metal angles; pitch, true chord, height, opening, solidity and blade count. Retain pressure and thermodynamic station data needed to verify the calculation. If a model has no particular diagnostic, mark it unavailable instead of treating it as zero.

Show baseline/candidate overlays and candidate-minus-baseline differences for efficiency and loss components. Express efficiency changes in percentage points and retain raw fractions in the data. Define every reported overall loss measure: do not sum stator and rotor coefficients with different reference velocities or add the diagnostics dictionary to the physical loss sum. Use the solver's documented stage/energy accounting when relating row coefficients to stage efficiency. Attribute major changes to the affected subfunction using saved diagnostics or controlled row evaluations at identical states and geometry.

## Failure handling, validation and deliverables

Use the repository's development environment and record Python/package versions, equation-of-state backend, source revision and hashes. Check required imports first, including NumPy, SciPy, Matplotlib, YAML, jaxprop/CoolProp and pandas for original table exports. Use a noninteractive plotting backend. Do not start Dash or execute original scripts in Arianna's source directory.

Save one status record per requested parameter point, including exceptions, warning text, nonfinite fields and the reason a curve point was masked. Keep failed points and efficiencies outside 0 to 1 in raw tables. Apply Arianna's original masks only to faithful reproduction figures; include an additional visible validity/failure panel for the new comparisons. Break curves across failed intervals. Do not fill, clamp, interpolate across failures or silently change inputs to obtain a plausible curve. Restrict retries to documented numerical troubleshooting that preserves the model and requested design, and save both attempts.

Validate source/reference hashes and import paths; verify baseline/candidate inputs are identical apart from model selection; check continuity, velocity triangles, Euler work, power/torque relations, row-loss bookkeeping, coefficient conversions and the penetration bounds. Compare the historical selected designs to the saved YAML efficiencies and CSVs using explicit scaled tolerances that account for equation-of-state/environment differences. Confirm the original sample counts and all report figure references. Recompute representative current sweep points independently through the public API, including endpoints, selected designs, large differences and failure boundaries. Record maximum numerical residuals and every failed check; a smooth curve is not evidence of physical validity.

Separate calculation/CSV time from plot-rendering time. Support rendering solely from saved data. Provide runnable commands for the actual created runner, a JSON provenance/verification record, tidy CSV or CSV.gz data with units and model labels, high-resolution PNG plus SVG plots, and a single linked `report/01_arianna_comparison.md`. The report should contain the model-version table, source/reproduction agreement, original-figure comparisons, paired current-model trends, selected-design and maximum-difference summaries, failure counts, and short explanations of the largest changes and unresolved correlation limitations. State exactly which results were regenerated, which were read from frozen references, and which analyses remain incomplete. Do not claim that numerical refinement establishes the physical accuracy of extrapolated correlations.
