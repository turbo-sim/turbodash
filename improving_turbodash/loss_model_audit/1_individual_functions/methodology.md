# Methods, validation and reproduction

[Individual-function reports](readme.md) · [Audit overview](../readme.md)

## What is being compared

The previous implementation is `turbodash/loss_models`; the revised candidate is `turbodash/loss_models_new`. The current run contains 78 individual-function sweeps, each with 200 base points plus shared refinement, and 66 standard comparison figures. Some stator and rotor cases share a figure. The earlier, pre-refinement candidate agreed exactly with the baseline across 77 sweeps; the revised dataset adds a zoom around the Benner incidence reference.

The included scripts recalculate these results directly from the repository functions. [Numerical reproduction checks](data/validation/reproduction.json) compare every regenerated value, derivative and status with the verified reference tables. [Current source hashes and actual imports](data/provenance/source_manifest.json) identify the repository functions and local scripts used, and the [render manifest](data/provenance/render_manifest.json) records saved-data rendering.

The [sweep manifest](data/sweep_manifest.json) and [figure manifest](data/plot_manifest.json) use paths relative to this section. Records prefixed `historical_` retain the earlier audit's provenance and test evidence; their internal paths describe that historical run and are not runtime dependencies.

## Reading the results

Solid curves represent the preserved baseline; dashed curves represent the revised candidate. Both evaluate the same input coordinates. Left panels show values and right panels show **d/dx**, including when the horizontal axis is logarithmic. Those panels do not show d/dlog(x).

Each compressed CSV contains both versions and retains input coordinates, base-grid membership, evaluated values, derivatives, statuses, warnings and errors. Geometry/flow changes, range endpoints and refinement counts are recorded separately for every sweep in the [manifest](data/sweep_manifest.json). These are individual-function experiments; they do not represent a coupled stage equilibrium or a full turbine.

Broad ranges deliberately include stress tests outside trustworthy empirical or thermodynamic domains. Invalid values remain visible as gaps. The [run summary](data/provenance/run_summary.json) records 20,698 finite baseline evaluations and 20,948 finite candidate evaluations, with 2,455 and 2,205 nonfinite evaluations respectively and no exceptions. A large maximum difference can come from unrestricted conversion intermediates near singularities; it is not a bounded final loss or an efficiency improvement. See the [coefficient-conversion report](report/05_conversions.md).

## Derivative method

Every distinct coordinate and function value is retained, including cached bisection evaluations. Both versions evaluate the union of coordinates. Exact duplicate coordinates share a cached evaluation; close but distinct floats remain distinct in the CSV.

Derivative estimation first selects sufficiently separated coordinates inside each original contiguous finite run. Its minimum separation is

$$
h_{\min}=\epsilon_{64}^{1/3}
\max\left(|x_i|,|x_j|,\frac{x_{\max}-x_{\min}}{200}\right).
$$

This is a practical float64 resolution floor that balances second-order truncation and subtraction cancellation. The range term supplies a scale near zero, and the rule changes consistently with physical units. It does not use value or derivative magnitude to reject points. Large coordinate offsets can leave too few usable stencils; the corresponding derivatives are missing rather than invented.

For retained spacings $a=x_i-x_{i-1}$ and $b=x_{i+1}-x_i$, and adjacent secants `L` and `R`, the interior quadratic derivative is $\frac{bL+aR}{a+b}$. The left-endpoint formula is $L-\frac{a(R-L)}{a+b}$; the right endpoint is analogous. Subtracting ordinates before weighting the secants reduces cancellation. At least three retained points are required.

Excluded derivative locations retain their original function values and receive `NaN` derivative entries. Nonfinite coordinates or values split runs before selection. Plots join derivative samples only within the same original finite run and never across an invalid region. This process does not smooth the function, clip slopes, reject outliers by magnitude or interpolate across gaps.

Separate shrinking one-sided probes resolve joins narrower than the ordinary derivative grid. Their full step-size sequences are retained in [first-derivative probes](data/provenance/fundamental_join_intervals.csv), [second-derivative probes](data/provenance/fundamental_curvature_intervals.csv) and [join findings](data/provenance/fundamental_join_findings.json). A single smallest-step estimate does not establish continuity. Reports distinguish C1 joins from C2 joins; C2 claims also rely on endpoint regression checks.

## Plot scaling and derivative validation

Near-constant derivative panels receive a minimum display half-span of 5% of their derivative magnitude when their spread is at most 0.1% of that magnitude. Exactly zero panels use ±0.05. The plots label the expanded scale; no recorded data changes.

Conversion round trips have a separate annotated display rule. Only `raw_roundtrip` and `roundtrip_delta_phi2` qualify, and only if the input increment is constant and all finite residuals are within `1e-7` of its magnitude. The value axis then spans at least ±1% of the invariant magnitude. The derivative axis spans at least 1% of that magnitude divided by the physical x range, expanding to include every retained derivative. The actual maximum residual/slope is annotated. Real variation or active limiting retains ordinary scaling.

The saved known-function validation has 258 recorded values and 221 retained derivative samples, with an original spacing ratio of `2.01e13`. Maximum absolute errors are `3.80e-10` for a derivative of 3 and `6.20e-10` for a derivative of `1e-6`. The examples also preserve a slope change from -2 to 3, a finite-jump spike and an explicit nonfinite gap.

| Evidence | File |
|---|---|
| Near-constant derivative display | [Figure](figures/near_constant_validation.png) |
| Kink, jump and invalid gap | [Figure](figures/features_and_gaps_validation.png) |
| Known-function values and derivatives | [CSV](data/validation/samples.csv) |
| Quantitative derivative validation | [JSON](data/validation/summary.json) |
| Known-function shrinking probes | [CSV](data/validation/shrinking_probes.csv) |
| Independent nozzle directional probes | [CSV](data/validation/nozzle_directional_probes.csv) · [Summary](data/validation/nozzle_directional_summary.json) |

## Coefficient-conversion details

The constant-gamma identities and coefficient definitions are retained. The candidate uses `log1p` and `expm1` to avoid cancellation. Its analytic zero-Mach limits are $d/(1-d)$ for kinetic-to-pressure conversion and $Y/(1+Y)$ for the inverse. These extensions are analytic in Mach squared and do not introduce an arbitrary small-Mach transition band.

At gamma = 1.4 and d = 0.05:

| Mach | Previous inverse | Revised inverse |
|---|---:|---:|
| 0 | NaN | 0.052631578947368425 |
| 1e-12 | NaN | 0.05263157894736843 |
| 1e-8 | NaN | 0.05263157894736842 |
| 1e-4 | 0.0526315831211393 | 0.0526315791966759 |

`limit_output=False` now bypasses both Mach preprocessing and the pressure cap. Its domain is gamma > 1, Mach ≥ 0 and `d < 1/[1+(gamma-1)Mach²/2]`. Invalid inputs remain `NaN`. Negative d is retained as a signed increment continuation needed by the existing Benner polynomial; it is not a physical negative total loss. Forward conversion retains the corresponding `Y > -1` continuation. The enthalpy wrapper explicitly returns `NaN` at its singular $\zeta=-1$ input.

Default limiting retains the safety factor 0.95 below the thermodynamic Mach singularity. For d > 0, the limit is $L=0.95\sqrt{\frac{2(1-d)}{(\gamma-1)d}}$; a compact C2 minimum joins Mach to L over `[0.98L, 1.02L]`. The guard is inactive for d ≤ 0. Negative/nonfinite Mach, invalid gamma and d ≥ 1 never become finite through the guard. The optional Y = 1 output cap has a C2 transition over raw Y in `[0.98, 1.02]`. These widths are numerical choices. Active guards intentionally break unrestricted round trips.

The candidate diagnostics expose unrestricted `Y_raw`, effective-Mach `Y_guarded`, capped `Y_limited`, `Mach_effective` and the raw `domain_margin`. At Mach = 2, gamma = 1.4 and d = 0.9, the raw margin is -0.62 and raw Y is `NaN`, while effective Mach is 0.70808819, guarded Y is 12150.89964 and capped Y is 1. This distinction prevents a numerical plateau from being interpreted as a physical loss asymptote.

At Mach = 0.7 and gamma = 1.4, the output transition edges correspond to d = 0.405134440327 and 0.413410113307. Five-point one-sided estimates at h = 1e-5 gave first-derivative mismatch below `4e-11` and second-derivative mismatch below `2.3e-5` at the lower edge; the upper edge matched zero derivatives within finite-step/roundoff errors. The cap remained bounded and nondecreasing over tested nonnegative inputs.

## Validation and preserved behavior

The prior final test run recorded **269 passed, 8 skipped, 12 warnings** in 15.31 seconds. The warnings are Dash deprecations. [Test output](data/validation/historical_pytest_final.txt) and [JUnit record](data/validation/historical_pytest_final.xml) are retained; these are historical results, not tests rerun during this reorganization. The pre-change suite recorded 227 passed and 8 skipped. An initial sandbox temporary-directory interruption was resolved on retry; no pre-existing failure remained.

The recorded verification reports 50 Sphinx math blocks with no warnings, seven protected source hashes preserved, and 36 existing detailed function docstrings preserved. See the [verification record](data/provenance/historical_validation.json). The original loss-model tree, shared `turbodash/math.py` and Benner penetration cap were preserved. Candidate changes affect selected joins, bounds, interpolation, zero limits and coefficient conversions.

Numerical smoothness does not establish empirical validity. Core coefficient normalization, real-fluid applicability, Moustapha incidence reference and sign conventions, extreme extrapolation, the retained positive-chi negative increment in Benner and inactive physical corrections remain separate physical decisions. Each mechanism report records its own remaining limitations. Stage/turbine trends are reviewed in the [second section](../2_stage_turbine/readme.md).

## Recorded runtime

One serial run evaluated both versions with Python 3.13.2. Function times include wrappers and validity checks; calculation includes refinement, differentiation and compressed CSV writes. Figures and reporting were timed separately.

| Work | Seconds |
|---|---:|
| Imports | 2.959 |
| Calculation and CSV output | 6.982 |
| Baseline function calls, included in calculation | 2.097 |
| Candidate function calls, included in calculation | 3.489 |
| Original plotting | 36.071 |
| Join probes | 0.697 |
| Original wall time | 46.714 |
| Report generation | 0.024 |
| Additional rendering from saved data | 36.506 |

The extra rendering is a later operation and is not included in the original wall time. [Historical timing](data/provenance/historical_timings.json) retains these original measurements. The current calculation records fresh [runtime measurements](data/provenance/timings.json) and [saved-data rendering measurements](data/provenance/render_manifest.json).

## Reproduction

The calculation and plotting scripts live in this section. They import the existing repository functions directly from `turbodash/loss_models`, `turbodash/loss_models_new` and `turbodash/math.py`; no duplicate loss-model source is stored here. Neither another audit version nor an installed copy of TurboDash is used.

Run the following from this section with the project Python environment, or give an absolute script path from any working directory:

```powershell
python -B scripts/run.py
python -B scripts/render_saved.py
python -B scripts/verify.py
```


[run.py](scripts/run.py) recomputes all 78 standard sweeps on 200 base points plus shared refinement, the join/curvature probes, the known-function derivative checks, 24 paired supplemental reference/Reynolds diagnostics, analytical-fit integration checks, the 13 main curves and five close-up curves for the implemented profile interpolation, and the current profile/secondary diagnostics. It regenerates **99 figures**: 66 standard comparisons, two derivative checks, four earlier analytical-reference views, one Reynolds view, four interpolation/thickness views, eleven term views, five final integration/thickness-factor views and six secondary-loss views.

[profile_terms.py](scripts/profile_terms.py) saves 34 direct reference slices on angles 0–80° and spacing 0–2, including the exact zero-spacing limit and extra near-zero samples. Four 2×2 grids place reaction/impulse values on the left and first derivatives on the right: each sweep has a correlation-range and an extended-range view. Complex-step derivatives use a step of $10^{-25}$ and are checked independently by five-point real differences. Extended spacing derivative panels use a smooth asinh display scale to retain both steep negative slopes and smaller positive slopes. Three further figures show $K_1(M_2)$, $K_2(M_1/M_2)$ and $K_p(M_2)$ at six fixed inlet Mach numbers. Their complex-step derivatives are checked against five-point real differences with step $10^{-6}$. The $K_p$ figure has three rows: unrestricted raw algebra, implemented smooth results, and the final continuation mapping. The raw comparison retains the corner at Mach 0.2, the negative $K_1$ straight-line extension above one, unbounded $K_2$, and negative $K_p$ values. Its analytical derivatives are checked independently away from the corner. All inlet/outlet Mach pairs remain visible. Four further shock figures are generated by [shock_review.py](scripts/shock_review.py), with component and derivative checks recorded in [shock validation](data/profile_shock/validation.json). [Term validation and provenance](data/profile_terms/validation.json) record the input domain, derivative errors, data hashes and unchanged production-source hashes during calculation.

[profile_review.py](scripts/profile_review.py) generates 13 reference pitch/chord families and ten angle sweeps using the unchanged analytical fits. [reynolds_review.py](scripts/reynolds_review.py) samples the Reynolds factor and corrected profile. [profile_interpolation_review.py](scripts/profile_interpolation_review.py) calls the current production helper for the two reference-blend figures and the full profile function for the final-loss figure. All three use $-1.2\le z\le1.2$ horizontally; the final-loss curves hold $t_{\max}/c=0.05,0.15,0.20,0.25,0.40$. A fourth figure resolves loss values and first derivatives on $-0.1\le z\le0.1$, with one-sided limits at zero and 0.05. No runtime function replacements are used in these calculations.

[profile_final_review.py](scripts/profile_final_review.py) calls the complete production profile entry point for 38 curves: spacing/angle sweeps at fixed $z=0.5$, outlet-Mach sweeps for six inlet Mach numbers or six hub/tip ratios in both rows, and a separately assembled reference-plus-shock budget. The same prescribed ideal-gas pressures are used throughout: row-frame total pressures 200 kPa and $\gamma=1.4$, with static pressures varying with Mach. Mach derivatives follow that path; Reynolds number remains fixed. All $M_1>M_2$ cases remain visible. Source and data hashes, independent five-point derivative checks, component/gradient sums and 4,000 positive combined-input cases are recorded in [final validation](data/profile_final/validation.json). A fifth figure plots the production thickness factor and its first two derivatives over the full $z$ range and a join close-up. First derivatives use complex step; curvature uses real differences of those slopes, with one-sided stencils near joins to avoid crossing the discontinuous third derivative. Independent analytical derivatives check the result. Running this focused script does not refresh frozen full-audit references.

[secondary_review.py](scripts/secondary_review.py) produces six focused secondary-loss figures: the aspect-ratio factor, loss versus aspect ratio at several exit angles, loss versus exit angle at several inlet angles, the aspect-ratio dependence of the compressibility correction, and separate inlet/outlet-Mach sweeps. It calls the current production secondary function and saves values and first derivatives alongside the paper-based reconstruction. These diagnostics include impulse and reaction configurations and retain the unexpected $M_1>M_2$ cases. The [secondary validation](data/secondary_review/validation.json) records source/data hashes and implementation/derivative checks. Calculation and saved-data rendering can be run independently; this focused review does not refresh frozen full-audit references.

The first two focused profile-interpolation ordinates exclude all downstream corrections. The third includes the thickness, Reynolds, compressibility, shock and KO factors. Standard profile sensitivity figures display the full coefficient and its numerical derivative. Historical CSV names blend_before_floor, floor and floor_margin are retained for comparison compatibility: in candidate rows they now mean the implemented reference blend, its 80%-reaction asymptote, and the difference between those values. They do not imply an applied profile floor.

[render_saved.py](scripts/render_saved.py) redraws saved values without evaluating a model or recalculating derivatives. [verify.py](scripts/verify.py) checks shared grids, all 200 base points, local navigation and exact parsed-number/status agreement with the [reviewed reference digests](data/reference_digests.json). Gzip timestamps do not affect comparison. A subsequent model change requires an explicit review rather than silently refreshing the digests.

The [snapshot before widening the compressibility joins](data/history/before_wider_compressibility_blends.zip) preserves the preceding source and results. The [current revision comparison](data/validation/profile_revision.json) measures the effect of the wider $K_1$ low-Mach join on 0.15–0.25, its high-Mach tail starting at 0.9, and $K_2$ saturation starting at ratio 0.8. The final rational $K_p$ continuation is unchanged: it has a positive slope throughout the valid input interval and a lower limiting value of approximately 0.4154. The comparison checks every baseline value/status at common coordinates across all 78 sweeps and protects every production Python file outside the new KO module.

[Integration validation](data/validation/analytical_integration.json) compares production reference surfaces to the independent analytical evaluator, including zero/tiny spacing, continuation joins, bounds and derivatives. Shared-model propagation is checked for both rows, four spacings and $z=-1,0,1$. [Interpolation validation](data/validation/profile_interpolation.json) checks 13 main curves, five close-up curves with one-sided slope limits, positive values across 605,151 diagnostic grid points, endpoints and nonzero negative-side slopes.

The [explicit reference review](data/validation/reference_review.json) accepts eight changed tables. Six concern compressibility: its component sweep, profile sweeps in inlet and outlet Mach, the secondary-loss sweep in outlet Mach, and the first/second-derivative join probes. Two hub-related tables refresh older data against the shock helpers already present in the archived source. The source comparison confirms that only the executable compressibility-factor function changed after that snapshot. Focused raw and implemented $K_p$ curves are checked separately by [term validation](data/profile_terms/validation.json). There are no additions or retirements among the 91 frozen tables. [review_references.py](scripts/review_references.py) requires this exact change set, protected sources and successful validation/test records; only its explicit acceptance option writes references.

Current validation passes **78 audit/stage-geometry tests** and **85 relevant repository tests**, including value/slope/curvature matching at the wider $K_1$ and $K_2$ joins, monotonicity, the rational $K_p$ continuation and its positive sensitivity, endpoint/bound/domain checks, complex-step derivatives, low-angle sensitivity and full-profile reconstruction under different Reynolds, Mach, thickness and shock conditions. The XML evidence is linked from the profile report. Passing checks establish implementation behavior, not aerodynamic calibration of the extrapolation fractions.

The stage-turbine cases are recalculated with the current interpolation and without a minimum outlet angle. Their comparison checks preserve baseline values, geometry and flow states.

All audit renderers use the repository `set_plot_options()` style and ordered viridis colors. Beige shading marks known extrapolation ranges; in the extended reference grids, dashed segments additionally identify individual curves outside the digitized chart envelope. Numerical smoothing bands and invalid regions retain distinct annotations.
