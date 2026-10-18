# Methods, validation and reproduction

[Individual-function reports](readme.md) | [Audit overview](../readme.md)

## Current source and experiment design

All 78 individual-function sweeps call `turbodash/loss_models_new` directly. The internal data label `candidate` identifies this production implementation. Each sweep begins with 200 points, then retains additional evaluations at branch joins and regions requiring refinement. There are 66 standard figures plus the focused mechanism diagnostics. Prescribed component states do not represent a solved turbine operating map.

The inlet-shock calculation uses one positive hub/tip coordinate in its hub-Mach fit and span weight, with join 0.5 and lower asymptote 0.3. Shock onset is C2 over hub Mach 0.4-0.5; the exit-normalized shock is exact through 0.25 and approaches 0.7. The Reynolds correction uses identity coordinates from $10^4$ to $10^7$, a low coordinate tending to 1000 and a high coordinate tending to $5\times10^7$. Its high-Re factor is bounded below by $50^{-0.2}=0.457305052$.

[Source hashes](data/provenance/source_manifest.json), [sweep definitions](data/sweep_manifest.json), [run status](data/provenance/run_summary.json) and [render hashes](data/provenance/render_manifest.json) identify the evaluated code and saved evidence.

## Values, failures and derivatives

CSV files retain coordinates, base-grid membership, returned values, derivatives, statuses, warnings and errors. Invalid states remain explicit gaps. Every distinct cached coordinate is retained; exact duplicates share an evaluation. Derivative stencils are chosen inside each contiguous finite run with minimum separation

$$
h_{\min}=\epsilon_{64}^{1/3}\max\left(|x_i|,|x_j|,\frac{x_{\max}-x_{\min}}{200}\right).
$$

The spacing rule controls floating-point cancellation without rejecting values or gradients by magnitude. Three-point physical-coordinate differences use unequal spacing where required. Missing derivative estimates remain NaN, and plots never connect across invalid runs. Shrinking one-sided probes check joins independently. Complex-step derivatives in focused scripts are checked against real differences; C2 claims also use endpoint tests.

[First-derivative probes](data/provenance/fundamental_join_intervals.csv) | [Curvature probes](data/provenance/fundamental_curvature_intervals.csv) | [Join findings](data/provenance/fundamental_join_findings.json)

## Coefficient-conversion details

The constant-gamma conversions use `log1p` and `expm1` for stable small-Mach limits. For kinetic-energy increment $\Delta\phi^2$, the zero-Mach pressure-loss limit is $Y=\Delta\phi^2/(1-\Delta\phi^2)$, with inverse $\Delta\phi^2=Y/(1+Y)$. Unrestricted conversion requires positive gas ratio above one, nonnegative Mach and a positive thermodynamic domain margin. Optional smooth Mach and output guards are documented by the source; active guards intentionally break unrestricted round trips.

Near-constant derivative panels receive a labeled finite display span. Constant conversion round trips additionally retain the measured maximum residual and gradient, so their axes do not exaggerate floating-point noise. These display rules never change stored data.

## Validation and reproduction

[Known-function values](data/validation/samples.csv) and [checks](data/validation/summary.json) test the derivative method on linear functions, a kink, a jump and an explicit gap. [Current numerical verification](data/validation/reproduction.json) checks data grids, source-linked diagnostic records and navigation. Smooth numerical continuation does not calibrate extrapolated aerodynamic behavior.

```powershell
python -B improving_turbodash/loss_model_audit/1_individual_functions/scripts/run.py
python -B improving_turbodash/loss_model_audit/1_individual_functions/scripts/render_saved.py
python -B improving_turbodash/loss_model_audit/1_individual_functions/scripts/verify.py
```

The first command calculates and renders all functions. The second uses saved values and derivatives. Focused profile, shock, Reynolds, secondary, trailing-edge and clearance scripts support `--render-only`. [The stage study](../2_stage_turbine/readme.md) uses the same production helpers and retains signed hub geometry and signed efficiency.
