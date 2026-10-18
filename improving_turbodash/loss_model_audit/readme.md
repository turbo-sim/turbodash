# Loss-model audit

This version keeps the scripts, inputs, numerical results and figures together. The scripts use the existing functions in `../../turbodash/`; no production source code is copied here. Neither section needs scripts or inputs from v1 or v2.

| Review section | Contents |
|---|---|
| [1. Individual functions](1_individual_functions/readme.md) | Eleven numbered reports, ordered Kacker–Okapuu first and Benner last, with old/new function values, derivatives and numerical checks. |
| [2. Stage and turbine](2_stage_turbine/readme.md) | Nine numbered reports covering blade velocity ratio, reaction, exit angle, solidity, meridional velocity ratio, blade aspect ratio, trailing-edge thickness/opening and maximum thickness/chord, with losses and efficiency trends. |

Each section contains its own `scripts/` directory. Inputs, data and figures stay within that section. The individual section uses `figures/`; the turbine section uses `plots/`.

The [profile-loss review](1_individual_functions/report/01_ko_profile.md) follows each term from its equations to value and derivative plots: reaction/impulse references, interpolation, thickness, Reynolds, compressibility and inlet shock. Its extended reference grids cover angles 0–80° and pitch/chord 0–2, alongside variants within chart coverage. Negative-side blending gives 90% of the reaction reference at zero turning and approaches 80% gradually. The individual section has 78 standard sweeps and 102 figures. The closing profile section includes complete-function geometry/Mach sensitivities, a component budget and a dedicated thickness-factor smoothness plot. Three compressibility figures show the smooth outlet-Mach factor, bounded Mach-ratio factor and combined profile correction; the last compares raw algebra, implemented smooth curves and gradual final limiting. The [secondary-loss review](1_individual_functions/report/02_ko_secondary.md) adds six current diagnostics of the aspect-ratio law, inlet/outlet-angle sensitivity and compressibility corrections, alongside the original paper's equations. The latest secondary diagnostics plot $0.2\le\mathrm{AR}\le5$ and include a smooth aspect-factor continuation toward $f_{\mathrm{AR}}=1$; $K_3$ also approaches one smoothly, giving $K_p\le K_s\le1$ without a separate secondary-factor floor. There is no upper continuation on the final $Y_s$. The turbine results retain their recorded revision and have not been recalculated for the secondary-factor or clearance changes. All figures use the repository plotting style, with extrapolation shading where reference ranges are known.

The [clearance-loss review](1_individual_functions/report/04_ko_clearance.md) adds three current figures for the restored 0.78 effective-gap/chord law: finite zero-gap derivatives, gap sensitivity across aspect ratios and outlet-angle sensitivity across inlet-angle ratios. The zero-gap cubic matches the published value, slope and curvature at gap/chord = 0.001. The source helper is shared by the new KO, Moustapha and Benner models; the report distinguishes shrouded-rotor applicability from the existing general rotor interface.

For both saved turbine cases, the [maximum-thickness/chord sweeps](2_stage_turbine/report/08_maximum_thickness_ratio.md) are flat for both stator and rotor: the active KO thickness exponent is zero and incidence stays at its reference. The aspect-ratio and trailing-edge sweeps show the corresponding full-turbine responses; the trailing-edge range includes zero and extends to 0.6.

## Generate the simulations and plots

Use the existing Turbodash project Python environment, with the dependencies from the repository's `pyproject.toml`. From this directory:

```powershell
python -B run.py
```

This evaluates both model versions with 200 base points per standard sweep, adds the section's documented refinement or reference points, and regenerates the outputs. It replaces generated results in this directory, so preserve a separate copy before comparing future changes to the model functions.

The individual-function script checks against the recorded numerical results and stops if they differ. Its explanatory reports remain reviewed text; changed model functions require reviewing those reports and their reference checks before preparing a new comparison.

To redraw the figures from the saved numerical data without repeating the simulations:

```powershell
python -B run.py --render-only
```

Run just one section with `--section individual` or `--section stage`; either can be combined with `--render-only`. The scripts locate files relative to their own locations, so an absolute path to `run.py` also works from another working directory. Keep this folder within the Turbodash repository because the model functions are imported from that checkout.

To check the report links, numbering and script layout:

```powershell
python -B run.py --check
```

The [validation record](validation.json) records this check. Each section records its own numerical validation, model-source hashes and runtime. Results depend on the repository model functions and Python environment used when the scripts are run.

The turbine reports retain failed evaluations and document the existing loss-coefficient normalization issue, two-phase states and supersonic conditions. Their efficiency curves are diagnostics of the current implementation, not validated design predictions.
