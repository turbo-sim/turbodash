# Blade profile controls: an illustrated sensitivity guide

Use this guide to see which inputs change a **2D blade profile**, and how.
The main gallery covers shape inputs; a separate appendix illustrates spacing,
span, surrounding domains and numerical settings. The
[blade geometry workflow](blade_geometry.md) explains where to edit the inputs
and how the resulting geometry reaches previews and exports.

## Which inputs change the outline?

| Desired change | Inputs to explore |
| --- | --- |
| Change the overall axial size | Axial meridional chord |
| Change the blade's tangent directions or distribute its turning differently | Inlet/outlet metal angles, interpolation, half-turning location, turning concentration |
| Add thickness, move the thick region or reshape its shoulders | Nominal maximum thickness and location, body exponent, independent leading/trailing fullness |
| Change the nose or trailing closure | Unrounded endpoint thickness ratios, rounding fractions and rounding exponents |
| Thin the pressure side locally near the nose | Leading-edge relief strength |
| Change a radial section's extent or compensate radius-dependent thickness | Radial inlet/outlet radii, physical-thickness correction |

On an axial cylinder, mean radius, blade height and blade count leave the
developed individual outline unchanged. They still change machine geometry,
span or pitch. Domain extensions leave the blade fixed; sampling changes its
discretization; thickness tolerance changes warnings. Physical-thickness
correction is inactive on an axial cylinder. Stream-surface type changes the
physical mapping and is set in **Meanline**, not edited independently for a row.

## Parametrization and the experiment

The production kernel constructs a camberline from signed endpoint metal
angles and a turning distribution in a conformal chart. A rational Denton body
adds independent leading/trailing shoulder fullness, endpoint body thickness,
tip rounding and optional pressure-side leading relief. It then maps the
section onto an axial cylinder or a radial plane. The study imports these
production functions; it contains no alternative shape equations.

Every sweep starts again from one of the two exact reference baselines and
changes **one independent input**. The other independent inputs remain fixed,
including the opposite shoulder, opposite edge and shape preset. Dependent
quantities, such as arclength, pitch and actual maximum thickness, can change.
Changing an angle in this study does not recalculate Meanline or interpolate
a new preset. No companion input is silently adjusted to rescue an invalid case.

Continuous controls have **four nearby values, including the exact baseline**.
Dimensions, thickness ratios and exponents generally vary from 90% to 110% of
their reference values; rounding extents and domain extensions use 80% to 120%.
Angles vary within 5° of the baseline, locations within 0.05, and fullness
within 0.1. Relief uses a small one-sided range when its baseline is zero.
**Blade count stays fixed at 24 for reaction and 12 for impulse in every case.**
Integers stay integers; categorical and Boolean controls show all
choices. Each entry distinguishes the chosen illustrative interval from the
kernel's admissible input bounds. The GUI slider span is an exploration aid,
not an additional mathematical or aerodynamic bound.

### How to read the figures

Every figure has **one row and three columns**: physical camberline, upper
unrolled thickness representation, and physical blade profiles. Reaction and
impulse have separate figures. Edge close-ups use this same layout, with the
third panel showing the edge detail. Colors come from **plasma sampled between
0.25 and 0.75**, with the same value/color pairing across all three panels and
the matching edge-detail figure. Lines are thin (0.85 pt); the baseline is
dashed and slightly heavier (1.15 pt). Every panel has a full box. The study
explicitly calls TurboDash's `set_plot_options()` for its publication styling.
Each panel uses fixed limits
across its accepted curves; camberline and profile axes have equal physical
scales. Different blade families have different dimensions and axis limits.

“**Upper thickness**” means **half of the total side-to-side paired physical
thickness**, plotted above zero against the **baseline camberline's physical
arclength at the same conformal station**. The abscissa is fixed to the
baseline even if a sweep changes chord or turning. This is a thickness
representation, not necessarily one actual blade surface: relief reduces the
total even when the opposite surface is unaffected. For the cylinder the
thickness is intrinsic distance on the developed surface; in the radial plane
it is the physical distance between paired side samples. It is not the minimum
distance between the complete side curves.

Axial profiles use developed coordinates `(z, rθ)` in mm, an isometric unrolling
of the cylinder. Radial profiles use machine `(X, Y)` in mm, obtained from the
kernel's physical mapping. The stream-surface comparison explicitly labels
both coordinate systems; its two surfaces are not an equal-size comparison.
No profile is presented as a physical plot using dimensionless chart angles.

The reference uses **601 base stations plus additional edge/shoulder stations**.
The resolution study uses its actual requested samples. No cosmetic spline
or plotting simplification is applied. Edge-detail figures make small rounding
and relief effects visible while retaining the same three-column layout.
Mean-radius figures show a neighbor in the blade panel at fixed blade count;
domain-extension figures show production domain boundaries around the blade.
Span and numerical effects are explained in the figure notes and manifest.
Identical overlays are labelled when the shape law is unchanged.

The production validator rejects invalid inputs and sampled self/neighbor
intersections. Rejections appear beside the relevant figure and in the
manifest, with their exact error; those cases are not drawn. Kernel cautions
are reported separately from rejections. These checks are resolution dependent.

## Baselines and the application

The reference dimensions and metal angles come from the standalone
`reaction.yaml` and `impulse.yaml` cases in
`turbodash/geometry/blade_parametrization/cases/`. Shape controls come from the
**current production** reaction/impulse preset dictionaries in
`turbodash/geometry/meanline.py`.

The reaction baseline has **0 leading/trailing fullness and 0.15 leading
relief**; its turning location/concentration are **0.52/1.2**, with leading and
trailing rounding fractions **0.1/0.025**. The impulse baseline has
**−0.25 leading/trailing fullness and zero leading relief**; its turning
location/concentration are **0.5/1.8753**, with both rounding fractions **0.005**.
Both use body exponent **2**, edge rounding exponents **3**, automatic leading
pressure-side inference and zero trailing relief. The impulse reference YAML
still contains zero fullness; the study explicitly overlays the app preset and
records that adjustment without modifying the reference file.

The integrated app inherits **dimensions, blade count, thickness, edge
dimensions and metal angles from Meanline**. A meanline-supplied thickness
location also takes precedence. Shape presets interpolate using the clipped
physical angle ratio `−βin/βout`; an undefined ratio falls back to reaction with
a diagnostic. Leading relief is disabled on initialization for a zero-turning
row because its automatic pressure side is undefined. The study's standalone
dimensions and edge/body ratios are not copied into an integrated row, so its
outline will generally differ from the examples here.

### Thickness, rounding and radius are different quantities

- **Nominal maximum thickness ratio** sets a full body value at a nominal
  conformal station. The final physical maximum can have a different value
  and location after rounding, relief and mapping.
- **Unrounded endpoint thickness ratios** set body shoulder scales before the
  tip is closed. They are not finite gaps at a rounded tip.
- **Rounding fractions and exponents** control closure extent and shape. A
  positive fraction closes the tip; zero leaves a straight closing cap.
- **Tip radius** is a local property of the finished curve. No current GUI
  control directly prescribes an exact circular radius or wedge angle.
  Meanline's leading radius is converted to a local rounded-tip target; on a
  radial mapping it is not an exact final-radius guarantee.
- **Actual physical thickness** is measured on the finished, mapped paired
  samples. Radial correction aims to preserve the intended physical scale
  approximately and reports residual error. It does not alter nominal inputs.

These illustrations describe **geometry**, not aerodynamic loading, loss,
efficiency, throat capacity or CFD performance. The application does not rerun
the aerodynamic solution when a geometry control is edited.

## Regenerate and inspect

From the repository root, with the project dependencies and mesh extra installed:

```powershell
poetry run python development/blade_parametrization/parameter_sensitivity/study.py
poetry run python -m pytest tests/test_blade_sensitivity_study.py
poetry run python docs/build_docs.py --no-autobuild
```

An existing project environment can use `.\.venv\Scripts\python.exe` in place
of `poetry run python`. The study regenerates every PNG, SVG, baseline table,
sweep description and the machine-readable manifest. The mesh extra supplies
the same TurboMoC passage construction used by the app; no CAD or mesh is built.

For an individual parameter preview:

```powershell
poetry run python development/blade_parametrization/parameter_sensitivity/study.py --only camberline.turning_concentration --output-dir development/blade_parametrization/output_parameter_preview
```

This writes a preview manifest without replacing the full gallery or its
manifest. Implementation details and the artifact contract are in
`development/blade_parametrization/parameter_sensitivity/README.md`.

{download}`Download the complete study manifest (JSON) <../_static/blade_sensitivity/manifest.json>`

```{include} _blade_geometry_sweeps.inc
```
