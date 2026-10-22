# One-at-a-time blade parameter study

The primary gallery explains inputs that change the **2D blade profile**. A
separate appendix covers spacing, span, surrounding domains and numerical
diagnostics, which should not be mistaken for section-shape controls.

From the repository root, in the project environment:

```powershell
poetry install --with dev -E mesh
poetry run python development/blade_parametrization/parameter_sensitivity/study.py
poetry run python -m pytest tests/test_blade_sensitivity_study.py
poetry run python docs/build_docs.py --no-autobuild
```

An existing environment can run the same commands with its Python executable,
for example `.\.venv\Scripts\python.exe`. The mesh extra provides the production
TurboMoC passage backend used for domain extensions. The study does not build
CAD or a mesh. It deliberately fails if that backend is unavailable rather
than substituting its own domain equations.

The full command recreates:

- `docs/source/_static/blade_sensitivity/*.png` and `*.svg` (separate reaction
  and impulse figures, each with one row and three columns: camberline,
  half paired thickness, and blade profiles). Edge details use the same layout.
- `docs/source/_static/blade_sensitivity/manifest.json` (SI independent inputs,
  exact baseline/context definitions, requested and resolved cases, validation
  stage/errors, warnings, production diagnostics, source/coordinate/artifact
  hashes, environment versions and physical-axis checks).
- `docs/source/guides/_blade_geometry_sweeps.inc` (baseline tables, GUI index,
  values, explanations and rejected-case reports included by the guide).

Assets live outside the repository's ignored `figures/` and `results/`
directories so the documentation deliverables can be tracked normally.
Figure limits come from all accepted curves in a panel; they never change
between its overlays. The study calls TurboDash's `set_plot_options()` and
turns on all four spines of every panel. Variation lines use 0.85 pt and the
dashed baseline 1.15 pt. The four colors sample `plasma` at equally spaced
positions from 0.25 to 0.75. Colors match across all panels and edge details
for a given family. The manifest records colors, positions, boxes and layout.

Each continuous sweep has four values including the exact baseline. Most
scale inputs use 90%, 95%, 100% and 110%; rounding fractions and domain
extensions use 80%, 90%, 100% and 120%. Angles use increments -5°, -2.5°, 0°,
+5°; locations use -0.05, -0.025, 0, +0.05; fullness uses -0.1, -0.05, 0, +0.1.
The zero-relief impulse baseline uses the explicit one-sided range 0-0.05.
Categorical and Boolean inputs show their two available choices.

Blade count is never swept: reaction retains 24 and impulse retains 12 in
every axial and radial case. Coverage records it as an explicitly fixed GUI
input. A full rebuild removes obsolete PNG/SVG files listed in the previous
manifest after the replacement figures have been generated successfully.

For a quick preview without replacing the full manifest or documentation:

```powershell
poetry run python development/blade_parametrization/parameter_sensitivity/study.py --only leading_edge.relief_strength thickness.leading_edge_fullness --output-dir development/blade_parametrization/output_parameter_preview
```

The case YAML files supply reference dimensions, angles and edge/body ratios.
The **current production** `REACTION_PRESET` and `IMPULSE_PRESET` supply all
shape defaults. In particular, the impulse YAML's zero fullness is replaced
with the app preset's -0.25/-0.25; the manifest records that difference without
editing the YAML. The reaction reference retains zero fullness and 0.15
leading relief. Radial contexts change only stream-surface type; the tolerance
context additionally enables correction **before** its sweep. These separate
baselines must not be mistaken for hidden changes within a sweep.

Each case starts from its baseline, applies exactly one independent override,
checks the full input difference, resolves parameters and calls production
`build_section` with intersection checks enabled. Rejected cases retain the
attempted inputs and exact exception, and have no plotted curve. No input is
adjusted to rescue a failure. Changes in pitch, arclength, physical thickness,
relief-window extent and actual thickness peak are dependent results.

The upper thickness curve is **half the total paired physical thickness**,
not a selected pressure/suction surface. Its horizontal coordinate is the
baseline physical camberline's sampled arclength at the same conformal station.
Consequently relief remains visible and camber/size changes do not silently
reparameterize the thickness abscissa. Radial physical mapping and paired
thickness come from the production kernel. The cylinder is developed in mm;
radial sections use machine X-Y coordinates in mm. Neither is a chart-angle
plot labelled as a physical profile.

Sampling remains at the reference 601 base points (plus edge/shoulder nodes)
except in the explicit resolution sweep. No cosmetic spline or line
simplification is used. The 2401-point sampling reference is a diagnostic
comparison only and does not replace the requested-resolution intersection
check. Sampled checks cannot prove absence of between-sample intersections.

The swept and explicitly fixed inputs together cover `FIELD_SPECS` minus the
GUI's actual `HIDDEN_PATHS`. Retired trailing relief and explicit pressure-side
selection remain fixed in the complete baseline record, not advertised as
current controls. Stream-surface type is covered but is read-only in the row
editor; it is changed in Meanline. View colors, visibility, displayed copies,
crosscut visibility, import/export and reset are not independent shape inputs.

The study and documentation do not change the application, solver or presets.
These are geometric sensitivities, not aerodynamic performance predictions.
