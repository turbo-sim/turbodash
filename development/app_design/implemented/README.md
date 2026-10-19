# Implemented Turbodash workbench

The approved redesign is implemented in `turbodash/apps/app_turbine_optimization/`. Dash, the numerical backend, optimization jobs and existing YAML schemas are retained. Shared fluid and plotting helpers remain reusable; the other app editions retain their layouts.

Launch from the repository root:

```powershell
poetry run app_turbine_optimization --open-browser
```

The existing virtual environment can also run `.venv/Scripts/python.exe -m turbodash.apps.app_turbine_optimization --open-browser`.

## Applied decisions

- A pale Blueprint inspector and canvas for Light; Graphite with blue accents for Dark. The first visit follows the operating system; an explicit choice is saved locally.
- The approved blade-ring logo and typography. The brand returns to Meanline without resetting the design; GitHub is a separate link.
- A left workspace rail, Load YAML / Save design in the header, and the theme selector at the far right.
- Five numbered, initially closed sections with named stage selection and **Stator / Rotor** scope. Every stage's input stays mounted to preserve exact edits.
- One continuous numeric field per row, with `step="any"`, debounced changes and no sliders. Continuous values display at least three decimal places, retaining extra supplied precision; formatting never rounds the model or interrupts typing. Inherently discrete counts remain integers.
- Keyboard typeahead search that selects the right workspace/stage/row, expands the containing sections and focuses the field. Desktop inspector resizing and a narrow-screen drawer are included.
- TurboMoC-inspired fluid/property labels with an atomic conversion of the last accepted inlet state. Impossible transitions clear both new inputs; invalid drafts retain the last valid results and prevent stale saves.
- Four overview metrics and the complete performance, geometry and flow reports, styled in both themes. Power uses two decimals and adaptive W/kW/MW units.
- Current design and optimization editor settings are saved together, including precise bounds, constraints and stage names.
- Delayed optimization application never restores older inputs over a newer edit; an unconfirmed update keeps the current controls and reports its status.

## Plot refinement

The latest visual feedback is incorporated in both paired views. The meridional channel shows continuous station-derived hub/shroud boundaries and upstream/downstream constant-section display extensions, each 1.5 adjacent-row chords long. Five smooth guide streamlines replace the arrows. They remain inside the passage; hover help identifies them as illustrative geometry paths, not a CFD velocity solution. Stator and rotor profiles use orange and blue outlines with matching translucent fills.

Single/multistage axial and radial designs are supported. Axial channel and blade plots share row placement and horizontal limits. Physical aspect ratio is preserved, all station extents are considered, and reset/autoscale includes the zero radial baseline. Manual zoom remains available. Extensions and guide curves do not change solver geometry or generated meshes. The existing cylindrical mesh model remains distinct from the flared meanline passage.

## Actual-app screenshots

- [Light, collapsed sections](workbench_light.png)
- [Dark, expanded editor](workbench_dark.png)
- [Stage geometry](workbench_stage.png)
- [Two stages, light](workbench_multistage_light.png)
- [Two stages, dark](workbench_multistage.png)
- [Narrow-screen inspector](workbench_mobile.png)

These screenshots use calculated results from the real Dash app. They supersede the illustrative review images for implementation review. The optional browser script can refresh them after future changes.

## Verification

The full regression suite passed: **553 passed, 8 skipped**. Seven skips require
the optional TurboMoC package, which is not installed in this environment; one
requires Windows symlink privileges. Real mesh generation was therefore not
exercised here. The updated workbench passed **12 browser workflow groups** with
zero JavaScript or callback errors, including a real optimization worker. The
updated optimization browser regression also passed all **14 checks** with zero
browser/callback errors, covering successful and failed workers, exact multistage
application, actual RPM, radial zero-flaring and session isolation.

[browser_checks.json](browser_checks.json) records actual browser checks. `tests/browser_workbench.py` exercises the workbench against a running local app, including precise edits, YAML round trips, theme persistence, parameter search, property conversion, reports, responsive layout and a real background optimization run. Browser profiles, downloads and temporary output remain in ignored `output_browser/`.

[optimization_browser_checks.json](optimization_browser_checks.json) records the
separate `tests/browser_optimization.py` regression.

The numeric display follow-up passed **9 browser check groups** with zero errors;
[numeric_browser_checks.json](numeric_browser_checks.json) records the results.
`tests/browser_numeric_format.py` verifies three-decimal padding, extra significant
digits and trailing zeros, tiny exponents, uninterrupted focused editing, blank
and invalid drafts, dynamic editors and precise YAML round trips. Presentation
changes and unchanged focus/blur do not trigger a numerical calculation.

Regression coverage lives in `tests/test_app_workbench.py`, `test_app_fluid_inputs.py`, `test_mesh_workbench.py` and `test_plotting_turbine.py`, alongside the existing app, solver, optimization and mesh tests. Guide containment, real station placement, upstream extent clipping, radial extension limits, and theme changes that preserve zoom are covered directly.

The decision source is [approved_feedback.json](../approved_feedback.json); the complete resolved requirements remain in [implementation_prompt.md](../implementation_prompt.md).
