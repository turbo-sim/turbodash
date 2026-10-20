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
- Documentation beside GitHub opens the served Sphinx site in a new browser tab. The calculation tab retains its exact inputs and results.
- Three initially open plot groups separate Turbine geometry, Efficiency and loss distribution, and Velocity triangles. Performance reports collapse independently.
- Start optimization and its labelled status stay at the bottom of the left inspector while Meanline is selected. On narrow screens they appear inside the open parameter drawer. Geometry and Mesh hide the action.
- The optimization terminal remains collapsible at the bottom of the Meanline results column. It opens when a run starts.
- Outcome states distinguish Ready, Running, Applying design, Applied successfully, Design changed, Application unconfirmed and Optimization failed. Success turns green only after application is confirmed; failures turn red and changed/unconfirmed application uses amber.
- Five numbered, initially closed sections with named stage selection and **Stator / Rotor** scope. Every stage's input stays mounted to preserve exact edits.
- Expanded sidebar sections have a continuous border and tinted header. Labels and controls share a row; long labels wrap within their column.
- Optimization uses **Configuration**, **Design variables** and **Constraints** tabs. Each selected quantity has one name row and one row containing its three editable values.
- Add menus provide searchable, hierarchical checkboxes for Overall, stages, blade rows and flow stations. Group presets support partial selection, preserve existing numeric edits and avoid duplicate entries when groups overlap.
- Variable presets cover speed, stage aerodynamics, work split, radial radii, meridional velocity ratios and applicable blade geometry. Optimize all keeps the final normalized work weight fixed. Constraint presets provide editable Mach, two-sided flaring and minimum-height limits.
- Lower/Initial/Upper and Comparison/Target/Scale labels include hover and keyboard-focus help. Target uses the displayed quantity's units; Scale is a positive divisor for the residual.
- One continuous numeric field per row, with `step="any"`, debounced changes and no sliders. Continuous values display at least three decimal places, retaining extra supplied precision; formatting never rounds the model or interrupts typing. Inherently discrete counts remain integers.
- Keyboard typeahead search that selects the right workspace/stage/row, expands the containing sections and focuses the field. Desktop inspector resizing and a narrow-screen drawer are included.
- TurboMoC-inspired fluid/property labels with an atomic conversion of the current inlet values. Impossible transitions clear both new inputs; invalid drafts retain the last valid results and prevent stale saves.
- Four overview metrics and the complete performance, geometry and flow reports, styled in both themes. Power uses two decimals and adaptive W/kW/MW units.
- Current design and optimization editor settings are saved together, including precise bounds, constraints and stage names.
- Delayed optimization application never restores older inputs over a newer edit; an unconfirmed update keeps the current controls and reports its status.

## Plot refinement

The latest visual feedback is incorporated in both paired views. The meridional channel shows continuous station-derived hub/shroud boundaries and upstream/downstream constant-section display extensions, nominally 1.5 adjacent-row chords long and limited to valid positive radii for radial designs. Five smooth guide streamlines replace the arrows. They remain inside the passage; hover help identifies them as illustrative geometry paths, not a CFD velocity solution. Stator and rotor profiles use orange and blue outlines with matching translucent fills.

Single/multistage axial and radial designs are supported. Axial channel and blade plots share row placement and horizontal limits. Physical aspect ratio is preserved, all station extents are considered, and reset/autoscale includes the zero radial baseline. Manual zoom remains available. Extensions and guide curves do not change solver geometry or generated meshes. The existing cylindrical mesh model remains distinct from the flared meanline passage.

Hub and shroud use solid lines. The meridional and blade-to-blade views hide their legends while keeping hover identities; efficiency and loss legends remain visible.

Efficiency and loss plots sit directly below the geometry pair. Efficiency
curves use the app's blue/orange palette and a percent axis. The loss chart
expresses each row's profile, trailing-edge, secondary, tip-clearance and incidence
losses as efficiency penalties on a percent scale, Δη [%], using the overall
isentropic enthalpy drop and multiplying the normalized contributions by 100.
Stator components use absolute exit kinetic energy and rotor components use
relative exit kinetic energy. Any unallocated remainder is shown explicitly.
A separate final-exit column includes the last rotor's absolute exit kinetic
energy once; the complete budget sums to `100 * (1 - efficiency_ts)`.
Hover text explains the percentage-point penalty and its specific energy in
J/kg. The bars use the app's blues and oranges, with muted gray for final exit
kinetic energy and colors adapted to both themes.

The third plot group presents one velocity triangle per stage in a vertical
stack. Absolute, relative and blade velocities have arrowheads. A Mach numbers /
Velocity [m/s] selector redraws the triangles from the current calculation,
without rerunning the numerical model.

Below 600 px, efficiency, loss and triangle figures retain a 540 px minimum
width and scroll within their cards. This keeps axes and legends readable
without horizontal page overflow.

## Actual-app screenshots

- [Light, collapsed sections](workbench_light.png)
- [Dark, expanded editor](workbench_dark.png)
- [Stage geometry](workbench_stage.png)
- [Two stages, light](workbench_multistage_light.png)
- [Two stages, dark](workbench_multistage.png)
- [Narrow-screen inspector](workbench_mobile.png)
- [Documentation browser tab](workbench_documentation.png)
- [Efficiency penalties and performance plots](workbench_performance.png)
- [Arrowed velocity triangles](workbench_velocity_triangles.png)
- [Compact optimization variables](workbench_optimization.png)
- [Compact constraint editor](workbench_optimization_constraints.png)
- [Hierarchical optimization groups](workbench_optimization_groups.png)

These files are captured from the real Dash app. Their refresh status is recorded
below; the optional browser script can update them after future changes.

## Recorded verification

### Current compact sidebar and optimization update — 20 October 2026

The five focused Python suites ran together with **170 passed and 1 skipped**:
workbench 57, optimization 29, presets 29, plotting 35 and documentation serving
20. The skip requires Windows symlink privileges. All 29 optimization tests also
passed after the final menu-opening and empty-search-message changes.

The browser reports below were refreshed for the compact sidebar and hierarchical
selectors. They record zero browser/callback errors and include real successful
and failed optimization workers, precise numeric edits and YAML round trips.

| Suite | Passed checks | Report |
| --- | ---: | --- |
| Workbench workflows | 18 | [Results](browser_checks.json) |
| Optimization and result application | 15 | [Results](optimization_browser_checks.json) |
| Numeric presentation and precision | 9 | [Results](numeric_browser_checks.json) |
| Optimization editor under delayed updates | 19 | [Results](editor_stress_checks.json) |
| Model and plot interactions | 11 | [Results](interaction_stress_checks.json) |
| Compact layout, themes and menu access | 31 | [Results](compact_sidebar_browser_checks.json) |

Editor checks cover hierarchical leaf/group selection, partial group states,
rapid toggles, overlapping groups, exact bounds and targets, stable row identity,
delayed path changes and selections made during a YAML load. Current numeric
values survive rerendering without display rounding entering the model.

The compact layout checks cover both themes at desktop, 390 px and 320 px,
a 280 px desktop inspector, inline field alignment, contained expanded sections,
three-column variable/constraint values, keyboard-focus help and popup access
near the bottom of the scrolling inspector. Opening a menu leaves configured
rows unchanged; Escape closes it and returns focus to its toggle.

The workbench suite also verifies the three independent plot groups, geometry
legend/line settings, percentage-point loss budget and theme palette, cached
velocity-triangle modes, Meanline-only optimization action, mobile drawer and
green success only after the optimized design is applied. Documentation opens
in its own tab without changing the calculation. The screenshots above were
refreshed from the actual app; the new optimization and hierarchy views were
visually inspected.

The strict Sphinx build passed with the unchanged companion-project builder.
The five original notes, rendering wrappers and original SVG passed
[verify_source_fidelity.py](../../../docs/verify_source_fidelity.py); original
note text remains byte-identical. The updated calculations guide serves HTTP
200. The earlier documentation browser inspection confirmed all 245 mathematical
expressions rendered (100 display and 145 inline), with zero MathJax errors.
Those mathematical sources have not changed in this update. See the
[source mapping](../../../docs/README.md#original-source-mapping).

### Earlier implementation evidence

The [stage/type/unit stress report](stage_stress_checks.json) retains **14
checks** with zero errors from the earlier implementation run. It was not
rerun for this sidebar update. Its report does not identify a source commit or
run timestamp.

The recorded earlier full Python suite passed **574 tests with 8 skips**. Seven
required the optional TurboMoC dependency, and one required Windows symlink
privileges. Native mesh generation was not exercised. This is historical
coverage, not a full-suite result for the current working tree.

The [interaction follow-up](interaction_fixes.md) explains the state and precision
fixes. The retained reports come from separate browser runs and do not represent
one full regression invocation.

### Refreshing the evidence

Use an existing Playwright installation and Chrome/Chromium. See [tests/README.md](../../../tests/README.md) for browser and stress commands. Build the Sphinx documentation and run the app separately before refreshing the workbench screenshots and report:

```powershell
poetry run python docs/build_docs.py --no-autobuild
python tests/browser_workbench.py --url http://127.0.0.1:8050 --browser "C:/Program Files/Google/Chrome/Application/chrome.exe"
python tests/browser_compact_sidebar.py --url http://127.0.0.1:8050
```

`browser_workbench.py`, `browser_numeric_format.py` and `browser_compact_sidebar.py` write their retained reports directly here. The optimization, stage, editor and interaction scripts write `report.json` to their ignored output directories; after reviewing a successful run, copy it to the corresponding `*_checks.json` here. Scratch profiles, downloads and logs can be removed after the browser/server exits; scripts recreate them when needed.

The decision source is [approved_feedback.json](../approved_feedback.json); the delivered design is described in [design_specification.md](../design_specification.md). The [logo collection](../logos/README.md) retains all six SVG variants.
