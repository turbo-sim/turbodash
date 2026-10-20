# Interaction stress-test follow-up

This report preserves the original implementation follow-up and records the
latest interaction changes separately. The optimization-editor and model/plot
stress reports were refreshed for the compact sidebar; the stage/type report
is retained from the earlier run. See the
[verification notes](README.md#recorded-verification).

The optimization edition was exercised through its actual browser controls,
including rapid actions, delayed requests and out-of-order responses. The tests
compare rendered Plotly coordinate arrays with the calculated figures, rather
than treating a successful server calculation as proof that the view updated.

## Reproduced faults and fixes

| Interaction | Fault | Correction |
| --- | --- | --- |
| Add/remove a stage or change axial/radial type | Dash rejected a scalar `no_update` for a wildcard output, returning HTTP 500. Direct Python callback tests had missed the serialization failure. | Return one unchanged entry for every matched overall control; cover the actual HTTP dispatch. |
| Click the stage stepper rapidly | Overlapping requests used the same earlier stage list and lost clicks. | Accumulate the requested count in the browser and apply that absolute count. |
| Edit a stage value or add a stage during a pending update/load | A stage rebuild could restore an older number, or use the previous design as the basis for an addition during YAML loading. | Merge current retained-stage values when committing an addition; keep pending loads based on the new YAML until its revision is acknowledged. |
| Change stage count, turbine type or unit selectors while another change is pending | A later callback could omit a still-pending change, or an older response could restore the previous stage values. | Reconcile all pending selections and commit a response only when it still matches the current request. |
| Select/remove optimization variables or constraints rapidly | Overlapping editor requests could lose actions or let a catalog refresh take priority over a removal. | Replay an ordered selection journal, with leaf and group identities and stable editor row IDs; keep Add menus above the lists. |
| Edit an optimization bound while an older editor response is pending | The older rendered editor could restore the previous bound. | Committed numeric/comparison edits trigger a newer reconciliation that supersedes the older response; stable row keys retain surviving controls. |
| Select a different optimization variable and immediately edit a bound | The pending variable change could replace the newly entered bound with its default. | Keep explicitly edited fields while resetting untouched fields for the new variable. |
| Select an optimization quantity while YAML is loading | Replacing the editor with loaded settings could discard a later selection. Reloading a design with the same turbine inputs could retain old bounds. | Give each load its own revision, capture the selection-journal baseline when loading begins and replay later operations against the loaded configuration. |
| Correct an invalid draft while its earlier response is delayed | An old status side effect could mark the corrected design stale. Conversely, an old successful response could hide a newer validation warning. | Return status and validity as declared outputs of the same callback as the plots, so Dash orders them together. |
| Change an inlet property pair before the preceding solve finishes | Conversion used the last calculated inlet and discarded a newer pressure/temperature edit. | Convert the current canonical pair and current values; invalid inlet drafts clear both new inputs. |
| Change shaft-speed representation after editing its value | Conversion could reuse an older calculated speed or reinterpret a number during overlapping mode changes. | Separate the visible selector from the canonical mode, then commit the converted value and mode together. |
| Change units or reopen a menu while a load is pending | Labels could lag behind converted values, and delayed load responses could close a reopened menu. | Update labels synchronously from canonical values and reset menus once when loading begins. |

The existing type-change policy still initializes stage parameters from the
selected turbine type's defaults while retaining stage count and names. Ordinary
stage additions/removals preserve the retained stages' current input values.

## Current interface follow-up — 20 October 2026

- Expanded sidebar cards contain their fields; ordinary labels and controls
  share a row. Optimization rows use one quantity header and three compact
  value columns, with hover and focus help.
- Add menus are hierarchical, searchable checklists. Group overlap preserves
  existing bounds and targets; partial selections show a mixed checkbox state.
  Opening a menu does not rebuild the editor rows. Escape restores toggle focus.
- Menus opened near the inspector's bottom scroll into view above the fixed
  optimization action. Their height accounts for the sticky section header.
  Help tooltips remain readable without intercepting clicks on nearby controls.
- Geometry, efficiency/loss and velocity-triangle plots have separate
  disclosures. Each retains its mounted figures when collapsed. Triangle mode
  changes redraw the current result in Mach numbers or m/s without a new solve;
  arrowheads identify the vector directions.
- The loss chart uses percentage-point efficiency penalties, Δη [%], and one
  final-exit kinetic-energy column. Its components sum to
  `100 * (1 - efficiency_ts)`, including explicit unallocated row contributions.
  Bars use the app's blue/orange palette and muted gray in both themes; hover
  text explains the percentage-point penalty.
- Documentation opens from the header in a new browser tab. The calculation
  remains in its original tab with its current values, results and editor state.
  Published theory and analytical pages include the original notes verbatim,
  checked by the documentation source-fidelity script.
- The optimization action stays below the scrolling Meanline inspector and
  hides in Geometry and Mesh. On mobile it follows the inspector drawer.
  The terminal remains collapsible at the bottom of the results column.
- Job outcomes are explicit state values. A completed worker first enters
  Applying design; green success appears only after both current inputs and
  displayed results match its target. Changed designs and unconfirmed
  application retain the current controls and show an amber outcome.
  Failed starts, workers and infeasible results show red and allow retrying.

Current Python and browser verification is recorded in the
[implementation notes](README.md#recorded-verification): **170 Python tests
passed, with 1 skip**, and six refreshed browser reports record **103 checks**
with zero browser/callback errors. These include 19 editor stress checks,
11 model/plot interaction checks and 31 compact-layout checks. The latter
exercise both themes, desktop and mobile widths, a resized inspector and
menus near the bottom of the viewport.

## Reproducible checks

- `tests/browser_stage_stress.py`: actual stage/type controls, rapid stepper
  actions, YAML count reset, current-value conversions and invalid-draft recovery.
- `tests/browser_editor_stress.py`: additions/removals under simulated latency,
  row identity, precise bounds, path changes, focus retention and real optimization.
- `tests/browser_interaction_stress.py`: exact plotted coordinates, rapid numeric
  edits, theme/workspace switching, YAML consistency and reordered responses.
- `tests/browser_optimization.py`: real single/multistage and radial jobs, failed
  worker recovery, application of results to every control/trace, triangle-mode
  persistence and session isolation.

The earlier implementation run passed **574 Python tests**, with **8 skips**: seven required the
optional TurboMoC dependency, and one required Windows symlink privileges. Native
mesh generation was not exercised. The retained stress reports are:

| Suite | Passed groups | Report |
| --- | ---: | --- |
| Stage, type and units (earlier run) | 14 | [Results](stage_stress_checks.json) |
| Optimization editor (refreshed) | 19 | [Results](editor_stress_checks.json) |
| Model and plot interactions (refreshed) | 11 | [Results](interaction_stress_checks.json) |

Plot checks use the design accepted by Dash and match its signature to the
calculated figure before comparing rendered coordinates. The last network
response alone is not a reliable indicator when requests finish out of order.

The browser lifecycle and generated profiles, downloads, logs and temporary
worker files stay inside Turbodash. No other repository is modified.
