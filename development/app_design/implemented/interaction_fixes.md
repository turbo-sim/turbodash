# Interaction stress-test follow-up

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
| Add/remove optimization variables or constraints rapidly | Overlapping editor requests could lose actions or let a catalog refresh take priority over a removal. The Add buttons also moved as rows appeared. | Reconcile cumulative additions and removals by stable row identity; keep Add buttons above the lists. |
| Edit an optimization bound while an older editor response is pending | The older rendered editor could restore the previous bound. | Committed numeric/comparison edits trigger a newer reconciliation that supersedes the older response; stable row keys retain surviving controls. |
| Select a different optimization variable and immediately edit a bound | The pending variable change could replace the newly entered bound with its default. | Keep explicitly edited fields while resetting untouched fields for the new variable. |
| Add an optimization row while YAML is loading | The editor could consume the click counter while replacing rows with the loaded settings, leaving the new row absent. Reloading a design with the same turbine inputs could retain old bounds. | Give each load its own revision, capture click counters when loading begins and preserve additions made after that baseline. |
| Correct an invalid draft while its earlier response is delayed | An old status side effect could mark the corrected design stale. Conversely, an old successful response could hide a newer validation warning. | Return status and validity as declared outputs of the same callback as the plots, so Dash orders them together. |
| Change an inlet property pair before the preceding solve finishes | Conversion used the last calculated inlet and discarded a newer pressure/temperature edit. | Convert the current canonical pair and current values; invalid inlet drafts clear both new inputs. |
| Change shaft-speed representation after editing its value | Conversion could reuse an older calculated speed or reinterpret a number during overlapping mode changes. | Separate the visible selector from the canonical mode, then commit the converted value and mode together. |
| Change units or reopen a menu while a load is pending | Labels could lag behind converted values, and delayed load responses could close a reopened menu. | Update labels synchronously from canonical values and reset menus once when loading begins. |

The existing type-change policy still initializes stage parameters from the
selected turbine type's defaults while retaining stage count and names. Ordinary
stage additions/removals preserve the retained stages' current input values.

## Reproducible checks

- `tests/browser_stage_stress.py`: actual stage/type controls, rapid stepper
  actions, YAML count reset, current-value conversions and invalid-draft recovery.
- `tests/browser_editor_stress.py`: additions/removals under simulated latency,
  row identity, precise bounds, path changes, focus retention and real optimization.
- `tests/browser_interaction_stress.py`: exact plotted coordinates, rapid numeric
  edits, theme/workspace switching, YAML consistency and reordered responses.
- `tests/browser_optimization.py`: real single/multistage and radial jobs, failed
  worker recovery, application of results to every control/trace and session isolation.

The final source passed **574 Python tests**, with **8 skips**: seven need the
optional TurboMoC dependency, and one needs Windows symlink privileges. Native
mesh generation was not exercised. Final browser runs passed **56 groups** with
zero JavaScript or callback errors:

| Suite | Passed groups | Report |
| --- | ---: | --- |
| Stage, type and units | 14 | [Results](stage_stress_checks.json) |
| Optimization editor | 17 | [Results](editor_stress_checks.json) |
| Model and plot interactions | 11 | [Results](interaction_stress_checks.json) |
| Optimization workflows | 14 | [Results](optimization_browser_checks.json) |

Plot checks use the design accepted by Dash and match its signature to the
calculated figure before comparing rendered coordinates. The last network
response alone is not a reliable indicator when requests finish out of order.

The browser lifecycle and generated profiles, downloads, logs and temporary
worker files stay inside Turbodash. No other repository is modified.
