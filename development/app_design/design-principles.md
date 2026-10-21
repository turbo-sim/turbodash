# Design principles and interaction contracts

This blueprint describes the **current optimization edition** of Turbodash in
[`app_turbine_optimization`](../../turbodash/apps/app_turbine_optimization/).
Its scientific workbench patterns can be reused in another package. The other
Turbodash app editions retain their own layouts.

Use this alongside the [blueprint index](README.md),
[visual style](visual-style.md), [components and layout](components-and-layout.md)
and [iconography](iconography.md). The specifications describe the intended
experience; the linked application source supplies its implementation.

## Product philosophy

The interface puts an editable engineering model beside its consequences. A
quiet results canvas gives geometry and plots room, while a compact inspector
keeps inputs organized by task and scope. Most valid input edits update results
automatically. Optimization and mesh generation remain deliberate actions.

The user should always be able to answer three questions: **What am I editing?
Which design produced these results? What action is currently running?** Named
stages, Stator/Rotor scope, explicit units, retained drafts and labelled outcome
states make those answers visible.

Use progressive disclosure to reduce visual load while keeping controls
discoverable. Numbered sections establish an order, nested sections remain
inside their parent, and expert settings appear on request. Parameter search
crosses collapsed sections and inactive scopes, so compactness does not require
the user to memorize where a field lives.

Preserve scientific precision and user intent. Formatting helps reading without
changing numbers, context switches preserve edits, and delayed responses must
respect the latest interaction. A successful worker is only one part of a
successful operation: the displayed inputs and results must also agree.

## Approved design decisions

These are the lasting decisions consolidated from the original D01–D13 feedback.
The rationale explains how each decision contributes to the workbench.

| Decision | Current choice | Rationale and reproduction rule |
| --- | --- | --- |
| D01 · Themes | Blueprint light; Graphite dark with restrained blue accents. | The light inspector stays pale; dark uses layered graphite surfaces. Keep the same hierarchy in both themes and reserve stronger blue for emphasis and selection. |
| D02 · Identity | Blade-ring symbol with a lowercase live-text `turbodash` wordmark. | A compact turbomachinery reference leaves room for the science. Preserve the approved vector silhouette and its theme variants. |
| D03 · Brand action | Clicking the symbol or name returns to Meanline and preserves the design. | The brand is a safe home action. GitHub has its own external link. |
| D04 · Workspace navigation | Labelled left rail; Light/Dark selector at the far right of the header. | Workspace changes stay separate from file actions and appearance settings, with a stable location for each. |
| D05 · Numeric interaction | One numeric field per row; no sliders in the default editor. | Exact engineering values are primary. Labels and units remain adjacent to their field. |
| D06 · Speed grouping | Blade velocity ratio belongs in Shaft speed. | Group controls by the relationship the user reasons about rather than by internal model storage. |
| D07 · Initial theme | Follow the operating system on first use; persist an explicit Light/Dark choice locally. | Provide a familiar initial appearance and respect the user's later choice. |
| D08 · Scope | Redesign the optimization edition first; reuse shared resources where useful. | Keep a coherent workbench while allowing plotting, fluid and styling patterns to serve other tools. |
| D09 · Inlet representation | Convert the current valid state when the property pair changes; otherwise request both new values. | Changing representation should preserve physical meaning. Never silently reinterpret an old number as a different property. |
| D10 · Meridional view | Show the zero radial baseline initially and after reset; allow manual radial zoom. | Provide a physically grounded default and preserve the user's ability to inspect details. |
| D11 · Optimization terminology | Name the objective and solver tab **Configuration**. | Use a direct task label alongside **Design variables** and **Constraints**. |
| D12 · Contained editors | Expand Stage design subsections inline; align optimization values in three columns. | Boundaries express ownership. Variable rows use Lower / Initial / Upper; constraint rows use Comparison / Target / Scale, with contextual help. |
| D13 · Quantity selection | Searchable hierarchical checkbox menus with editable presets. | Organize quantities by scope and add only missing rows. Reflect partial group selection, retain edits and respect axial/radial applicability. |

## Durable interaction contracts

### Navigation changes context without changing the design

- Keep Meanline, Geometry and Mesh as clearly labelled workspace destinations.
  Returning to Meanline through the brand preserves the current design.
- Keep named stage selection and Stator/Rotor selection explicit. Changing
  selection changes the editing context and preserves the other scopes' values.
- Start the five numbered inspector sections closed on a fresh visit or an
  explicit YAML load. Open them independently; **Collapse all** closes them
  together. Ordinary editing preserves disclosure state.
- Parameter search finds names, aliases, symbols and scope. Selecting a result
  selects the workspace/stage/row, opens its containing sections, scrolls to the
  field, focuses it and briefly highlights it. Search itself does not edit values.
- Open Documentation in a separate browser tab. The calculation tab retains
  its inputs, results and editor state.

### Numeric presentation preserves precision and drafts

- Use continuous numeric fields with `step="any"` and committed/debounced
  changes. Keep inherently discrete values, such as stage count, as integers.
- Display at least three decimal places where applicable while retaining extra
  supplied digits, deliberate trailing zeros and useful scientific notation.
  Presentation formatting must not round model values or trigger a solve.
- Preserve active typing, blank drafts, incomplete exponents and invalid values.
  Do not replace a draft with a default merely because it cannot yet be parsed.
- Keep the last successful plots visible when a draft is invalid, visibly mark
  them stale or erroneous, and block a save that would misrepresent those drafts.
  Correcting the draft resumes normal updates.
- Convert property-pair and shaft-speed representations from the **current
  canonical values**, including edits whose calculation is pending. Commit the
  representation and converted values together. If the inlet conversion is
  impossible, clear both new property inputs for explicit entry.

### The latest user intent wins

Revisions, stable row identity and ordered reconciliation are implementation
techniques for a user-facing rule: an older response must never restore a value,
selection or warning superseded by a newer interaction.

- Rapid stage-count clicks must accumulate. Adding/removing stages retains
  surviving stages' current values and names. Changing turbine type initializes
  its stage parameters from the selected type's defaults while retaining count
  and names.
- Pending YAML loads establish a new design revision. Later edits or selections
  apply to that loaded design; delayed pre-load responses cannot restore the old
  design. Reimporting a file must apply its settings even when turbine inputs
  happen to match the existing model.
- Add/remove selections, path changes, bound edits and catalog refreshes must
  reconcile without losing later actions. Keep surviving editor rows stable and
  preserve explicitly edited values through delayed defaults or refreshes.
- Validity, status and displayed plots must refer to the same accepted design.
  An older failure cannot mark a corrected input invalid; an older success
  cannot hide a newer validation warning.
- Background optimization belongs to its browser session. Applying a result
  must not overwrite a newer edit or another session's design.

### Selection menus preserve configured work

- Organize searchable checklists by Overall, named stage, blade row and flow
  station as applicable. A leaf checkbox adds/removes its corresponding rows.
  Opening a menu alone does not add or rebuild rows.
- Presets add missing members, preserve existing bounds/targets/scales and
  avoid duplicates. Full and partial checkbox states reflect the current rows.
- Shared preset members have no separate ownership: unchecking a group removes
  its matching members even when another preset includes them. Opposite
  constraint comparisons remain distinct rows for two-sided bounds.
- Menus must remain reachable near the bottom of the scrolling inspector. Size
  and scroll them within the available area above the action footer. Escape
  closes the menu and returns focus to its toggle.
- Provide help on hover and keyboard focus. Help must not intercept neighboring
  controls or require a pointer for access.

### Results and operations stay trustworthy

- The four overview metrics and all reports/plots use one coherent successful
  calculation. Missing or nonfinite metrics show an em dash.
- Appearance changes, plot-display unit changes, disclosure changes and
  workspace navigation must not rerun the numerical model. Retain manual plot
  zoom across theme changes and keep collapsed figures mounted.
- Keep the Meanline optimization action and labelled status below the scrolling
  inspector. On narrow screens they belong inside the parameter drawer. Hide
  the action in Geometry and Mesh. Open the collapsible results terminal when a
  run starts.
- Distinguish Ready, Running, Applying design, Applied successfully, Design
  changed, Application unconfirmed and Optimization failed. Disable duplicate
  starts while running/applying. Use green only when both displayed controls
  and results match the optimized design; use amber for changed/unconfirmed
  application and red for failure. Always include a text label with color.
- Save model values, stage names and optimization settings together in the
  established YAML schema. Reloading must restore precise bounds and constraints.

## Reuse in another scientific package

| Reusable pattern | Turbodash-specific expression to replace |
| --- | --- |
| Workspace rail, inspector, results canvas and explicit action footer. | Meanline / Geometry / Mesh and turbine optimization. |
| Numbered task sections, independent disclosure and nested ownership. | Turbine setup / Operating point / Shaft speed / Stage design / Optimization. |
| Scoped editing with persistent values and readable context. | Named turbine stages and Stator / Rotor. |
| Searchable quantity hierarchy and editable selection presets. | Aerodynamic inputs, blade geometry, flow stations and axial/radial applicability. |
| Precision-safe fields, atomic representation changes and revision-aware state. | Inlet thermodynamic property pairs and specific/actual shaft speed. |
| Coherent summary metrics, grouped plots and detailed reports. | Efficiency, shaft speed, power, meridional passage, loss budget and velocity triangles. |
| Sparse blue interaction accents with semantic outcome states. | Blade-ring identity and blue/orange turbine plotting conventions. |

Retain the reusable interaction contracts when adapting the palette, names or
domain quantities. Keep domain-specific meaning explicit: illustrative passage
guides are not CFD streamlines, efficiency sweeps are kinematic, and the loss
chart shows percentage-point efficiency penalties. The
[component specification](components-and-layout.md) records the relevant
Turbodash layout and presentation rules.

## Acceptance checks for a reconstruction

These are review criteria, not claims that a new implementation has passed them.

1. Compare light and dark views at desktop, 390 px and 320 px. Check contained
   expanded sections, readable labels, reachable menus, a dismissible parameter
   drawer and chart scrolling within cards without horizontal page overflow.
2. Use only the keyboard to navigate workspace and section controls, search for
   a hidden scoped field, select menu items, access help, close a popup and resize
   the desktop inspector. Focus should remain visible and predictable.
3. Enter long fractions, trailing zeros, a small scientific-notation tolerance,
   a blank and an incomplete exponent. Verify exact committed values, retained
   drafts, visible stale status and recovery after correction.
4. Change an inlet property pair and speed representation immediately after an
   edit. Confirm that physical state is preserved or the required inputs clear
   explicitly; labels and values must agree.
5. Add/remove stages and toggle scoped selections rapidly under delayed
   responses. Repeat edits while loading YAML and changing optimization rows.
   The final controls, saved design and plots must reflect the latest intent.
6. Select overlapping presets, edit their values, toggle a leaf and uncheck a
   group. Confirm mixed states, no duplicates and preservation of surviving edits.
7. Change theme, triangle-display units and plot disclosures after a manual
   zoom. Confirm coherent restyling and retained model values without a new solve.
8. Exercise optimization success, failure and an edit during application. Check
   action availability, explicit outcomes, retry, session isolation and green
   success only after matching controls and results are confirmed.
9. Save and reload a named multistage design with precise optimization settings.
   Open Documentation separately and return; calculation state should persist.

The maintained checks live in [`tests`](../../tests/README.md). Capture fresh
screenshots from the running application when changing this blueprint; record
their scope clearly instead of treating an older image or report as current
verification.
