# Implemented workbench specification

This describes the final design implemented in the optimization edition, based on the [approved decisions](approved_feedback.json). The [implementation notes](implemented/README.md) contain actual-app screenshots and recorded verification. Other app editions retain their own layouts; shared plotting and fluid helpers remain reusable.

## Themes and identity

Blueprint light uses a pale inspector and a quiet results canvas. Graphite dark uses restrained blue accents. The initial theme follows the operating system; an explicit Light/Dark selection persists in the browser. Theme changes restyle controls, plots, reports and dialogs without changing model values or rerunning the solver, and preserve manual plot zoom.

These are the current tokens in [workbench.css](../../turbodash/apps/app_turbine_optimization/assets/workbench.css):

| Role | Blueprint light | Graphite dark |
| --- | --- | --- |
| Canvas | `#EFF3F8` | `#111922` |
| Panel | `#FFFFFF` | `#19242F` |
| Inspector | `#F8FAFD` | `#17222E` |
| Input | `#FFFFFF` | `#111D29` |
| Main text | `#18314D` | `#E7EFF9` |
| Secondary text | `#62758B` | `#ADBDCE` |
| Border | `#D9E2EE` | `#344558` |
| Accent | `#2462D1` | `#88B7FF` |
| Selected surface | `#EAF1FC` | `#253950` |

The interface uses the system sans-serif stack headed by Segoe UI, with tabular numerals and monospaced technical output. The approved blade-ring mark accompanies the lowercase `turbodash` wordmark. Clicking the brand returns to Meanline while preserving the design; GitHub has a separate link. [All three logo families](logos/README.md) remain available, including the unused cascade and monogram alternatives. Production uses its own blade-ring assets and does not load files from this folder.

The browser favicon uses the same blade-ring SVG, with light/dark variants selected by the browser's preferred color scheme. GitHub and Documentation both show the ↗ external-link indicator and open in a new tab.

## Shell and editor

The header contains Load YAML, Save design and Documentation beside GitHub, with the Light/Dark selector at the far right. A labelled left rail selects Meanline, Geometry or Mesh. The desktop inspector starts at 348 px beside a 92 px rail; it can be resized with the pointer or keyboard and scrolls independently of the results. Narrow screens use a dismissible inspector drawer and stacked plots.

Start optimization and its live status occupy the bottom of the left inspector, below the independently scrolling editor. The action remains visible with sections collapsed while Meanline is selected. Geometry and Mesh hide this action. On narrow screens it appears inside the open parameter drawer and hides when the drawer closes. The collapsible optimization terminal sits at the bottom of the Meanline results column and opens when a run starts.

Explicit outcome states label Ready, Running, Applying design, Applied successfully, Design changed, Application unconfirmed and Optimization failed. Success uses green only after both the inspector values and displayed results match the optimized design. Failures use red; changed inputs and unconfirmed application use amber. Running and applying disable the action and show the corresponding button label. Session ownership and protection against overwriting newer edits remain in place.

Documentation beside GitHub opens the Sphinx site served at `/docs/` in a new browser tab. The calculation tab retains its exact inputs and results without navigation or a new solve. There is no embedded documentation view or Back to calculations button. The published theory and analytical-case pages reproduce the original repository notes verbatim; the documentation fidelity script checks all five source copies. See the [documentation build and hosting guide](../../docs/README.md).

Five numbered sections start closed on a fresh visit or newly loaded design. Opening one does not close another. Collapse all closes them together; ordinary edits preserve disclosure state.

| Section | Controls and scope |
| --- | --- |
| **1. Turbine setup** | Axial/radial type, integer stage count with add/remove controls, and loss model. |
| **2. Operating point** | Searchable working fluid, readable inlet property pair and both values, exit pressure, mass flow and inlet flow angle. Labels and units follow the selected properties. |
| **3. Shaft speed** | Specific/actual speed mode, the corresponding value and units, and the global blade velocity ratio. |
| **4. Stage design & geometry** | Named stage selection, work split, angles, reaction, velocity/radius ratios and blade geometry, with explicit **Stator / Rotor** scope. Axial/radial applicability is retained. |
| **5. Optimization** | Configuration, Design variables and Constraints, including objective, solver settings, precise bounds and targets, background job status, convergence and successful-result application. |

All stage controls stay mounted while inactive editors are hidden. Stage/row selection changes the editing context without losing values. Adding/removing stages preserves retained-stage values and names. Changing turbine type initializes stage parameters from the selected type's defaults while retaining stage count and names.

Aerodynamic design and Blade geometry expand inline within the Stage design & geometry section. Their headings and controls use a compact shared alignment inside the parent section. Optimization tabs and their editors are similarly contained within the numbered Optimization section.

Find a parameter searches names, aliases, symbols and stage/row/workspace context, including collapsed controls. Choosing a match selects its workspace and scope, opens the containing sections, scrolls to the field, focuses it and briefly highlights it. Keyboard controls include Ctrl/Cmd+K, arrows, Enter and Escape. Search and navigation do not edit the model.

## Optimization selection and presets

The Configuration tab contains the objective, goal and solver controls. Design-variable rows put Lower, Initial and Upper in three aligned columns. Constraint rows use the same compact layout for Comparison, Target and Scale. Field labels provide contextual help. A blank initial value uses the current design; values outside the bounds are clipped and logged.

The Add variable, Add constraint and Add groups controls open searchable hierarchical checkbox menus. Overall quantities, all-stage selections and named stages form the hierarchy; stage branches expose the applicable inputs, stator/rotor geometry and flow stations. Search includes the quantity label and its scope. Checking a leaf adds its editable row; unchecking removes it. Existing row identities, bounds, targets and scales survive other selections and catalog refreshes.

Groups add only missing members, so overlaps do not create duplicate variables or duplicate constraints with the same quantity and comparison. Their checkboxes show full or partial selection based on the current rows. Unchecking a group removes its matching members even if another preset also contains them; presets do not retain separate ownership of shared members. Individual constraint leaves represent all bounds on their quantity, and unchecking one removes those bounds. Opposite comparisons remain separate constraint rows, allowing two-sided limits.

Variable presets cover shaft speed and blade velocity ratio, reaction and stator angle for one or all stages, work split, radial passage radii, meridional velocity ratios and applicable blade geometry for one row or all stages. Optimize all applicable variables combines effective continuous inputs while holding the operating conditions fixed. The speed variable follows the current specific-speed or rpm representation. Radial blade-row bounds remain on the current side of a radius ratio of one, preserving radial direction and avoiding zero radial chord.

Work-split presets vary the first N−1 relative weights and keep the final stage weight fixed, removing the redundant common scaling of normalized weights. Their bounds scale with the current weight. All individual stage weights remain available for custom selections. The first stage's unused inlet radius ratio is absent; later stages retain that radius link. Automatic presets omit geometry inputs that cannot affect supported results for the selected model. Loss-sensitive maximum thickness is available where applicable; display-only shape parameters and zero-incidence metal geometry are excluded.

Constraint presets provide editable limits of maximum blade-row Mach number 1, flaring between −10° and +10°, and mean blade height at least 1 mm, globally or for a named stage. Mach uses absolute velocity in stators and relative velocity in rotors. Two flaring rows enforce both signs. These are starting choices, not universal design requirements.

Target uses the quantity's displayed units; efficiency quantities labelled `[-]` use fractions. Scale must be positive and divides the physical constraint residual before applying the dimensionless feasibility tolerance. The standard Mach, flaring and height scales are 1, 10° and 0.01 m respectively, including a positive scale for a zero target. Reapplying a preset preserves edited targets and scales.

## Numeric values, state and files

The inspector uses one numeric field per row, without sliders. Continuous fields use `step="any"` and committed/debounced changes. Display formatting pads to at least three decimal places while preserving additional supplied digits, trailing zeros and useful scientific notation. Formatting leaves stored precision and active typing intact; discrete counts remain integers. Blank, incomplete and invalid drafts remain visible rather than silently reverting to defaults.

Inlet-pair conversion uses the **current canonical pair and current input values**, including edits whose calculation is still pending. The property backend validates the equivalent state, and the pair and both values commit together. If conversion is impossible, both new inputs clear and require explicit entry. Labels and units follow the committed values. Shaft-speed representation changes likewise convert value and mode together.

Committed valid inputs update the meanline calculation automatically. Invalid drafts retain the last successful results with a visible stale/error state and block stale saves. Load/edit/save/reload preserves the existing YAML model schema, stage names and optimization settings, including precise bounds and constraints. Pending loads, model changes and optimization application reconcile revisions so delayed responses cannot overwrite newer edits. Background optimization retains session isolation.

## Results and geometry

Four overview cards use one coherent successful calculation:

| Metric | Result field | Display |
| --- | --- | --- |
| Total-to-static efficiency | `overall_performance.efficiency_ts` | Percent, two decimals. |
| Total-to-total efficiency | `overall_performance.efficiency_tt` | Percent, two decimals. |
| Shaft speed | `overall_performance.rotational_speed` | rpm, whole number. |
| Power | `overall_performance.power_actual` | Signed actual shaft power, two decimals, adaptive W/kW/MW. |

Power units follow magnitude, including promotion when rounding would display 1,000.00 of the smaller unit. Missing/nonfinite metrics display an em dash. Full Performance, Geometry and Flow stations reports remain available in both themes; the cards supplement them.

Three initially open plot disclosures can be collapsed independently:

- **Turbine geometry** pairs the meridional channel and blade-to-blade view.
- **Efficiency and loss distribution** pairs efficiency versus blade velocity ratio with the loss budget directly below geometry.
- **Velocity triangles** shows one full-width stage plot per row, with a Mach numbers / Velocity [m/s] selector.

The paired plots stack on narrow screens. Below 600 px, efficiency, loss and triangle figures keep a 540 px minimum width and scroll horizontally within their cards, preserving readable axes and legends without page overflow. Performance reports have a separate initially open disclosure. Closing a group keeps its controls and results mounted; reopening resizes its plots without a numerical calculation.

Hub and shroud lines are solid. Meridional and blade-to-blade plots have no legends; hover labels identify their geometry. Efficiency, loss and triangle plots retain their legends. The efficiency curves use the app's blue/orange palette and show efficiency in percent. They are kinematic sweeps at fixed work split and loss coefficients, not thermodynamic off-design calculations.

The loss distribution reports efficiency penalties on a percent scale, labelled **Δη [%]**, for every stator and rotor, followed by the final exit kinetic-energy column. Each mechanism's loss coefficient is multiplied by its row's exit kinetic energy, divided by the turbine's overall isentropic enthalpy drop, and multiplied by 100. The displayed value is a percentage-point contribution to the efficiency deficit. Stators use absolute exit speed and rotors use relative exit speed. Profile, trailing-edge, secondary, tip-clearance and incidence contributions retain the model's existing weighting; any difference from a row's total appears explicitly as Other / unallocated. Only the final rotor's absolute exit kinetic energy enters the exit column. Summing the budget gives `100 * (1 - efficiency_ts)`. Hover text identifies the percentage-point penalty and corresponding specific energy in J/kg.

Loss components use the app's blue and orange palette, with muted gray for the final exit contribution. Colors adapt to the light and dark themes while matching their legend entries.

Velocity triangles use arrowheads on absolute, relative and blade-velocity vectors, with distinct rotor-inlet and rotor-outlet colors. Stages share axis extents for comparison. Switching between Mach numbers and m/s redraws the triangle figures from the current result without rerunning the turbine calculation.

Plots retain Plotly zoom, pan, reset and image export while hiding the Plotly logo. Theme changes preserve the current view. Initial/reset/autoscale meridional views include the zero radial baseline; deliberate manual radial zoom remains available.

The meridional channel uses calculated station-derived hub/shroud boundaries, connected row/stage gaps and constant-section inlet/outlet display extensions nominally 1.5 adjacent-row chords long. Radial extensions are limited to valid positive radii. Five smooth guide streamlines remain inside the channel and are identified as illustrative geometry paths rather than a CFD solution. Stator/rotor regions and blade profiles use orange/blue outlines and matching translucent fills.

Single/multistage axial and radial designs retain their coordinate conventions and physical aspect ratio. Axial channel and blade plots share row placement and horizontal limits. Limits consider all stations so upstream shroud geometry is not clipped. Extensions and guides affect presentation only. The meanline's flared channel remains distinct from the cylindrical averaged row radii used by the existing mesh adapter.

Geometry and Mesh retain their existing scoped controls, generation/inspection state and applicable exports. Optimization and mesh generation are explicit actions. Recorded verification did not exercise native mesh generation because the optional TurboMoC dependency was unavailable.

## Source and verification

| Responsibility | Source |
| --- | --- |
| Shell, scoped editor, numeric controls and metric cards | [workbench.py](../../turbodash/apps/app_turbine_optimization/workbench.py) |
| Theme, search, resizing and numeric presentation | [workbench.css](../../turbodash/apps/app_turbine_optimization/assets/workbench.css), [workbench.js](../../turbodash/apps/app_turbine_optimization/assets/workbench.js) |
| Live calculations, state transitions and YAML | [app.py](../../turbodash/apps/app_turbine_optimization/app.py) |
| Optimization editor, selection presets and jobs | [panel.py](../../turbodash/apps/app_turbine_optimization/panel.py), [selection.py](../../turbodash/apps/app_turbine_optimization/selection.py), [model.py](../../turbodash/apps/app_turbine_optimization/model.py), [jobs.py](../../turbodash/apps/app_turbine_optimization/jobs.py) |
| Fluid labels and validated conversion | [fluid_inputs.py](../../turbodash/apps/shared/fluid_inputs.py) |
| Geometry plots, efficiency budgets and velocity triangles | [plotly_turbine.py](../../turbodash/visualization/plotly_turbine.py) |
| Mesh and TurboGrid workflows | [meshing.py](../../turbodash/apps/shared/components/meshing.py), [turbogrid.py](../../turbodash/apps/shared/components/turbogrid.py) |
| Sphinx site and same-app serving | [docs](../../docs/README.md), [documentation.py](../../turbodash/apps/shared/documentation.py) |

See the [recorded verification and refresh instructions](implemented/README.md#recorded-verification) and [interaction fixes](implemented/interaction_fixes.md) for state, precision and browser coverage. The verification notes distinguish the current interface checks from earlier implementation evidence.
