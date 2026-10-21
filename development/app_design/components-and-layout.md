# Components and layout blueprint

This specifies the current **turbine optimization workbench**, including its Meanline, Geometry and Mesh workspaces. It records implemented behavior and CSS dimensions, rather than a proposed redesign. Other TurboDash app editions have their own layouts. Start with the [blueprint index](README.md), then use the [visual style](visual-style.md), [design principles](design-principles.md), [iconography](iconography.md) and [implementation map](implementation-map.md) alongside this document.

The layout is built in [workbench.py](../../turbodash/apps/app_turbine_optimization/workbench.py). The final app styling comes from [workbench.css](../../turbodash/apps/app_turbine_optimization/assets/workbench.css), layered over Bootstrap, [optimization.css](../../turbodash/apps/app_turbine_optimization/assets/optimization.css) and the shared [workspace.css](../../turbodash/apps/shared/assets/workspace.css). Read the final cascade when rebuilding: standalone shared components and Python plot defaults do not always have the same dimensions and colors as the workbench.

## 1. Shell and reading order

The page fills the browser viewport. A horizontal header sits above three adjacent regions: workspace rail, design inspector, and results canvas.

```text
Header: blade-ring + turbodash | Load YAML | Save design | GitHub ↗ | Documentation ↗ | Light / Dark
┌──────────────┬───────────────────────────┬─────────────────────────────────────────┐
│ Workspace    │ Design inputs             │ Selected workspace results              │
│ rail         │ Search + Collapse all     │                                         │
│              │                           │                                         │
│ ◉ Meanline   │ Independently scrolling   │ Independently scrolling results         │
│ ◇ Geometry   │ numbered input sections   │                                         │
│ ▦ Mesh       │ + workspace controls      │                                         │
│              ├───────────────────────────┤                                         │
│              │ Optimization status       │                                         │
│              │ Start optimization        │                                         │
└──────────────┴───────────────────────────┴─────────────────────────────────────────┘
```

The optimization footer appears only in Meanline. On narrow screens the inspector and its footer move into a parameter drawer. The model editor is shared across workspaces; Geometry and Mesh add their own controls below it instead of creating another copy of the model inputs.

Use semantic `header`, `nav`, `aside`, `main` and `footer` landmarks. The root is a flex column, with a flex row for the shell beneath the header. Give shrinking children `min-width: 0` and scrolling flex children `min-height: 0` so long labels and plots cannot force page overflow.

| Element | Current desktop specification |
| --- | --- |
| Root | Width `100%`; height `100vh` with `100dvh` override; outer overflow hidden; border-box sizing throughout |
| Header | Minimum height `72px`; padding `12px 25px`; gap `20px`; panel background; bottom `1px` border; `z-index: 40` |
| Brand | Icon and lowercase wordmark; gap `10px`; transparent button; image `37 × 37px` |
| Header actions | Right aligned; gap `9px`; controls do not shrink below their content unnecessarily |
| Workspace rail | Width/flex basis `92px`; padding `17px 9px`; gap `10px`; right `1px` border; panel background |
| Rail button | Vertical icon/label; minimum height `68px`; padding `11px 4px`; gap `8px`; radius `8px`; transparent border until active |
| Inspector | Width/flex basis `348px`; pale inspector surface; right `1px` border; flex column |
| Inspector head | Padding `20px 14px 16px`; title row gap `12px`, bottom margin `17px` |
| Inspector scroll region | Flex `1`; overflow-y auto; overflow-x hidden; padding `0 12px 20px`; thin scrollbar with stable gutter |
| Inspector resize handle | Transparent `8px` wide target overlapping right edge by `4px`; full inspector height; `z-index: 30` |
| Results canvas | Remaining width; overflow-y auto; padding `26px 25px 45px`; thin scrollbar |

Clicking the brand returns to Meanline while preserving the design. Rail buttons always have text labels: **Meanline**, **Geometry**, **Mesh**. The active button has a tinted surface, visible border, accent text and a stronger weight. Inactive buttons are muted; hover adds a canvas-colored surface and stronger text. The icon glyphs are specified in [iconography](iconography.md).

Header action order is deliberate: **Load YAML**, **Save design**, **GitHub ↗**, **Documentation ↗**, then **Light / Dark**. File actions are bordered buttons; the project/documentation links are visually quieter. External links open a new tab, preserving the active calculation and editor state. Light/Dark forms a labelled button group with a visibly selected segment.

### Resizing and persistence

The desktop inspector can be resized by dragging its separator or using the keyboard. Its allowed width is `290–480px`, further capped at `window.innerWidth - 410px`, with the `290px` minimum retained. Arrow keys change width by `10px`; Home chooses `290px`, End requests `480px`. Pointer capture keeps the drag stable outside the small handle. On release, resize the visible plots.

Persist explicit inspector width under `turbodash.workbench.inspector-width`. Theme choice is separate, under `turbodash.workbench.theme`. Keep these view preferences out of the engineering design file. See [workbench.js](../../turbodash/apps/app_turbine_optimization/assets/workbench.js), `setupResize`, `setupHeader` and `applyTheme`.

## 2. Typography, spacing and component geometry

Use the system font stack headed by **Segoe UI**: `"Segoe UI", -apple-system, BlinkMacSystemFont, sans-serif`. The app base is `13px / 1.5`. Inputs inherit the app font. Numeric values use tabular figures; technical logs use monospace. Heading letter spacing is `-.025em` unless a more specific value is given.

| Role | Size / weight / spacing |
| --- | --- |
| Wordmark | `25px`, weight `650`, letter spacing `-1px` |
| Main view title | `26px / 1.3`, weight `600`, margin `4px 0 0` |
| Canvas H2 | `25px`, weight `600` |
| Inspector title | `20px`, weight `600`, zero margin |
| Result/plot heading | `14px`, weight `600` |
| Eyebrow | `9px`, weight `600`, letter spacing `1.4px`, muted, uppercase text |
| Numbered section summary | `12px`, weight `600` |
| Ordinary compact field label | `11px / 1.4`, weight `500` |
| Ordinary compact field value | `12px`, tabular figures |
| Header action | `12px`, weight `500` |
| Rail label | `11px`, weight `500`; active `650` |
| Optimization tab label | `11px / 1.3`; active weight `600` |
| Embedded bound/constraint label | `9px / 11px`, muted, weight `500` |
| Bound/constraint value | `11px` |
| Metric label | `11px`, muted, bottom margin `9px` |
| Metric number | `27px / 1.2`, weight `600`, letter spacing `-.7px`, tabular figures |
| Metric unit | `12px`, weight `400`, muted, left margin `5px` |
| Report table cell / header | `12px` / `11px`; header weight `600` |
| Progress text | `11px / 1.5 Consolas, monospace` |
| Convergence terminal | `12px / 1.65 Consolas, 'Cascadia Code', monospace`; tab size `4` |

The spacing system is compact rather than a strict single-step scale. Reproduce these recurring distances: `3–5px` inside segmented controls and compact bounds; `6–10px` between dense inputs and related controls; `12–18px` inside cards and between plots; `20–21px` between major result groups. Do not substitute large marketing-page whitespace for the editor density.

| Component | Geometry |
| --- | --- |
| Standard card/numbered section/result disclosure | `1px` border, radius `7px`, no decorative shadow |
| Numbered section | Bottom margin `6px`; summary minimum height `49px`, padding `11px 10px`, gap `8px` |
| Number badge | Circle `23 × 23px`; `9px`, weight `500`, tabular two-digit number |
| Section body | Padding `12px 10px 13px` |
| Ordinary field row | Bottom margin `9px`; final row has no bottom margin |
| Compact input | Minimum height `34px`; padding `6px 7px`; radius `5px` |
| Standalone inspector input | Minimum height `39px`; base padding `8px 10px`; radius `5px` |
| Segmented control | Border `1px`; padding `3px`; gap `3px`; outer radius `6px`; choice radius `4px` |
| Stage-count stepper | Height `34px`; radius `5px`; each end button `30 × 34px`; centered numeric output |
| Header/file button | Padding `8px 12px`; gap `7px`; radius `7px` |
| Parameter search | Padding `10px 12px`; radius `6px`; `13px` text |
| Floating search results | Radius `8px`; padding `5px`; shadow; maximum height `min(450px, 65vh)` |
| Ordinary help marker | `15 × 15px` in a field-label row; `9px` question mark in a bordered circle |
| Standard help bubble | Width up to `230px`; padding `10px 12px`; radius `5px`; `11px / 1.6` text |
| Optimization row | Padding `8px`; bottom margin `8px`; radius `5px`; inspector-colored inner surface |
| Optimization row header | Quantity dropdown plus `18px` remove column; gap `4px`; bottom margin `8px` |
| Compact three-column row | Equal flexible columns; gap `5px`; field height `40px` |
| Results disclosure | Bottom margin `21px`; summary padding `16px 18px`; `14px`, weight `600` |
| Plot pair | Equal flexible columns, gap `18px`; group inset `0 16px 16px` |
| Plot title | Padding `17px 18px 0`; zero margin |
| Meanline plot graph | Inline height `360px` in `plot_card`; responsive width |
| Velocity-triangle graph | Inline height `340px`; one graph per stage |
| Shared engineering plot | Workbench height `350px`; flex column with graph filling remaining height |
| Optimization terminal | Outer radius `10px`; top margin `24px`; preformatted area padding `18px`, minimum height `120px`, maximum height `340px`, independent scrolling |

The plot graph's inline height takes precedence over the `350px` CSS fallback for `.td-plot-card > .dash-graph`. Preserve this distinction when matching the current Meanline view. Likewise, the terminal retains its base `10px` outer radius even though numbered sections and ordinary result cards use `7px`.

Use line borders and surface changes to establish containment. Reserve the shared shadow token for floating menus, search results, help and the mobile drawer. The exact colors and selected-state recipes are in [visual style](visual-style.md).

## 3. Input hierarchy and disclosure recipe

The inspector has five numbered peer sections. Every section begins closed on a fresh visit and on explicit YAML import. Opening one does not close another. **Collapse all** closes the inspector's disclosures together. Ordinary edits preserve their expanded state. Applying optimized values preserves the Optimization section and selected tab.

```text
Design inputs
├── Find a parameter
├── 01 Turbine setup
├── 02 Operating point
├── 03 Shaft speed
├── 04 Stage design
│   ├── Stage selection + Stator / Rotor scope
│   ├── Aerodynamic design
│   └── Blade geometry
└── 05 Optimization [SLSQP]
    ├── Configuration
    │   └── Show expert settings
    ├── Design variables
    │   ├── + Add variable / + Add groups
    │   └── Quantity + Lower / Initial / Upper
    └── Constraints
        ├── + Add constraint / + Add groups
        └── Quantity + Comparison / Target / Scale
```

| Section | Contents and scope |
| --- | --- |
| **01 Turbine setup** | Axial/Radial segmented selector; integer number of stages with add/remove buttons; loss-model dropdown |
| **02 Operating point** | Searchable working fluid; readable inlet property-pair dropdown; two property values with their current units; exit pressure; mass flow; inlet flow angle; nearby conversion/validation feedback |
| **03 Shaft speed** | Specific/Actual representation selector; corresponding converted value and units; global blade velocity ratio |
| **04 Stage design** | Named stage dropdown and explicit Stator/Rotor scope; inline Aerodynamic design and Blade geometry disclosures; fields applicable to the current axial/radial model |
| **05 Optimization** | Objective and goal, solver settings, variable and constraint selection/editing; background run control and terminal are placed outside this collapsible body |

Build primary sections with native `details` and `summary`. Replace the primary section's native marker with a right-aligned `›` that rotates `90deg` when open. The number badge precedes the title. Open primary summaries stick at the top of the inspector scroll region; the stage/row scope sticks below the Stage design summary. A long section therefore retains its identity and editing context during scrolling.

Nested Stage design disclosures remain visibly subordinate: `11px` heading, `8px` vertical summary padding, and a body with `2px` left border and `8px` inset. Light-mode nested summaries receive the quieter blue surface and a small bordered box. See `stage_editor`, `section` and `build_layout` in [workbench.py](../../turbodash/apps/app_turbine_optimization/workbench.py).

### Field alignment and retained state

Place an ordinary label and its control on the same row with:

```css
display: grid;
grid-template-columns: minmax(0, 1fr) minmax(108px, 1.18fr);
gap: 8px;
align-items: center;
```

Allow labels to wrap, and keep controls within the parent width. Put the optional question-mark help beside the label, not on top of the input. Use units in labels and short descriptive names rather than raw model keys. Context selectors clarify whether a number belongs to the turbine, a named stage, a stator or a rotor.

All stage and blade-row controls remain mounted. Selecting another stage/row changes visibility only. Adding or removing stages preserves the retained stages' values and names. Changing turbine type intentionally initializes stage parameters from the selected type's defaults, preserving stage count and names. Workspace navigation must not discard inputs.

Continuous numeric controls use `step="any"` and debounced commits. Avoid sliders for precise engineering quantities. Display padding adds at least three decimal places without removing additional supplied digits or deliberate trailing zeros. Scientific tolerances use scientific notation. Formatting must preserve stored precision and active typing; integer counts remain integers. Blank and invalid drafts stay visible rather than silently reverting to defaults. See [workbench.js](../../turbodash/apps/app_turbine_optimization/assets/workbench.js), numeric formatting helpers, and the [design principles](design-principles.md).

### Optimization configuration and compact editors

The Configuration tab shows **Objective**, **Goal** (Maximize/Minimize), **Iteration limit**, **Show expert settings**, and **Solver tolerance**. The expert switch additionally reveals **Feasibility tolerance**, **Optimality tolerance**, **First step fraction**, **Step growth factor**, **Step reduction factor** and **Trial retry limit**. Hiding expert settings preserves their values. The expert preference is stored locally. This is progressive disclosure of solver controls, not a second configuration dialog.

The settings body uses a single column. Objective and Goal use narrower dedicated label columns (`51px` and `42px` respectively); ordinary optimization fields use the standard compact grid. Tabs occupy equal widths, can wrap their text, and use `11px / 1.3` labels. At an optimization-container width of `245px` or less, labels drop to `10px` with `8px 2px` padding. This is a container query, separate from viewport breakpoints.

Each selected quantity gets one contained editable row. The row header is a searchable quantity dropdown plus an explicitly labelled remove button. Under it, embed short labels inside three equal fields:

| Design variable | Constraint |
| --- | --- |
| Lower | Comparison (`≤`, `=`, `≥`) |
| Initial | Target |
| Upper | Scale |

The embedded label sits `4px` from the top, `6px` from the left; input padding is `16px 5px 3px`. Labels remain visible when fields contain values. Initial may be blank to use the current design; its placeholder identifies the current value. Constraint Scale is a positive physical divisor for normalizing the residual, including when Target is zero. Preserve edited bounds/targets/scales when other quantities are added or catalog options refresh. See [opt_panel.py](../../turbodash/apps/app_turbine_optimization/opt_panel.py), `panel`, `variable_row` and `constraint_row`.

## 4. Menus, search and contextual help

### Hierarchical selection menu

Place **+ Add variable** or **+ Add constraint** and **+ Add groups** above their editable row list. Each toggle is an equal-width control in a flex row with `6px` gap. It has minimum height `33px`, padding `6px`, radius `5px`, `11px` medium text and a small drawn chevron.

Open a bordered, shadowed menu below the action row with a `5px` gap, `8px` padding and `6px` radius. The menu spans the action row's width. Its filter is sticky at the top; the filter is `32px` high, with `6px 8px` padding, `4px` radius and `11px` text. Its scroll area contains a hierarchy, rather than an undifferentiated list of model paths.

Under each stage in the design-variable picker, use exactly three categories: **Stage inputs**, **Stator geometry**, and **Rotor geometry**. Stage inputs contains reaction, stator exit flow angle, applicable work split, meridional velocity ratios, and applicable radial radius ratios. Ratios do not form separate categories; their availability still follows the turbine model.

The catalog supplies scope branches for overall quantities, all-stage selections, named stages and applicable aerodynamic, stator/rotor geometry and flow-station quantities. Use stable identities for branch and editor state. Quantity leaves use short labels with an optional muted description; the full accessible label and search terms include their scope.

| Menu part | Recipe |
| --- | --- |
| Branch summary | `11px`, weight `600`; padding `7px 5px`; gap `7px`; radius `4px`; `5px` CSS chevron |
| Child indentation | Margin-left `7px`; padding-left `14px`; `1px` vertical connector |
| Child connector | `14px` horizontal line at `16px` from row top, attached to the vertical connector |
| Leaf item | Full-width button; min-height `33px`; padding `7px 5px`; gap `7px`; radius `4px` |
| Checkbox indicator | `14 × 14px`; radius `3px`; full tick or horizontal mixed-state dash |
| Leaf label | `11px / 1.4`; wraps long labels |
| Description | `10px / 1.4`, muted; margin-top `3px` |

Checked and mixed groups use a tinted row and accent checkbox. Partial selection is represented by `aria-checked="mixed"`, not by a separate saved ownership state. Group additions add only missing members; overlapping presets do not duplicate rows or reset existing edits. Unchecking a group removes its matching members even if another group also contains them. Individual constraint leaves cover all comparisons on that quantity; opposite comparisons may remain separate editable rows.

Only one picker is open at a time. Opening focuses its filter; search expands matching branches, and clearing restores remembered branch expansion. Escape closes the picker and returns focus to its toggle. Outside clicks close it. Preserve menu scroll, focus and editor identity through rerenders. Opening a menu alone must not rebuild the numeric rows or disturb numeric focus.

Limit height to `min(360px, 45vh)` and additionally to the space in the inspector after allowing for the sticky heading and action row. Scroll the inspector if necessary to expose the menu above the fixed optimization footer. Do not let a dropdown become unreachable below the visible editor. The current implementation uses an ordered selection journal so rapid clicks survive overlapping requests. See [opt_selection.py](../../turbodash/apps/app_turbine_optimization/opt_selection.py), [optimization.css](../../turbodash/apps/app_turbine_optimization/assets/optimization.css) and [optimization.js](../../turbodash/apps/app_turbine_optimization/assets/optimization.js).

### Find a parameter

Global parameter search belongs at the inspector top. Match readable names, aliases, engineering symbols, model keys and workspace/stage/row context, including collapsed or hidden controls. Normalize Greek-symbol aliases and whitespace. Show up to `30` results with a prominent field name and smaller context subtitle.

Choosing a result selects its workspace and stage/row scope, opens containing disclosures and the applicable optimization tab/expert setting, scrolls to the field, focuses it and adds an accent outline for `1700ms`. It never edits the model. Search is a navigation tool, so keep result context visible and make a no-match result explicit. See `indexFields`, `search` and `reveal` in [workbench.js](../../turbodash/apps/app_turbine_optimization/assets/workbench.js).

### Help

Help supplements labels. Ordinary fields use a focusable `?` circle with `title` and accessible explanation. Hover and keyboard focus reveal the same information. The bubble uses panel/text/border tokens and a shadow; it does not intercept pointer events on controls beneath it.

Optimization bound/constraint labels expose their own help. Those compact bubbles are `210px` wide and shift left/center/right by column to remain inside the narrow editor. Optimization tabs expose explanatory text through `aria-describedby` and a `data-tooltip` CSS bubble after `750ms` of hover or keyboard-visible focus. The bubble spans the tab strip, sits `6px` below it, and uses `10px 12px` padding, `6px` radius and `12px / 1.5` text. It stays inside the workbench and follows the active theme tokens, with pointer events disabled. The accessible description is retained in a visually hidden span. Keep geometry-model limitations in nearby help where they affect interpretation.

## 5. Actions, status and asynchronous state

The optimization dock is an inspector footer, outside the collapsible editor and outside its scroll region. It has a top border, panel background, `16px` padding and `12px` gap. Stack **Optimization**, the named outcome, explanatory status, compact progress text and the full-width action. Start button padding is `11px 14px`, radius `7px`, weight `600`. The dock hides in Geometry and Mesh.

| State | Visible outcome | Tone / action |
| --- | --- | --- |
| Ready | Ready | Neutral status, Start optimization |
| Running | Running | Accent status, Optimizing…; action disabled |
| Applying | Applying design | Accent status, Applying design…; action disabled |
| Success | Applied successfully | Success tone; enabled Start optimization |
| Current design changed | Design changed · result not applied | Warning tone; current inputs retained |
| Application cannot be confirmed | Application unconfirmed | Warning tone; no claimed success |
| Failure | Optimization failed | Danger tone; enabled action permits retry |

A small colored dot accompanies the outcome label, but the outcome is always written in text. Use polite live regions and an explicit busy state. Green success appears only after both current controls and displayed results match the optimized target. Running optimization locks its configuration tabs and selection controls; the engineering model inspector remains editable. If the user changes the design during a run, preserve the newer design rather than applying an obsolete result.

Open the convergence terminal when a run begins. It sits at the bottom of the Meanline results column and can be collapsed independently. Follow incoming output while the reader is near the bottom; pause following when they scroll upward. Reopening the terminal respects that choice. The terminal is focusable and scrollable. See [opt_panel.py](../../turbodash/apps/app_turbine_optimization/opt_panel.py), `actions`, `outcome_presentation`, `register_callbacks`, and [optimization.js](../../turbodash/apps/app_turbine_optimization/assets/optimization.js).

Calculation and load statuses live above the workspace content. Empty status/validation regions do not consume space. Invalid committed drafts retain the last successful result with a visible error/stale status and prevent saving stale results. Late responses must not overwrite newer edits, restore obsolete values, or clear a newer warning. Unit/property conversion commits the converted value and representation together using the current canonical inputs. Details of these ownership rules are part of the [design principles](design-principles.md), not optional visual polish.

## 6. Meanline results, charts and reports

The results column begins with **TURBINE PERFORMANCE** as a small eyebrow, **Meanline design** as the main title, and a quiet result-state badge to the right. The heading row has a `21px` bottom margin. Its result badge uses `11px` muted text plus a `5px` accent dot.

Four headline metrics share one bordered strip with a `3px` accent top edge, radius `7px`, vertical padding `20px` and bottom margin `21px`. Four equal cells have `20px` horizontal padding and dividing borders. They represent the same successful calculation:

| Metric | Format |
| --- | --- |
| Total-to-static efficiency | Percent, two decimals |
| Total-to-total efficiency | Percent, two decimals |
| Shaft speed | Whole-number rpm |
| Power | Signed actual shaft power, two decimals, adaptive W / kW / MW |

Power promotes units when rounding would otherwise show `1,000.00` of the smaller unit. Missing/nonfinite values use an em dash. Units are smaller and muted rather than included in the large numeral.

Below the metrics, use separate independently collapsible result disclosures, all initially open:

1. **Turbine geometry**: Meridional channel view and Blade-to-blade view, paired.
2. **Efficiency and loss distribution**: Efficiency vs blade velocity ratio and Efficiency penalties by loss mechanism, paired.
3. **Velocity triangles**: Display selector for Mach numbers / Velocity [m/s], then one full-width card per stage.
4. **Performance reports**: Performance / Geometry / Flow stations tabs.

The Optimization terminal follows these groups and begins collapsed. Closing a result group keeps its figures, selectors and data mounted. Reopening resizes its plots; it does not repeat the numerical calculation. Switching triangle representation uses current results and preserves the chosen mode for the browser session.

### Plot presentation rules

Keep responsive Plotly figures inside bordered cards. Preserve zoom, pan, reset and image export while hiding the Plotly logo. Theme changes restyle the plot and its export colors without rerunning the model or resetting manual zoom. A newly returned server figure receives the active theme before its light defaults can flash visibly. See [workbench.js](../../turbodash/apps/app_turbine_optimization/assets/workbench.js), `themeFigure`, and [plotly_turbine.py](../../turbodash/visualization/plotly_turbine.py).

Use physical aspect ratio for engineering geometry and velocity triangles. Geometry uses solid hub/shroud boundaries, orange stator regions and blue rotor regions, matching translucent fills, and hover identification instead of redundant legends. Five smooth guide streamlines are explicitly illustrative geometry paths, not CFD results. Initial/reset/autoscale meridional views include zero radius; deliberate manual radial zoom remains available. The meridional channel and physical mesh-row model have different geometry assumptions; retain the nearby explanatory help.

Keep legends on efficiency, loss and velocity-triangle plots. Efficiency is shown in percent. Its curves are kinematic sweeps at fixed work split and loss coefficients; do not label them thermodynamic off-design calculations. Loss bars represent percentage-point efficiency deficits `Δη [%]`, with one stator/rotor contribution per row and one final exit-kinetic-energy column. Their sum is `100 * (1 - efficiency_ts)`, including an explicit unallocated component when needed. Hover text names the quantity and units. Velocity vectors have arrowheads, distinct inlet/outlet colors and common extents across stages. The exact chart palette is in [visual style](visual-style.md).

### Tables

Reports use native sorting and CSV export. Align the descriptive first column left and numeric columns right. Keep number columns on one line and scroll wide tables horizontally within their container. Use `12px` cells, `11px` semibold headers and padding `10px 14px`; headers use the selected-surface tint and muted text. Alternate rows receive a subtle canvas/panel mix; hover uses the tint; a focused cell has an inset `2px` accent outline. Export is a small bordered `11px` action. See `make_data_table` in [workbench.py](../../turbodash/apps/app_turbine_optimization/workbench.py) and the final table overrides in [workbench.css](../../turbodash/apps/app_turbine_optimization/assets/workbench.css).

## 7. Geometry and Mesh workspaces

Keep the same header, rail and shared model inspector. `build_layout` extracts the shared components' side controls into the common inspector and removes the duplicate geometry model editor. Geometry/Mesh results occupy a single-column nested workspace in the main canvas; do not copy the standalone shared workspace's full-height side-by-side shell into this app.

| Workspace | Inspector additions | Results structure |
| --- | --- | --- |
| Geometry | Turbine geometry heading with explanatory help; Inspect and export a row disclosure with stage/row selectors and Download 3D model ZIP | Paired Blade-to-blade view / Meridional channel; 3D turbine with Configure display; separate TurboGrid export disclosure |
| Mesh | Mesh generation heading with help; Active row disclosure with stage/row selection and Generate mesh; per-stage/per-row settings; Generation status and download disclosure with progress, summary and Download mesh ZIP | Paired Blade-to-blade mesh / Meridional mesh; 3D mesh with Configure display and Mesh inspection controls |

These workspace additions sit below the common numbered sections, separated by a top rule, `22px` top padding and `16px` top margin. Row settings use bordered `6px` cards, `12px` vertical margin and contained summaries. The workbench overrides their text, borders and surfaces to the active theme. These secondary settings disclosures do not use all the numbered section's blue-header overrides.

Generate mesh is explicit. Editing settings invalidates the existing mesh rather than silently regenerating it. Display and inspection controls change the view without regenerating mesh data. Downloads are enabled only for an applicable successful result. Keep optional-viewer availability messages and generation/inspection status near their relevant controls.

The VTK viewport has an adjacent **Configure display** panel. Its sections are Blade rows, Blades per row, Surfaces, Mesh edges where applicable, Fit view and additional Mesh inspection controls. Use a stable viewer host and update its data rather than remounting the viewport, preserving display state. Default visible surfaces are blade-only for the turbine overview and blade + hub for row/mesh views.

| VTK/shared geometry | Current dimensions |
| --- | --- |
| 3D host | Height `54vh`; minimum `390px`; maximum `700px`; width `100%` |
| Viewer frame | Flexes into available width; padding `8px` |
| Display panel | Width/flex basis `245px`; padding `14px`; maximum height `74vh`; vertical scrolling; left border |
| Display heading | `15px`; section heading `13px`, weight `600` |
| Display section | Top border; margin `12px 0`; padding-top `10px` |
| Shared plot group in workbench | Border/radius `7px`; padding `17px`; bottom margin `20px` |
| Engineering plot | Workbench height `350px`; title `14px`, weight `600` |

See [meshing.py](../../turbodash/apps/shared/components/meshing.py), [turbogrid.py](../../turbodash/apps/shared/components/turbogrid.py), [vtk.py](../../turbodash/visualization/vtk.py), and [workspace.css](../../turbodash/apps/shared/assets/workspace.css), together with the workbench overrides. At `1200px` viewport width or less, the VTK display panel becomes `220px` with `12px` padding. At `1000px` or less, it stacks below the viewer at full width, without the former maximum height, with a top border replacing the left border.

## 8. Responsive behavior

Use these implemented viewport breakpoints. The inspector width shown below is the CSS default; an explicit persisted desktop resize may supply an inline width instead. Drawer width is independently specified for narrow screens.

| Viewport width | Required changes |
| --- | --- |
| Above `1250px` | Header `72px`; rail `92px`; inspector `348px`; four metric cells; paired plots where space permits |
| `1250px` or less | Rail `78px`; inspector `320px`; canvas padding `22px 18px 36px`; metric value `23px`, label `10px`, cell horizontal padding `12px` |
| `1080px` or less | Metrics become two columns with `22px` row gap; remove divider after second metric; plot pairs become a single column |
| `850px` or less | Header baseline `68px`; rail `72px`; header padding `12px 15px`, gap `11px`; brand `23px`, logo `32px`; show Parameters button; inspector becomes a drawer; canvas padding `21px 16px` |
| `600px` or less | Header baseline `110px`, wrapping into additional rows when required; rail `62px`; header padding `10px 12px`, gap `9px`; brand `21px`; header actions wrap and use `10px` text, smaller links `9px`; rail labels `9px`, buttons minimum `62px`; canvas padding `17px 10px 30px`; metric value `20px`, label `9px`, unit `10px`; plot gap `12px` |

At `850px` or less, the inspector is initially hidden. Parameters opens a fixed drawer starting immediately below the **measured** header, left of the rail's right edge. Its width is `min(380px, 100vw - rail-width)`. The drawer is above a dark translucent backdrop. Resize handles are hidden. Backdrop click or Escape closes the drawer; Escape returns focus to Parameters. The optimization dock remains inside the drawer, so it appears only while the Meanline parameter drawer is open. A ResizeObserver measures the header bottom; do not assume the nominal height after header actions wrap.

At `600px` or less, efficiency, loss and velocity-triangle graphs retain a minimum width of `540px` and scroll **inside their cards**. Keep the efficiency/loss headings pinned to the card's left edge during horizontal scrolling. This intentionally protects readable axes/legends without causing whole-page horizontal overflow. Geometry plot pairs stack and scale to their available width.

The mobile optimization dock uses `10px 12px` padding, `10px` gap and a full-width button with `9px 12px` padding. Its text drops to `11px`. Shared workspace plot groups reduce padding to `12px`.

## 9. Keyboard and accessible interaction contract

Accessibility here describes implemented semantics and interaction support; it does not assert formal WCAG certification. Preserve the following when rebuilding:

| Action / component | Keyboard and semantic behavior |
| --- | --- |
| Workspace navigation | Real buttons with visible labels; current workspace exposed with `aria-current="page"` |
| Theme | Labelled group, real buttons, active state exposed with `aria-pressed` |
| Primary/nested disclosures | Native keyboard-operable `details` / `summary`; focus remains visible |
| Find parameter | Ctrl/Cmd+K opens the drawer if needed and focuses/selects search; ArrowUp/Down cycles matches; Enter reveals chosen field; Escape dismisses results |
| Search semantics | Combobox with list autocomplete; linked listbox; selected result and active descendant; no-match message is a status |
| Inspector resize | Focusable separator with vertical orientation and min/max/current width; ArrowLeft/Right, Home, End supported |
| Help | Same content on hover and focus; linked/accessible explanatory labels; tooltips do not block input clicks |
| Selection tree | Keyboard-operable native disclosure branches and leaf buttons with checkbox roles; `false` / `true` / `mixed` states; Escape restores toggle focus |
| Field labels | Associated labels and programmatic names, including dropdown input names; visible units and scope |
| Remove controls | Explicit Remove variable / Remove constraint names, not only a visual × |
| Status | Written outcome plus semantic tone/dot; polite live regions; explicit busy state; alerts for errors |
| Terminal | Focusable labelled output with independent scrolling; does not pull reader back to new output after they scroll up |
| Narrow inspector | Parameters has expanded/controls state; labelled backdrop closes it; Escape restores toggle focus |

All interactive elements receive a `2px` accent focus outline with `3px` offset. Inputs additionally change border to the accent and receive a restrained focus halo. Selected segmented controls keep a keyboard-visible outline through their hidden native radio input. Disabled controls visibly reduce emphasis; do not rely on color alone to communicate failure or success.

Honor `prefers-reduced-motion: reduce`: disable motion, animated transitions and animated scroll behavior. Keep the tab-help visibility delay at `750ms`; the delay has no movement and prevents help flashing while navigating. Search/context changes, opening a disclosure, changing theme and inspecting a mesh are view actions and must not cause numerical recalculation. See [workbench.js](../../turbodash/apps/app_turbine_optimization/assets/workbench.js), [optimization.js](../../turbodash/apps/app_turbine_optimization/assets/optimization.js), and the focus/reduced-motion rules in [workbench.css](../../turbodash/apps/app_turbine_optimization/assets/workbench.css).

## 10. Rebuild checks and transfer to another package

Check the actual rendered interface in both themes at a wide desktop, a resized inspector, `1080px`, `850px` and a narrow phone width. Confirm aligned fields, visible open-section context, readable menus near the inspector bottom, independently scrolling results, preserved plot legends, and no whole-page horizontal overflow. Confirm that view-only interactions retain model values and plot zoom, and that delayed asynchronous work cannot replace newer edits. Use the current browser and Python checks linked by the [implementation map](implementation-map.md); old run totals are not part of this specification.

For a different engineering package, the following are **adaptations**, not claims about current TurboDash:

- Keep the stable rail / scoped inspector / result-canvas relationship, replacing the three workspaces with the new tool's real tasks.
- Reuse numbered setup sections and separate navigation tint from editable surfaces; choose section names and scope selectors appropriate to the model.
- Keep precise numeric entry, explicit expensive actions, truthful result status and retained editor state even if the framework changes.
- Adapt desktop dimensions and breakpoints when the new tool has materially different label lengths or visualization needs; preserve minimum usable control width and internal overflow behavior.
- Carry over the accessible interaction contract and menu hierarchy rather than copying only colors and rounded boxes.

The [visual style](visual-style.md) records the color and surface system; [iconography](iconography.md) records the identity assets and icon rules. Together these documents provide a reproducible current style while allowing deliberate changes for another package.
