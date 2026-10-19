# Turbodash: a clearer place to think

Four design studies for the left inspector and its relationship to the plots. These are review mockups, not changes to the running app. Every concept starts with all expandable sections closed; the expanded images show a deliberate user action. Plots, results and solver histories are illustrative, not solver-certified calculations.

**Recommendation: Nordic Studio.** Its quiet light surfaces, teal emphasis and concise section summaries make a dense scientific model feel approachable without hiding its precision. Start with this structure; borrow Graphite's convergence dock when optimization is active. Keep Graphite as a possible dark edition, rather than making a dark interface mandatory for everybody.

Open [the visual gallery](index.html) to compare the concepts, [the overview PNG](overview.png) for a single presentation image, or [the enlarged left-pane comparison](sidebar_comparison.png) to inspect widgets side by side. The four individual HTML studies use native `details` disclosures and start closed. The separate comparison sheet deliberately shows expanded examples. Collapse all demonstrates closing the sections. Numeric fields are read-only. Search, tabs, segmented controls, the task rail, sliders, resizing and file/run actions are visual examples and are not wired to application behavior.

| Concept | Visual character and interaction | Best use / tradeoff | Visuals |
| --- | --- | --- | --- |
| **01 Nordic Studio** | Light canvas, white accordion cards, teal actions. Seven compact sections with useful summaries. A proposed 350 px resizable inspector gives more room to plots than the current 600 px sidebar. | Best everyday default. Familiar and easy to scan; long variable lists still need a larger editor. | [Closed](png/01_nordic_studio_collapsed.png) · [Stage design opened](png/01_nordic_studio_expanded.png) · [HTML](01_nordic_studio.html) |
| **02 Graphite Console** | A narrow task rail, dark compact inspector and restrained cyan. Bound triplets for optimization variables; convergence and logs in a lower dock. | Best for sustained optimization work. More chrome and density; task-rail icons need labels/tooltips and a clear selected state. | [Closed](png/02_graphite_console_collapsed.png) · [Optimization opened](png/02_graphite_console_expanded.png) · [HTML](02_graphite_console.html) |
| **03 Blueprint Workbench** | Navy sidebar against a light plotting canvas. Numbered workflow chapters communicate a sensible sequence while remaining freely accessible. | Best for setting up unfamiliar cases and teaching. Numbers must not become a restrictive wizard or imply physical validation. | [Closed](png/03_blueprint_workbench_collapsed.png) · [Operating point opened](png/03_blueprint_workbench_expanded.png) · [HTML](03_blueprint_workbench.html) |
| **04 Precision Paper** | Warm paper surfaces, indigo emphasis, restrained rules and compact property rows. Explicit stage/row scope makes geometry editing feel like a technical property inspector. | Best for geometry refinement and long reading sessions. Scope must remain visible so changes never land on the wrong blade row. | [Closed](png/04_precision_paper_collapsed.png) · [Blade geometry opened](png/04_precision_paper_expanded.png) · [HTML](04_precision_paper.html) |

## Palette and typography

| Concept | Canvas | Panel / sidebar | Main text | Accent |
| --- | --- | --- | --- | --- |
| Nordic Studio | `#F3F6F7` | `#FFFFFF` | `#142F3B` | `#087F83` teal |
| Graphite Console | `#111922` | `#19242F` | `#E4EDF3` | `#58CCD4` cyan, with dark text on filled buttons |
| Blueprint Workbench | `#EFF3F8` | `#162D4B` navy | Dark ink on light plots; light text in sidebar | `#2462D1` blue; `#B76D16` amber for attention |
| Precision Paper | `#F5F2EB` | `#FFFEFA` | `#303741` | `#5951A6` indigo |

Use a clear system sans-serif, comfortable 13–14 px form text, tabular numerals and consistent decimal alignment in the eventual app. These presentation boards use denser text to show a complete workspace; their smallest captions are not a recommended production text size. Reserve monospaced text for solver output and paths. Give graphs the largest quiet surfaces. Use color for emphasis, not decoration; labels and icons also communicate selection, validity and job state. The palette is a starting point: every final text, border, focus and chart combination needs checking at its actual size and background.

## Information architecture and widgets

The current meanline sidebar mixes global setup and many stage inputs. The boards split it into the seven sections below, with Load YAML / Save design in the toolbar and job status outside the disclosures. Geometry and Mesh remain separate workspaces with contextual controls. The expanded examples show representative fields; the mapping below specifies the complete control families to preserve in a future implementation, including controls not rendered on the boards.

| Section | Controls retained from the app | Recommended presentation |
| --- | --- | --- |
| **Turbine setup** | Axial/radial type, stage count, working fluid | Segmented type control, searchable fluid selector and stage count. The board shows the count as a number field; future explicit add/remove controls should distinguish model changes from stage selection. Closed summary: `Axial · 1 stage`. |
| **Operating point** | Inlet property pair and both values, exit pressure, mass flow, inlet angle | Readable property-pair names; number inputs with persistent units. Closed summary repeats the fluid for context, alongside the inlet conditions. |
| **Shaft & loading** | Blade velocity ratio, specific/actual speed mode and value | Mode control with a dynamically labelled speed field; precise numeric value with an optional slider for velocity ratio. |
| **Stage design** | Work fraction, stator exit angle, reaction, three meridional velocity ratios and four radius ratios | Stage selector, followed by named field groups; numbers first, optional sliders for exploratory changes. Keep ratio direction and station subscripts unambiguous. |
| **Blade geometry** | Solidity, axial aspect ratio, maximum thickness ratio/location, leading-edge radius ratio, trailing-edge thickness ratio, leading/trailing wedge angles | Visible stage and stator/rotor scope; compact property rows. Full labels distinguish true chord, meridional chord, pitch, opening and maximum thickness. Hide inapplicable axial-only aspect ratio for radial cases. |
| **Models & losses** | Loss model selection | A dedicated model selector, with Benner, Kacker–Okapuu, Moustapha and Isentropic choices. Closed summary names the active model. |
| **Optimization** | Objective/direction, solver and feasibility tolerances, iteration limit, design variables with bounds/initial values, constraints with comparison/target/scale, start and status | Settings / Variables / Constraints tabs inside one disclosure. Lower / Initial / Upper triplets; units beside values. Collapsed summary shows objective and counts. Running/error state remains visible outside the editor. |

YAML save/load stays outside the seven sections. Future TurboGrid exports and applicable 3D/mesh downloads belong beside their relevant viewport or in a contextual export disclosure, with an explanation of what result is being exported and why an action is disabled.

Geometry retains row selection, stage/shape editing, the 3D model, and TurboGrid options. Mesh retains independent per-row blade/passage element sizes, spanwise layers/stretch, profile points, inlet/outlet extents and inflation height/thickness/growth. Use a switch for inflation and an explicit **Generate mesh** action. Mesh inspection retains surface/slice/layer/worst-cell/threshold views, quality metric, slice axis/position, layer, count and threshold. These workspaces are not separately mocked up; the recommendations ensure their existing tasks survive a later redesign. Place inspection controls beside the viewport; changing a view should not look like regenerating a mesh.

## Behavior that makes the design useful

- **Closed by default, informative while closed.** Show title, short value summary and any issue count. Every section can be closed; opening one need not close another. Add a labelled Collapse all action. Persist disclosure state only after user interaction during the session; a genuinely new design starts closed. Ordinary value edits must not rebuild or collapse the inspector.
- **Keep precision accessible.** Numeric entry is authoritative; sliders supplement it. Preserve existing precision and bounds instead of rounding stored values to match a pretty label. Scientific notation should remain possible for small tolerances and pressures. Explicit labels stay visible even when placeholders disappear.
- **Make units adaptive.** The boards show pressure in Pa: inlet 500000 Pa and 600 K, with exit pressure 100000 Pa. A future optional display of 500 kPa / 100 kPa would require explicit presentation conversion while preserving solver inputs. Inlet names/units follow the selected property pair. Actual shaft speed is rpm; specific speed is dimensionless. Angles are degrees; mesh element sizes are mm; optimization scales carry the physical units of their targets.
- **Respect the live model.** Meanline already recomputes after committed edits. Use `Live model · Up to date`, `Updating`, or an actionable error state; do not invent a Calculate button. Preserve the last successful result and clearly mark it stale while inputs are invalid or a calculation is pending. Optimization and mesh generation remain explicit actions.
- **Show honest job state.** Display solver state, iteration, objective and feasibility when available. Avoid invented progress percentages or completion times. A collapsed optimization editor still exposes running/failure status; the lower log dock can be expanded independently.
- **Let the canvas breathe.** Propose independent sidebar/canvas scrolling, a draggable width boundary and a narrow-screen drawer. Keep the active stage/row and file controls stable while scrolling. Resizing, search and session-state behavior are recommendations, not implemented app features.
- **Avoid scope surprises.** Stage selection changes the editor context; add/remove changes the model. Label them separately. Geometry uses rounded physical blade counts while meanline uses continuous counts; retain that explanation near the relevant geometry result. Axial-only mesh availability and stale-design restrictions remain visible.
- **Keyboard and error handling are part of the style.** Headers support Enter/Space, visible focus and predictable Tab navigation; state is exposed to assistive technology. Put a short text error on the relevant collapsed header and a full message beside its field. Use text/icons in addition to color. These accordion interactions follow the [W3C accordion pattern](https://www.w3.org/WAI/ARIA/apg/patterns/accordion/).

For a future Dash implementation, use `dbc.Accordion(start_collapsed=True, always_open=True)` and remove existing explicitly opened `active_item` values; native `html.Details` should omit `open=True`. Keep controls mounted when changing workspace or collapsing groups, and use stable IDs so edits preserve state. Numeric fields should retain debounce behavior: [Dash input documentation](https://dash.plotly.com/dash-core-components/input) specifies that `debounce=True` sends changes on Enter or loss of focus. Validation must handle incomplete numeric input without silently replacing the last successful design.

## Shared illustrative case

All four concepts use inputs from the existing [axial optimization case](../optimization/case_axial.yaml): single-stage Air/Benner, inlet 500000 Pa / 600 K, exit 100000 Pa, mass flow 10 kg/s, inlet angle 0°, blade velocity ratio 0.60 and specific speed 0.70. It defines 12 design variables and one wildcard flaring-angle equality that expands to two scalar constraints, one per blade row. The expanded views demonstrate different tasks within that case:

| Example | Values suitable for the review |
| --- | --- |
| Nordic: stage design | Work fraction 1.000; stator exit angle 70.0°; reaction 0.500; meridional velocity and axial radius ratios 1.000. |
| Graphite: optimization | Maximize total-to-static efficiency. Three of the 12 variables are shown: velocity ratio 0.25 / 0.60 / 1.00; reaction 0.00 / 0.50 / 0.75; exit angle 50° / 70° / 75° (Lower / Initial / Upper). The existing case also defines specific speed bounds 0.20–1.50, solver tolerance 1e-8, feasibility tolerance 1e-6 and a 150-iteration limit. Each row's flaring angle equals 0° with a physical normalization scale of 10°. The dock is idle; no solver run is implied. |
| Blueprint: operating point | Pressure–temperature pair, inlet 500000 Pa / 600 K, exit 100000 Pa, 10 kg/s and 0° inlet flow angle. Working fluid Air is selected in Turbine setup. |
| Paper: Stage 1 / Stator | Solidity 1.121; axial aspect ratio 2.0; thickness/chord 0.300; thickness location/chord 0.300; LE radius/max thickness 0.500; TE thickness/opening 0.070; LE wedge 30°; TE wedge 5°. |

Output numbers and plots on the boards establish hierarchy and visual balance. They must not be used as performance predictions or evidence of convergence.

## Reproducing the boards

Run these commands from the repository root. The builder uses the Python standard library; rendering requires an already installed Playwright package and an existing compatible Chromium browser. Neither script installs packages or downloads a browser.

```text
python -B development/app_design/build_mockups.py
python -B development/app_design/render_mockups.py
```

If needed, select an existing Chromium executable explicitly:

```text
python -B development/app_design/render_mockups.py --browser "C:\path\to\chrome.exe"
```

The builder writes four concept HTML files, `index.html`, and `sidebar_comparison.html`; `design.css` supplies their styles. The renderer rebuilds the HTML and exports ten PNGs: eight full-app boards in `png/`, plus `overview.png` and `sidebar_comparison.png`. It records checks in `output_render/validation.json`. Its browser profile, temporary files and downloads directory stay within `output_render/`. Review pages load local assets only; they can also be opened directly without a server. The rendered boards target desktop review, while a narrow-screen drawer is a proposed future app behavior.

## Repository basis

The concepts were checked against the existing [meanline controls and live callbacks](../../turbodash/apps/turbine_experimental/app.py), [optimization editor](../../turbodash/apps/turbine_optimization/panel.py), [optimization quantity catalog](../../turbodash/apps/turbine_optimization/model.py), [geometry and mesh workspaces](../../turbodash/apps/shared/components/meshing.py), [TurboGrid exports](../../turbodash/apps/shared/components/turbogrid.py) and [core turbine speed handling](../../turbodash/core/turbine.py). Shared controls also appear in the [stable app](../../turbodash/apps/turbine/app.py). This folder contains design proposals only; application behavior remains governed by those source files.
