# Visual style and theme specification

This is the visual system of the **optimization workbench**, reviewed on 21 October 2026 against the working implementation. Blueprint light and Graphite dark are two presentations of the same application. The reusable [token stylesheet](assets/tokens.css) and [interactive visual guide](style_guide.html) make the system usable without running Dash. The guide is a component specimen; [actual-app references](implemented/README.md) show the real application.

## Visual intent

Keep the working canvas calm so geometry, numbers and active controls receive attention. The composition uses pale or slate surfaces, thin borders, modest corner radii, compact text and a restrained blue accent. Orange identifies contrasting scientific series and stators. Reserve semantic green, amber and red for meaningful outcomes.

The light inspector is very pale, rather than a dark navigation slab. Blue disclosure headers distinguish navigation from white editable bodies. The dark theme uses layered graphite surfaces, readable light text and lighter blue accents. It is not a simple inversion of the light palette.

## Exact application tokens

Names below are production CSS custom properties. Hex values are case-insensitive. Use semantic tokens throughout components instead of embedding a second unrelated palette. `assets/tokens.css` uses `.td-blueprint` as its portable scope; the app uses `.td-workbench`.

| Token | Role | Blueprint light | Graphite dark |
| --- | --- | --- | --- |
| `--td-canvas` | Results canvas | `#EFF3F8` | `#111922` |
| `--td-panel` | Header, cards and report surfaces | `#FFFFFF` | `#19242F` |
| `--td-inspector` | Left editor background | `#F8FAFD` | `#17222E` |
| `--td-input` | Editable control background | `#FFFFFF` | `#111D29` |
| `--td-ink` | Primary text | `#18314D` | `#E7EFF9` |
| `--td-muted` | Labels, units and supporting text | `#62758B` | `#ADBDCE` |
| `--td-line` | Component borders and plot axes | `#D9E2EE` | `#344558` |
| `--td-accent` | Focus, selection and primary action | `#2462D1` | `#88B7FF` |
| `--td-tint` | Selected/hover surfaces | `#EAF1FC` | `#253950` |
| `--td-grid` | Plot grid | `#E6EDF6` | `#2B3B4D` |
| `--td-danger` | Invalid input and failed outcome | `#B33946` | `#FF9CA6` |
| `--td-warning` | Changed, stale or unconfirmed state | `#92601D` | `#E5B575` |
| `--td-success` | Confirmed successful application | `#237548` | `#85D6A4` |
| `--td-shadow` | Floating menus and mobile drawer | `0 12px 40px #18314D15` | `0 12px 40px #00000030` |

Production also provides compatibility aliases: `--td-surface → --td-panel`, `--td-text → --td-ink`, `--td-border → --td-line`, and `--td-surface-alt → --td-tint`.

### Light disclosure hierarchy

These overrides apply **only to light mode**. Do not copy them into Graphite dark.

| State | Treatment |
| --- | --- |
| Closed numbered section, result group or terminal | `--td-tint` header; `#C7D8EF` border |
| Header hover | `#E2EDFC` |
| Open header | `#DCE9FC`; 3 px inset accent at left; bottom separator `#C7D8EF` |
| Closed numbered badge | White fill; accent numeral; `#C7D8EF` outline |
| Open numbered badge | Solid accent fill; white numeral |
| Nested disclosure | `#F3F7FD` fill; ordinary line border; 5 px radius |
| Open nested disclosure | Tint fill; disclosure-line border |
| Selected segmented option | Solid accent fill and white text |
| Editable body/field | White; never reuse the open-header blue as input fill |

These colors correspond to `--td-disclosure-line`, `--td-disclosure-hover`, `--td-disclosure-open` and `--td-disclosure-nested`. In dark mode, section emphasis comes from the dark panel/tint rules; the blue light-only accents are absent.

## Typography and numeric presentation

| Use | Reproduction rule |
| --- | --- |
| App base | `13px/1.5 "Segoe UI", -apple-system, BlinkMacSystemFont, sans-serif` |
| Wordmark | Live lowercase text, 25 px, weight 650, tracking −1 px; [responsive brand rules](iconography.md) |
| Canvas title | 26 px, weight 600; 21 px at ≤600 px |
| Inspector title | 20 px, weight 600 |
| Numbered section heading | 12 px, weight 600 |
| Ordinary field label and value | Labels 11 px/1.4, weight 500; compact field values 12 px; [component details](components-and-layout.md) |
| Secondary caption / unit | Typically 10–11 px; muted ink |
| KPI number | 27 px in the current metric strip; 23 px at ≤1250 px, 20 px at ≤600 px |
| Eyebrow | 9 px uppercase with 1.4 px letter spacing |
| Technical output | `Consolas, "Cascadia Code", monospace`; terminal 12 px/1.65 |

Keep units next to the quantity and stable numeric columns. Use tabular numerals where measurements align. Continuous values display at least three decimals, preserve additional supplied digits and useful scientific notation, and retain full model precision. Integer counts stay integers. Do not pad or reformat while a user is typing. Percentage KPI displays use two decimals; rpm uses whole numbers; power uses two decimals and adaptive W/kW/MW.

The plot helper has an Arial fallback, but the running workbench overrides figure fonts to `Segoe UI, sans-serif`. A renderer outside the workbench will therefore need the same override to match the UI.

## Spacing, borders and elevation

Use the measured component dimensions in [components-and-layout.md](components-and-layout.md), rather than assuming a universal spacing grid. The implementation commonly uses 6–10 px inside compact controls, 12–18 px inside cards, and 20–25 px around major regions.

Most surfaces have a 1 px border. Main sections and buttons use approximately 5–8 px radii; small input segments use smaller radii. Cards are flat, not floating. Reserve `--td-shadow` for overlays, menus, search results and the narrow-screen drawer. The overview metric strip uses a 3 px accent top border. Use inset separators rather than enclosing every number in its own heavy card.

Keyboard focus uses a 2 px accent outline with a 3 px offset. Hover, active, selected, expanded, disabled and invalid states each need an explicit visual treatment. Keep the state label or marker as well as color; a red border alone does not explain an error.

## Plot palette and scientific meaning

Plots are part of the interface, so their backgrounds, axes, hover labels, annotation text and legends change with the theme. In the running workbench, those framing colors come from the application tokens above. Preserve ranges and numerical data when restyling; a theme switch must not rerun the solver.

| Scientific role | Light | Dark |
| --- | --- | --- |
| Rotor / blue line series | `#1F77B4` | `#69A9E2` |
| Stator / orange line series | `#FF7F0E` | `#FFAD5C` |
| Hub and shroud | `#647892` | `#A0B2CA` |
| Illustrative guide paths | `rgba(68,118,188,0.48)` | `rgba(128,183,255,0.58)` |
| Solid region fill | `rgba(100,120,146,0.12)` | `rgba(160,178,202,0.12)` |
| Channel fill | `rgba(36,98,209,0.05)` | `rgba(136,183,255,0.08)` |

Loss-chart bars have a separate role-based palette:

| Loss component | Light | Dark |
| --- | --- | --- |
| Profile | `#2462D1` | `#88B7FF` |
| Trailing edge | `#4B84E5` | `#6D9DE8` |
| Secondary | `#7AA6EC` | `#B3CEFF` |
| Tip clearance | `#FF7F0E` | `#FFAD5C` |
| Incidence | `#F5AD66` | `#F3C48F` |
| Final exit kinetic energy | `#647892` | `#A0B2CA` |
| Other / unallocated | `#8995A5` | `#BCC6D4` |

Use trace `meta.theme_role` to keep role-specific styles stable when switching themes. Maintain the same colors in legends and traces. The Python plotting helper defines slightly different standalone frame defaults; `workbench.js` supplies the final app appearance. Port both layers deliberately.

Geometry uses solid hub/shroud lines, translucent row fills and orange/blue outlines. Hide geometry legends when hover already identifies boundaries. Preserve physical aspect ratio, common stage axes and all station extents. Initial/reset meridional views include zero radial baseline; deliberate user zoom is retained. Guide streamlines are illustrative and must be described that way.

Efficiency is plotted in percent. The loss budget is an efficiency penalty in percentage points, labelled `Δη [%]`, and sums to `100 × (1 − efficiency_ts)`. Include the final exit kinetic-energy contribution once. Velocity triangles have arrowheads, shared stage extents and a Mach / m/s display selector. Retain Plotly zoom/pan/reset/export; hide the Plotly logo.

## Theme behavior to reproduce

1. Follow `prefers-color-scheme` on first use; persist only an explicit choice.
2. Keep the Light/Dark control at the far right of the header and expose selection with `aria-pressed`.
3. Apply tokens to editors, reports, tooltips, dropdowns, dialogs and figures together.
4. Preserve form values, focus/selection context, running jobs and manual plot zoom.
5. Reapply the current theme after server-rendered figures or controls arrive. Theme selection must survive ordinary edits without a flash back to light mode.
6. Respect reduced-motion preferences. The production workbench disables its animations, transitions and smooth scrolling for that preference.

The app uses `turbodash.workbench.theme` for persisted theme and `turbodash.workbench.inspector-width` for width. Give another package its own storage keys. The standalone guide uses a separate key and never modifies application preferences.

## Source of truth

The exact styles are in [workbench.css](../../turbodash/apps/app_turbine_optimization/assets/workbench.css), [optimization.css](../../turbodash/apps/app_turbine_optimization/assets/optimization.css), [workbench.js](../../turbodash/apps/app_turbine_optimization/assets/workbench.js) and [plotly_turbine.py](../../turbodash/visualization/plotly_turbine.py). The [implementation map](implementation-map.md) records the reviewed source fingerprint and the rebuild/acceptance process.
