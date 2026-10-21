# Implementation and rebuild map

This map connects the reusable design specification to the current optimization workbench. It is a reconstruction guide, not a record of earlier app variants. Read it with [components and layout](components-and-layout.md), [visual style](visual-style.md), [design principles](design-principles.md) and [iconography](iconography.md).

## Runtime source map

Paths below are relative to this directory. The app is built with Python/Dash and uses Bootstrap as a baseline, then app-scoped CSS. Reproducing the appearance in another framework does not require reproducing Dash's component tree.

| Responsibility | Source |
| --- | --- |
| Semantic shell, brand, five-section editor, field factories, metric strip, plot cards and report tabs | [workbench.py](../../turbodash/apps/app_turbine_optimization/workbench.py) |
| Final theme tokens, typography, compact layout, field alignment, component states and breakpoints | [workbench.css](../../turbodash/apps/app_turbine_optimization/assets/workbench.css) |
| Theme persistence, figure restyling, parameter search, numeric formatting, resize/drawer behavior | [workbench.js](../../turbodash/apps/app_turbine_optimization/assets/workbench.js) |
| Configuration, variables, constraints, expert settings, selection rows and optimization outcome presentation | [opt_panel.py](../../turbodash/apps/app_turbine_optimization/opt_panel.py) |
| Searchable quantity/group hierarchy and applicable presets | [opt_selection.py](../../turbodash/apps/app_turbine_optimization/opt_selection.py) |
| Optimization controls and tree/menu baseline styles | [optimization.css](../../turbodash/apps/app_turbine_optimization/assets/optimization.css) |
| Picker keyboard/selection behavior, pending selection handling and terminal follow-scroll | [optimization.js](../../turbodash/apps/app_turbine_optimization/assets/optimization.js) |
| Live calculation callbacks, canonical input state, files, result application, external documentation and favicons | [app.py](../../turbodash/apps/app_turbine_optimization/app.py) |
| Settings and session-owned background work | [opt_model.py](../../turbodash/apps/app_turbine_optimization/opt_model.py), [opt_jobs.py](../../turbodash/apps/app_turbine_optimization/opt_jobs.py) |
| Meanline geometry, efficiency/loss plots, triangles and semantic trace colors | [plotly_turbine.py](../../turbodash/visualization/plotly_turbine.py) |
| Shared Geometry/Mesh layout, native viewer and mesh/export controls | [workspace.css](../../turbodash/apps/shared/assets/workspace.css), [meshing.py](../../turbodash/apps/shared/components/meshing.py), [turbogrid.py](../../turbodash/apps/shared/components/turbogrid.py) |
| Asset registration and application/shared stylesheet ordering | [assets.py](../../turbodash/apps/shared/assets.py) |
| Exact current brand vector files | [blade-ring-light.svg](../../turbodash/apps/app_turbine_optimization/assets/blade-ring-light.svg), [blade-ring-dark.svg](../../turbodash/apps/app_turbine_optimization/assets/blade-ring-dark.svg) |

### The cascade matters

The workbench CSS is scoped to `.td-workbench`. Shared workspace rules and `optimization.css` provide defaults; `workbench.css` supplies the final workbench overrides. Within each registered asset directory the files load in filename order. Copying an earlier standalone rule can produce incorrect density, light backgrounds or larger control spacing.

In particular, the final optimization editor uses inline label/value rows and labels embedded above the values in its three-column bounds fields. It is not the more spacious baseline layout visible near the beginning of `optimization.css`.

Python figure defaults also differ slightly from the running app. `workbench.js` applies the final CSS colors and system-font styling after initial rendering and after replacement figures arrive. Port this synchronization as well as the plotting helper if you need matching screenshots and exports.

## Rebuild sequence

1. **Define state boundaries.** Separate canonical model values, in-progress field drafts, selection context, current results and job identity. Store cosmetic preferences separately from numerical inputs.
2. **Establish the tokens.** Use [assets/tokens.css](assets/tokens.css), adapting its scope to the target application. Apply light/dark preferences before revealing the interface where possible. Use unique storage keys for another package.
3. **Construct the shell.** Match the 72 px header, 92 px rail and 348 px inspector at a wide viewport; independently scroll the editor and result canvas. Add the responsive drawer rather than shrinking all content indefinitely.
4. **Implement one full section.** Include the numbered disclosure, compact labelled field, units, help, selected segment, nested disclosure, keyboard focus and validation message. Compare the open and closed versions in both themes.
5. **Build the information hierarchy.** Add the five numbered sections and their stage/row context. Keep scope switches from destroying drafts. Add search that can reveal hidden, collapsed or expert controls.
6. **Add advanced selection.** Build the searchable tree, checked/partial states and group semantics before adding a large parameter catalog. Use stable row identities. Keep Lower/Initial/Upper and Comparison/Target/Scale aligned even at narrow inspector widths.
7. **Add results.** Use a four-cell metric strip, independent result disclosures, paired charts, full report tables and explicit data freshness. Retain equal physical aspect ratios and common axes where comparison requires them.
8. **Connect computations.** Debounce valid edits; preserve invalid drafts and the last good result. Prevent stale saves. Use revision/job/session ownership so late responses cannot overwrite a newer design.
9. **Add explicit long-running actions.** Pin action and status below the scrollable inspector. Only show success after the accepted design is reflected in both controls and results. Keep the log readable without forcing auto-scroll when a user reads earlier output.
10. **Check the complete experience.** Use the checklist below, compare references, then save an updated design snapshot. A static specimen validates style; the running app validates state and scientific behavior.

For Geometry/Mesh, keep the same outer shell and scope the controls to the active workspace. Native mesh generation remains an explicit optional operation. A static style guide does not validate the native meshing backend.

## Acceptance checklist

| Area | A faithful recreation must demonstrate |
| --- | --- |
| Shell | Header and rail remain stable; the editor scrolls independently; the action stays visible below it |
| Themes | Light blue navigation/white editing surfaces and dark slate surfaces; controls, plots and reports change together |
| Brand | Exact transparent SVG geometry, correctly selected light/dark mark and live lowercase wordmark |
| Density | Five independently collapsible sections; aligned compact labels/values; three optimization values remain on one row |
| Menus | Searchable scope tree, visible connectors, partial groups, no duplicate entries, preserved edited bounds/targets |
| Numeric input | Precision survives edit/save/reload; incomplete drafts remain visible; display formatting never changes the model |
| Scope | Stage/row/workspace changes retain intended values and drafts; type changes follow documented reset semantics |
| State | Late responses cannot restore old inputs; stale data is labelled; success means applied controls and results agree |
| Search | Ctrl/Cmd+K, arrows, Enter and Escape; hidden sections/tabs open and the matching field receives focus |
| Plot fidelity | Correct units, semantic colors, arrowheads, aspect ratios, final-exit budget and illustrative-guide labels |
| Cosmetic changes | Theme, disclosure, search, plot-display mode and navigation do not trigger a numerical solve |
| Keyboard | Logical focus, operable disclosures/menus, accessible help, theme pressed state, resize separator controls |
| Responsive | Check 1680×1080, 1440×1000, 390×844 and 320×740; no horizontal page overflow; small charts scroll within cards |
| Motion | Reduced-motion preference suppresses nonessential transitions/animations |

These are reconstruction requirements. They do not claim a formal accessibility certification or a new full numerical regression result.

## Refreshing the reference

Run commands from the repository root using an environment with the project's dependencies. Native mesh checks additionally require the optional mesh extra. Chrome/Chromium and Playwright are needed only for browser checks.

```powershell
# Open the current workbench.
poetry run app_turbine_optimization --open-browser

# Optional browser workflows against the already running server.
poetry run python tests/browser_workbench.py --url http://127.0.0.1:8050 --browser "C:/Program Files/Google/Chrome/Application/chrome.exe"
poetry run python tests/browser_compact_sidebar.py --url http://127.0.0.1:8050

# Precision check; starts and stops its own local app.
poetry run python tests/browser_numeric_format.py --browser "C:/Program Files/Google/Chrome/Application/chrome.exe"
```

The workbench, compact-sidebar and numeric scripts save reports/captures to ignored `development/app_design/output_workbench/`, `output_compact_sidebar/` and `output_numeric/`. Other optional browser suites also use ignored `output_*` directories. Their temporary profiles, downloads and reports are not part of the blueprint. See [tests/README.md](../../tests/README.md) for regression scope and other commands.

Refresh the curated references from the **same current implementation**, using a clean browser preference state, viewport and representative design. Capture both themes, the mobile drawer, open configuration/expert settings, variables, constraints, tree selection and the two plot groups. Copy only reviewed PNGs to `implemented/`; update its captions and date. Then update the relevant Markdown rule, portable tokens, HTML specimen and the source fingerprints below together.

When removing earlier iterations, preserve durable decisions in the principle/component documents. Do not retain alternative designs as unexplained current specifications. Keep source/runtime behavior outside this documentation cleanup unchanged.

## Reviewed source fingerprint

Reviewed on **21 October 2026** against the local optimization workbench, including uncommitted interface changes. These SHA-256 values describe the source bytes used for this blueprint; they are not a release or a claim that the working tree was clean. Recompute them after adopting future visual changes.

| Source path (repository root) | SHA-256 |
| --- | --- |
| `turbodash/apps/app_turbine_optimization/workbench.py` | `136e75d9c2b88f3c4ee6a41d6e95629bb5f11e84e7305b3f91ec8d3943620b78` |
| `turbodash/apps/app_turbine_optimization/assets/workbench.css` | `51e9bd23a43b55a0fa16923e6651f7741c2f46dac05fec65a21c9e83b36bd327` |
| `turbodash/apps/app_turbine_optimization/assets/workbench.js` | `8fd4827b0a83b629920a1f76dcabd526bf001cb33204c5589501db86c57b06de` |
| `turbodash/apps/app_turbine_optimization/opt_panel.py` | `46d8df0524d576bc752f1565d0f09486c2eb3308ac94c85f5da0e9e67b49d015` |
| `turbodash/apps/app_turbine_optimization/opt_selection.py` | `65e6da4d59a9271e754ee18eaae1f6d5c0422b4cae178b22f688c4a0b5cb5700` |
| `turbodash/apps/app_turbine_optimization/assets/optimization.css` | `caede6a8c929d1922c44a253bb5832b0d84c184bab3a1832771e63dde1aead76` |
| `turbodash/apps/app_turbine_optimization/assets/optimization.js` | `226c86fff6febef997c63a7dbdbe882d8ba6f16d60d5b8ec07b23fa25b2bd0a5` |
| `turbodash/apps/app_turbine_optimization/app.py` | `dcf1984e73ad09055b1f38131f16e9d9cc4dd574cd881876b6f398245a115734` |
| `turbodash/visualization/plotly_turbine.py` | `a3c0faf9d4b9ba619aaea710fdfb800508aa29255c3fed35d31009289ef80c46` |
| `turbodash/apps/shared/assets/workspace.css` | `268b5ad996d022ff2a1f77d1f2537216c549ec201620e29392cbfb5890f01aa6` |
| `turbodash/apps/shared/assets.py` | `9caf7c3d3c2203bf374afb9d705d2b1030219747005da753384af39f2154d4b6` |
| `turbodash/apps/app_turbine_optimization/assets/blade-ring-light.svg` | `449ba383012ef2bd2ccb66b8b5863d3bce203843230afafc75bcb40c4d0c8eed` |
| `turbodash/apps/app_turbine_optimization/assets/blade-ring-dark.svg` | `0545beedfc915da2767d51f8002b08b54cf78eaba3ab69fe37544dd771790c07` |
| `turbodash/apps/app_turbine_optimization/opt_model.py` | `c4b30763bd4fea8118d7b30d3c97c290a4e97b18e2f621a1c789627ed2222efb` |
