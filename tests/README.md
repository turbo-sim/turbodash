# Tests

Run the production suite from the repository root:

```powershell
python -B -m pytest tests -q -ra
```

`test_optimization_api.py` contains the API contracts and real axial/radial
checks formerly kept with the first development example. The input cases now
live in `demos/optimization/`. The independent KKT maintenance matrix remains
in `development/optimization/`; see its README for the longer study commands.

Native meshing and renderer integration tests require the optional TurboMoC,
Gmsh, PyVista and Dash VTK dependencies. They skip when those dependencies are
unavailable. The symlink test also skips on Windows without symlink privileges.

The fixtures are independent of the analysis scripts:

- `meanline_geometry_reference.json` retains captured geometry, flow stations
  and physical export hashes. Supported saved geometry and Zweifel inputs
  remain covered. Obsolete loss and efficiency results are excluded.
- `production_loss_reference.json` records revised-model losses and overall
  performance captured before their promotion into the production package.
- `blade_row_case.json` supplies prescribed row-frame inputs for the public
  model assembly checks. Each test receives a fresh copy.
- `ko_trailing_edge_points.csv` contains the 42 KO Figure 14 readings, including
  the held-out points, from the trailing-edge study. Tests evaluate the current
  production fit against these measurements directly.
- `stage_refactor_reference.json` preserves original axial/radial stage results,
  report tables and console output captured before the package relocation.

`test_refactored_layout.py` checks calculation/report parity, the Excel adapter,
all four YAML app cases through Dash's HTTP calculation callbacks, demo runners,
and actual worker module execution from outside the checkout. Tests import the
implementation modules directly; tests of removed compatibility paths are gone.
`test_core.py` remains unchanged and commented out.

Study-specific tests remain alongside their analyses in `development/loss_model_analysis/`.
`test_development_layout.py` checks their repository roots from another working
directory, local report links, and source-paper availability after relocation.

The optimization workbench has focused state/precision tests in
`test_app_workbench.py`, thermodynamic pair-transition tests in
`test_app_fluid_inputs.py`, and channel/mesh presentation tests in
`test_plotting_turbine.py` and `test_mesh_workbench.py`.
`test_app_optimization.py` covers explicit outcome states and confirms that
success is shown only after optimized controls and displayed results agree.
It also checks selection journals, preset overlap, load baselines and preserved
numeric edits. `test_optimization_presets.py` verifies the hierarchical catalogs,
applicable geometry variables, radial bounds, the fixed reference work weight
and two-sided flaring constraints against real model outputs.
The turbine plotting tests check the percentage-point efficiency-penalty budget,
including the final exit kinetic energy and the total `100 * (1 - efficiency_ts)`,
and arrowed velocity triangles in Mach and m/s.

Optional real-browser checks use an existing Playwright installation and Chrome
or Chromium. `browser_optimization.py` starts and stops its own app server and
checks clipping, successful/failed workers, multistage application, radial
constraints and session isolation. Its triangle checks match the independent
plot response to the published result and retain velocity-mode selection across
optimized application. `browser_workbench.py` checks the redesign
against a separately running local app:

```powershell
poetry run python docs/build_docs.py --no-autobuild
python tests/browser_workbench.py --url http://127.0.0.1:8050 --browser "C:/Program Files/Google/Chrome/Application/chrome.exe"
```

The workbench script saves actual-app PNGs and a JSON report under
`development/app_design/implemented/`; browser profiles and downloads stay in
its ignored `output_browser/` directory. Keep the server's temporary directory
inside the checkout when running repository-confined development checks.
It also checks the three independent plot groups, solid endwalls and hidden
geometry legends, velocity-triangle display modes, and the row/final-exit loss
budget in Δη [%] with the app's blue/orange/gray palette. Documentation opens in
a new tab without changing the calculation.
Narrow-screen checks confirm that efficiency, loss and triangle charts scroll
inside their cards without horizontal page overflow.
The optimization action stays at the bottom of the Meanline inspector; tests
check its workspace scope, mobile-drawer visibility and confirmed success state.
The collapsible terminal remains at the bottom of the results column.
`test_documentation.py` verifies same-app Sphinx serving, static assets, route
prefixes and path containment.

The published theory and analytical pages include the original repository notes
verbatim. Check their source fidelity separately:

```powershell
python docs/development/verify_source_fidelity.py
```

`browser_numeric_format.py` starts its own server and checks three-decimal
display padding, additional entered digits, tiny tolerances, focused editing,
dynamic controls and precise YAML round trips. Formatting must not cause an
extra calculation or change numeric model values.

Interaction stress regressions start and stop their own local servers. They use
real controls and compare the rendered graphs with the calculated figures:

```powershell
python tests/browser_stage_stress.py --chromium "C:/Program Files/Google/Chrome/Application/chrome.exe"
python tests/browser_editor_stress.py --chromium "C:/Program Files/Google/Chrome/Application/chrome.exe" --output development/app_design/output_stress_optimization
python tests/browser_interaction_stress.py --chromium "C:/Program Files/Google/Chrome/Application/chrome.exe"
```

These cover stage/type changes, rapid add/remove actions, overlapping calculations,
optimization row identities and precise edits, invalid drafts, property-pair
conversion, and out-of-order responses. Browser profiles, logs and downloads stay
in ignored `development/app_design/output_stress_*` directories. Relevant Python
tests also exercise Dash's HTTP dispatch so wildcard-output serialization errors
cannot hide behind direct callback calls.

The optimization browser scripts use `browser_picker_helpers.py` to select
quantities and groups through the hierarchical checkbox menus. Opening an Add
menu does not itself create an editor row. The editor stress suite verifies
rapid selection and removal while earlier updates are still pending.

`browser_compact_sidebar.py --url http://127.0.0.1:8050` uses a running app and
checks contained sections, inline fields, compact optimization rows and both
themes at desktop, 390 px and 320 px. It also checks a resized 280 px inspector
and menus opened near the bottom of the scrolling area. It saves three
optimization screenshots and `compact_sidebar_browser_checks.json` in the
implemented design folder, with scratch output in `output_compact_sidebar/`.
