# Tests

Run the production suite from the repository root:

```powershell
python -B -m pytest tests -q -ra
```

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

Optional real-browser checks use an existing Playwright installation and Chrome
or Chromium. `browser_optimization.py` starts and stops its own app server and
checks clipping, successful/failed workers, multistage application, radial
constraints and session isolation. `browser_workbench.py` checks the redesign
against a separately running local app:

```powershell
python tests/browser_workbench.py --url http://127.0.0.1:8050 --browser "C:/Program Files/Google/Chrome/Application/chrome.exe"
```

The workbench script saves actual-app PNGs and a JSON report under
`development/app_design/implemented/`; browser profiles and downloads stay in
its ignored `output_browser/` directory. Keep the server's temporary directory
inside the checkout when running repository-confined development checks.

`browser_numeric_format.py` starts its own server and checks three-decimal
display padding, additional entered digits, tiny tolerances, focused editing,
dynamic controls and precise YAML round trips. Formatting must not cause an
extra calculation or change numeric model values.
