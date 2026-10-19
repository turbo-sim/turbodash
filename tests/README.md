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
