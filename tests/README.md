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

Study-specific tests remain alongside their analyses in `loss_model_analysis/`.
