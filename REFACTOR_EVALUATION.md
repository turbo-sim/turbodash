# Refactor evaluation

The package refactor and the subsequent compatibility cleanup are complete.
Turbodash is organized by responsibility, all four applications remain available,
and repository callers use the current module paths. The numerical implementations
and saved analysis results were preserved during these changes.

This document replaces the proposal with a summary of the completed work and an
evaluation against your requests. Detailed results are recorded in the
[verification report](docs/refactoring_validation.md) and its
[machine-readable evidence](docs/refactoring_validation.json).

## Checklist against your requests

- [x] **Implement the package refactor.** Grouped calculations, geometry,
  meshing, visualization, reporting, file interfaces and applications into
  their own packages. Extracted stage reporting and shared row-section
  construction; separated assembly geometry, plotting and its CLI.
- [x] **Keep all four applications working.** Retained the stable turbine,
  experimental turbine, original stage and dimensionless efficiency apps.
  Each passed calculation and browser checks from the rebuilt wheel.
- [x] **Provide a runner and launch YAML for each app under `demos/apps/`.**
  All four [demo directories](demos/apps/README.md) contain `run.py` and
  `case.yaml`; the runners resolve their adjacent YAML files.
- [x] **Check the tests and behavior after refactoring.** The final production
  run recorded **411 passed, 8 skipped**. Stage calculations and report output
  match pre-refactor captures; geometry, exports, callbacks and worker lifecycle
  checks pass. The wheel was also checked outside the source checkout.
- [x] **Leave `tests/test_core.py` commented out.** Its file hash is unchanged.
  Stage parity checks live separately in
  [test_refactored_layout.py](tests/test_refactored_layout.py).
- [x] **Allow breaking module imports and clean up the leftovers.** Removed
  23 forwarding modules and `_compat.py`, removed root module aliases, and
  made `geometry` and `meshing` ordinary packages. Removed 14 tests that only
  exercised compatibility. Current modules have one implementation owner.
- [x] **Update repository callers and integration paths.** Migrated remaining
  imports in tests, demos and analysis scripts; updated documentation, console
  targets, deployment configuration and the worker command. Excel instructions
  use the current `turbodash.io` modules.
- [x] **Keep the analysis reports usable after the moves.** Refreshed affected
  source inventories through the study workflows and checked source guards.
  All **158 analysis tables** retain identical decompressed contents.
- [x] **Use SVG figures in the analysis reports.** Exporters generate SVG only;
  **214 figures** are embedded across **27 Markdown reports**, including the
  three-column stage comparison. Original paper PDFs remain digitization inputs.

The eight skips are the existing optional/platform cases: six require Gmsh,
one requires Dash VTK, and one requires Windows symlink privileges. Their
limitations are listed under remaining verification below.

## What changed

Calculation code and report generation now have separate owners. Shared blade
row-section construction belongs to geometry and is used by exporters and
assembly construction. Assembly layout, plotting and command-line handling
are also separated. Apps retain their own servers and callbacks; reusable
meshing and TurboGrid panels live under `apps/components/`.

```text
turbodash/
    __init__.py              # scientific functions, plotting style, launchers
    math.py
    loss_models/
    core/                    # stage and turbine calculations
    geometry/                # blade profiles, row sections, assembly layout
    meshing/                 # optional TurboMoC adapter
    visualization/           # Matplotlib, Plotly, VTK adapters and assets
    reporting/               # stage and turbine reports
    io/                      # TurboGrid exports and Excel interfaces
    apps/                    # four apps, startup configuration, workers, panels
    assets/
    docs/
loss_model_analysis/         # research, numerical evidence and SVG reports
tests/
demos/
docs/
```

`geometry` and `meshing` are ordinary packages. They do not forward attributes
or monkeypatches to implementation modules. Patch and import the implementation
that owns the function, such as `turbodash.meshing.turbomoc`.

## Module migration map

The paths in the first column have been removed. Update downstream imports and
launch commands to the corresponding current modules.

| Removed flat module | Current owner |
| --- | --- |
| `core_stage` | `core.stage` for calculations; `reporting.stage` for reports |
| `core_turbine` | `core.turbine` |
| `geom_blade` | `geometry.blade` |
| `graphics` | `visualization.matplotlib_style` |
| `plotting_mpl` | `visualization.matplotlib_turbine` |
| `plotting_plotly_stage`, `plotting_plotly_turbine` | `visualization.plotly_stage`, `visualization.plotly_turbine` |
| `plotting_mesh` | `visualization.plotly_mesh` |
| `visualization_vtk`, `vtk_assets` | `visualization.vtk`, `visualization.vtk_assets` |
| `reporting_utils` | `reporting.turbine` |
| `export_turbogrid`, `export_turbogrid_assembly` | `io.turbogrid`, `io.turbogrid_assembly` |
| `turbogrid_assembly` | `geometry.assembly` for layout; `visualization.matplotlib_assembly` for plots; `apps.assembly_preview` for the CLI |
| `excel_interface_stage`, `excel_interface_turbine` | `io.excel_stage`, `io.excel_turbine` |
| `app`, `app_lite` | `apps.stage`, `apps.stage_efficiency` |
| `app_turbine`, `app_experimental` | `apps.turbine`, `apps.turbine_experimental` |
| `app_meshing`, `app_turbogrid` | `apps.components.meshing`, `apps.components.turbogrid` |
| `mesh_jobs` | `apps.mesh_jobs` |

All names are relative to `turbodash`. Direct functions formerly forwarded by
`turbodash.geometry` now live in `geometry.blade`; those formerly forwarded by
`turbodash.meshing` live in `meshing.turbomoc`. Root aliases such as `td.mpl`,
`td.plotly` and `td.core_turbine` are removed. The root calculation/report
functions and `launch_app_*` helpers remain direct convenience exports.

For example:

```python
from turbodash.core.turbine import compute_turbine_performance
from turbodash.geometry.blade import compute_blade_coordinates_cartesian
from turbodash.reporting.turbine import flow_stations_table
from turbodash.visualization import matplotlib_turbine
```

## Application evaluation

| Application | Module | Launch command | Browser check |
| --- | --- | --- | --- |
| Stable turbine | `turbodash.apps.turbine` | `turbodash` | 5 plots; no browser errors |
| Experimental turbine workspace | `turbodash.apps.turbine_experimental` | `turbodash-experimental` | 8 plots; no browser errors |
| Original stage | `turbodash.apps.stage` | `turbodash-stage` | 4 plots; no browser errors |
| Dimensionless efficiency | `turbodash.apps.stage_efficiency` | Its demo runner or `main()` | 2 plots; no browser errors |

Deployment uses `gunicorn turbodash.apps.turbine:server`. Isolated meshing jobs
use `python -m turbodash.apps.mesh_jobs REQUEST_JSON`. The stable app continues
to start without the optional meshing/VTK stack. See the
[application guide](turbodash/apps/README.md) for resources and startup cases.
External workbooks or scripts importing the removed Excel modules must select
the corresponding `turbodash.io` module; workbook UDF names are unchanged.

## Preserved behavior

The numerical models, geometry conventions, continuous sizing, physical integer
realization and signed efficiencies are unchanged. The two meanline APIs remain distinct. Loss models and `math.py`
retain their paths. Reporting and export helpers have one implementation owner.
No model dispatcher, `src/` layout, generic utility layer or app factory is added.

The final count of 411 passing tests follows removal of 14 compatibility-only
checks from the earlier 425-test run. Calculation/report parity, application
callbacks, job lifecycle, geometry, exports and scientific loss checks remain.

Release 0.7.0 pins the optional mesh dependency to published TurboMoC 1.2.0
so the package can be published on PyPI. The adapter uses that version's two
renderer scripts and physical STL coordinates, and selects algorithm 6 for
inflation meshes. With the optional stack installed, 418 tests passed and one
Windows symlink case was skipped. All four apps also calculated from the built
wheel. See the [meshing integration guide](docs/turbomoc_meshing.md) for the
version-specific behavior and native mesh results.

## Remaining verification

- [x] Run the six native meshing integration cases with Gmsh and the required
  native dependencies; passed against TurboMoC 1.2.0 for release 0.7.0.
- [x] Run the Dash VTK renderer integration case with Dash VTK installed; passed.
- [ ] Run the symlink artifact-access case with Windows symlink privileges.
- [ ] Verify the Excel adapters inside Excel. Python import and configuration
  lookup checks passed; Excel-host execution was not exercised. External
  workbooks still need their module configuration updated to the current paths.

These are verification limits, not unfinished package moves. The results above
come from the completed refactor, cleanup checks and subsequent 0.7.0 release
compatibility checks.
