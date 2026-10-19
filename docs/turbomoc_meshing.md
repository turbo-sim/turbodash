# TurboMoC geometry and meshing integration

The experimental Turbodash app connects its solved meanline blade profiles
directly to the `turbo_moc` package. It does not copy the Gmsh mesher or run a method-of-
characteristics nozzle calculation to construct a turbine blade.

## Assessment of the two repositories

| Responsibility | Existing implementation | Integration choice |
| --- | --- | --- |
| Turbine thermodynamics and stage sizing | `turbodash/core/turbine.py` and `core/stage.py` | Keep the existing solver and `stages_performance` result schema. |
| Blade profiles | `turbodash/geometry/blade.py` | Construct the same curvature-based camberline and NACA thickness profile used by the existing views. Use absolute stator angles and relative rotor angles. |
| Existing 3D export | `turbodash/io/turbogrid.py` and `io/turbogrid_assembly.py` | Preserve the TurboGrid workflow, including its separate free-vortex span model. It produces geometry for another mesher, not an in-app CFD mesh. |
| General blade-row geometry | `turbo_moc.geometry.blade_row.Design`, `meridional_channel`, `rotor_blade` | Adapt Turbodash's three closed profile segments to SciPy B-splines, then reuse passage construction, containment checks and annular mapping. |
| 2D reference mesh and 3D sweep | `turbo_moc.meshing.mesh_3d` | Call `reference_mesh`, `load_section`, `volume_mesh` and `export_mesh` directly. These reuse the package's sizing, inflation, recombination, periodic correspondence, volume validation and CGNS audit. |
| 3D surface models | `turbo_moc.visualization.geometry_preview.preview_geometry` | Reuse closed surfaces for previews and binary STL download. Version 1.2.0 caps the blade at its physical endwalls; export its coordinates directly. |
| Quality and cell inspection | `turbo_moc.meshing.inspection` | Reuse saved cell quality, physical slicing, spanwise layer extraction, threshold filtering and worst-cell selection. |
| 3D rendering | TurboMoC's `studio.js` and `mesh_edges.js`, using Dash VTK | Serve the installed package's renderer directly, including actor reuse, camera controls, original polygon edges, and cell-quality colors. Keep Plotly for the 2D and meanline plots. |
| Experimental app runtime | `turbodash/apps/app_turbine_experimental/app.py` | Keep meanline controls mounted across three tabs; use a separate subprocess for each Gmsh job. The stable meanline app remains in `apps/app_turbine/app.py`. |

The current TurboMoC checkout already has general spline-based meridional
channels and flared passage support. This first Turbodash adapter deliberately
uses the narrower cylindrical axial case. The main reusable entry point is
`Design`; the higher-level TurboMoC nozzle/stator workflow expects a converged
MoC solution and would be the wrong source of a meanline turbine profile.

The optional dependency is pinned to the published PyPI package `turbo_moc==1.2.0`
so Turbodash's wheel and source distribution can be published on PyPI. The adapter uses
`rotor_blade._flow_domain`, a private but generic contour-based passage builder;
the pin and integration tests protect this boundary until TurboMoC exposes a
public generic-profile constructor. No changes to the TurboMoC source are needed.

With inflation enabled, the adapter selects Gmsh's Frontal-Delaunay algorithm 6
before recombination. Algorithm 8 produced an invalid fluid topology for the
rotor integration case with TurboMoC 1.2.0; algorithm 6 passes the same topology
checks. Without inflation, the adapter retains algorithm 8. Version 1.2.0 does
not expose the later `recombine_minimum_quality` control, so it is not passed.

The 3D geometry and mesh viewers use VTK, as in TurboMoC.
Turbodash serves the two renderer scripts from the installed package through
an allowlisted route. Version 1.2.0 supports drag rotation, zoom and camera fitting;
it does not include the orientation cube available in later development versions.
The views stay mounted across tab changes; TurboMoC manages camera fitting,
resizing and stable actors. No copy of its VTK rendering implementation is
maintained in Turbodash.

## Geometry assumptions and units

Each selected stator or rotor is an independent annular periodic passage with
cylindrical hub and shroud. Its mean radius and blade height are the arithmetic
averages of the row's inlet and outlet station values. If a meanline solution
changes radius or height along the row, the metadata and app explain that its
endwall flare has been flattened. The meanline calculation itself is unchanged.

The midspan blade metal angles and physical meridional chord are retained across
the span. Circumferential pitch follows the integer blade count and local
radius. This model does not include free-vortex twist, tip clearance, coupled
row interfaces, or a connected multistage CFD domain. Radial turbine results
remain usable in Meanline but are rejected by this adapter.

| Artifact | Length units | Rotation axis |
| --- | --- | --- |
| Turbodash solved results | metres | Meanline convention |
| TurboMoC design, STL, native Gmsh mesh, plots | millimetres | Z |
| Exported CGNS | metres | Z |
| Existing TurboGrid curves | millimetres | X |

The reference chart is `(tangential_mm, -axial_mm)`; the 2D app plot shows it
with conventional axial and tangential axes. The exported STL is a closed blade
surface, not a CAD STEP solid or a fluid volume.

## Installation and app workflow

From the Turbodash checkout, using the intended virtual environment:

```bash
python -m pip install -e .
python -m pip install -e ../method_of_characteristics
python -c "import turbodash; turbodash.launch_app_experimental()"
```

The second command installs the companion checkout in editable mode, so its
meshing implementation is called directly. Alternatively, install the published
TurboMoC 1.2.0 optional dependency with `poetry install -E mesh` or
`python -m pip install -e ".[mesh]"`. The `turbodash-experimental` console command
launches this three-tab workspace. The existing `turbodash` command and
`turbodash.launch_app_turbine()` launch the stable meanline turbine app;
`turbodash-stage` retains the original single-stage app.
The Render/Procfile entrypoints use the stable turbine app without the mesh
extra. To host the experimental workspace separately, install the mesh extra
and use `gunicorn turbodash.apps.app_turbine_experimental.app:server`.

1. **Meanline:** calculate or load an axial turbine, including its blade geometry.
   The existing performance plots, tables and expandable theory documentation
   remain here.
2. **Turbine geometry:** expand the stage and blade-row parameter sections in
   the left sidebar. These edit the same design parameters as Meanline. The
   selected row's blade-to-blade passage and cylindrical meridional channel
   appear side by side, with the whole-turbine VTK view below. Use **Configure
   display** beside the 3D view to show or hide rows, set displayed blade counts,
   and choose boundary surfaces. These display preferences do not edit the
   design. Repeated blades follow each row's angular pitch; axial assembly gaps
   are illustrative and do not define coupled mesh interfaces. Model ZIP and
   TurboGrid export remain available.
3. **Mesh:** expand a blade row's section in the left sidebar, choose element
   sizes and spanwise layers, optionally enable blade inflation, then select
   the row and click **Generate mesh**. Each row retains its own mesh settings
   when switching between rows. Blank size fields use chord-based defaults.
   The actual reference mesh and meridional mesh projection appear side by side
   above the VTK viewer. Use the display and inspection controls to view
   boundary faces, physical slices, individual span layers, worst cells, or a
   quality threshold; then download the mesh ZIP. Inspection and display
   changes do not run the mesher.

Both workspaces follow TurboMoC's layout: row parameters on the left, paired
2D views above the 3D viewport, and display controls beside it. The parameter
sidebar and plot area scroll independently on desktop; narrow windows stack
the plots and display controls. Geometry and mesh meridional plots use the
same cylindrical passage as the exported model; the original meanline channel
remains in Meanline. The mesh plots reuse TurboMoC's plotting functions and
saved mesh artifacts.

Higher skewness and edge aspect ratio indicate poorer cells; lower
orthogonality indicates poorer cells. Colors and hover values come from the
original volume cells. Inspection does not remesh or interpolate quality
metrics. Excessively large browser selections are reported explicitly rather
than silently sampling mesh faces.

Changing the solved design, live meanline inputs, selected row, or its mesh settings
invalidates the displayed mesh and its download. A failed meanline calculation
cannot make an old mesh appear current. Generating a new mesh always requires
an explicit click. Invalid or overlapping profiles, excessive inflation and
invalid volume cells produce visible errors rather than a successful download.
If an inflated reference mesh fails topology validation, refine the blade
element size or reduce the inflation thickness. Inflation is off by default.

## Downloads and Python API

The model ZIP contains `blade.stl`, `design.json` and
`geometry_metadata.json`. The mesh ZIP contains native Gmsh reference/volume
meshes, `passage.cgns`, the saved TurboMoC design and numerical arrays,
`mesh_report.json`, `mesh_summary.json`, boundary surfaces and a manifest.
It also records the source profile, resolved mesh configuration, dependency
version and source fingerprint so the output can be traced to its inputs.
Boundary names are `blade`, `hub`, `shroud`, `inlet`, `outlet`,
`periodic_left` and `periodic_right`. CGNS includes rotational periodic
correspondence. Its boundary type labels are defaults; operating boundary
conditions and inter-row coupling still belong to the CFD solver setup.

```python
from pathlib import Path
import yaml
from turbodash.meshing.turbomoc import generate_mesh, geometry_zip, inspect_mesh

results = yaml.safe_load(Path("demos/basic_demo/demo_axial_turbine.yaml").read_text())
Path("stator_model.zip").write_bytes(geometry_zip(results, stage_index=0, row="stator"))

# Run this from a standalone process, not concurrent threads: Gmsh is global.
manifest = generate_mesh(
    results,
    "output/stage_01_stator",
    stage_index=0,
    row="stator",
    options={"span_layers": 12, "inflation_enabled": False},
)
inspection = inspect_mesh("output/stage_01_stator", mode="worst", n=20, metric="skewness")
```

Use a fresh output directory for each mesh. The web app handles this itself:
each subprocess receives an immutable JSON snapshot and an isolated directory.
Only random job handles and status summaries are sent to the browser. The app
allows two concurrent mesh workers on one host and stops jobs after 30 minutes.
Completed artifacts expire after 24 hours and are cleaned when later jobs
start, so download outputs you want to keep. A multi-host deployment needs a
shared job/artifact service; the current file-backed cache supports web workers
on a single host.

The tests under `tests/test_meshing.py`, `test_mesh_jobs.py`,
`test_plotting_mesh.py` and `test_app_meshing.py` cover the adapter, real mesh
generation and export, artifact lifecycle, visualization, and app state guards.
`test_visualization_vtk.py` and `test_vtk_assets.py` check the VTK data adapters
and direct reuse of the installed renderer.
Mesh integration tests require the optional dependencies.

The 0.7.0 integration was checked against published TurboMoC 1.2.0, Gmsh 4.15.2,
PyVista 0.49.1 and VTK 9.7.1. The full suite passed 418 tests, with one Windows
symlink-permission skip. All four YAML-driven apps also calculated from the
built wheel, and the experimental app served both renderer bundles.

Real meshes for the saved axial demo contained 3,753 hexahedra plus 573 prisms
for the stator, and 62,994 hexahedra for the rotor with inflation. Both passed positive-Jacobian, periodic
correspondence and CGNS boundary-partition checks. These checks establish
topology and export consistency; mesh resolution still requires a convergence
study for the intended CFD calculation.
