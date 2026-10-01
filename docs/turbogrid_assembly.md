# Whole-turbine assembly export

The web download now offers an optional assembly mode alongside independent-row
export. It does not change turbine calculations, blade models, or the standalone
rotor exporter. Assembly mode exports globally positioned axial blade rows with
matching endwalls and boundary locations. It does **not** create meshes or
configure a coupled CFX simulation.

## Download from the app

1. Calculate or load an axial design and wait for the results to update.
2. In the TurboGrid export card, select **Whole-turbine assembly**.
3. Enter **Row gaps [mm]**, one comma-separated value per adjacent row pair.
   For example, `10` for a single stage or `10, 15, 10` for two stages.
   These are examples, not recommended clearances. No gap is prefilled.
4. Click **Download TurboGrid ZIP**. The button remains disabled until the number
   of positive, finite gaps matches the latest successfully calculated design.
5. Extract the ZIP and read its `README.txt` before importing the row files.

The download uses the latest successful results, not uncomputed or failed input
changes. Gaps are export-only settings; they do not change the meanline model,
normal turbine plots, or saved design YAML. They are recorded in `assembly.yaml`.
Changing the number of stages requires updating the gap list.

The archive contains:

```text
README.txt
assembly.yaml
row_boundaries.csv
common/hub.curve
common/shroud.curve
stage_01/stator/BladeGen.inf
stage_01/stator/profile.curve
stage_01/stator/hub.curve
stage_01/stator/shroud.curve
stage_01/stator/metadata.yaml
stage_01/rotor/...
stage_02/...
```

Each `.inf` references the curve files in its own folder. Blade coordinates
already include the row translation: **do not apply `x_origin` again**.
Profile and endwall coordinates are millimetres about the X rotation axis.
The boundary CSV also uses millimetres; YAML metadata uses metres and rad/s.

## Required TurboGrid setup

- Import and mesh each row separately. Set both endwall **Curve Type** options
  to **Piece-wise linear**, not the default Bspline. Use the same Flowpath
  Parameterization Type for connected rows.
- Each row's endwalls are cropped from a common polyline. Adjacent rows meet at
  the same axial plane and radii, with the same gap-segment slope. Full endwalls
  in `common/` are provided for reference, not used by the row `.inf` files.
- Set/check the complete row mesh inlet and outlet against `row_boundaries.csv`.
  The `.inf` files do not configure mesh interface placement automatically.
- For passage-only meshes, place boundaries at those planes and disable separate
  inlet/outlet domains. If separate domains are enabled, their passage interfaces
  must lie inside the row bounds. The complete mesh must still terminate at the
  listed planes. Fully extending a passage interface to a curve end leaves no
  space for an additional domain beyond it.
- Verify/save each connecting interface and reuse it for the adjacent row.
  Inspect actual meshed interfaces and mesh quality before setting up CFX.

These settings follow the [Ansys stage-interface guidance](https://ansyshelp.ansys.com/public/Views/Secured/corp/v261/en/tg_user/i016457045820452615547.html),
which also describes the alternative of separate endwall curves meeting without
overlap. Our common-polyline subsets avoid introducing a slope change at the
mid-gap junction, provided Piece-wise linear representation is selected.

Combine the row meshes in CFX with appropriate stationary/rotating domains and
interfaces. Unequal blade counts require appropriate pitch handling. This ZIP
does not configure boundary conditions, fluid properties, or solver settings.

## Run from the repository root

Use the project's Python environment and a YAML containing `stages_performance`:

```text
python -m turbodash.turbogrid_assembly demos/cyclopentane_case/cyclopentane_8.yaml --row-gaps 0.01 --output-dir demos/cyclopentane_case/output/assembly_preview
```

The 0.01 m clearance is an example, not a recommended aerodynamic spacing.
Specify one positive gap per adjacent pair. A two-stage turbine needs three
values: stator 1 to rotor 1, rotor 1 to stator 2, and stator 2 to rotor 2.
No plotting-gap default is used. Output files in the chosen directory are
overwritten when rerunning the command.

Outputs:

- `assembly.yaml`: global row origins, blade bounds, row-domain bounds, blade
  counts, angular speeds in rad/s, shared endwall points, and interface locations.
- `assembly_preview.png`: meridional endwalls, shaded axial row envelopes, and
  proposed interface planes. These shaded regions are not blade surfaces.

All layout lengths are metres; the rotation axis is X. The first stator's
nominal leading edge stays at X = 0. Subsequent row origins are translations
to apply to existing local blade coordinates.

## Python API

```python
from turbodash.turbogrid_assembly import build_assembly_layout, plot_assembly_layout

layout = build_assembly_layout(results, row_gaps=[0.01])
figure, axes = plot_assembly_layout(layout)
```

To create a positioned ZIP directly, with gaps in **metres**:

```python
from turbodash.export_turbogrid_assembly import export_turbogrid_assembly_zip

zip_bytes = export_turbogrid_assembly_zip(results, row_gaps=[0.01])
```

`build_turbogrid_assembly_files` in the same module returns a filename-to-text
dictionary. Both functions are in-memory and leave input results unchanged.

The layout uses the same free-vortex section generator as the row exporter.
Default geometry options remain `curvature_based` and `NACA`, with 11 span
sections and `profile_points=800`. These options can be passed to the Python API.
Both the layout and assembly export sample the full span including hub and
shroud. Unlike independent-row export, assembly profiles have no 1% span inset.
No tip clearance is prescribed by this export.
The nominal chord endpoints are also included in each row envelope. Therefore
the actual sampled blade-to-blade clearance is at least the requested gap.
This is not a continuous-surface collision guarantee; resolution must be checked
for the geometry being exported.

Common endwalls follow station radii linearly along each nominal chord, remain
constant across leading/trailing edge overhangs, and transition linearly across
row gaps. This matches the existing section generator's radial mapping without
introducing a new blade shape. Corners in these piecewise-linear contours are
intentional at this stage; smoothing requires a compatible blade-span mapping.

Interfaces are at gap midpoints. Adjacent domain bounds share exactly the same
axial location and common endwall radii. Overall inlet/outlet extensions are
one first-stator chord and two last-rotor chords by default, controlled by
`--inlet-extension-fraction` and `--outlet-extension-fraction`. Internal row
domains stop at shared interfaces, rather than retaining overlapping extensions.

Inspect the preview and choose physical clearances before meshing. Layout
metadata does not itself apply TurboGrid interface settings. The independent-row
ZIP remains available unchanged by selecting **Independent rows** in the app.
