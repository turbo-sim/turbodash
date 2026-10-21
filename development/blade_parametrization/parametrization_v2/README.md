# Meridional blade parametrization v2

A standalone NumPy geometry module and Matplotlib slider app. The camberline uses transformed meridional distance and azimuth; Denton thickness uses physical meridional distance. The demo supplies blade count from meridional solidity. There are no JAX, SciPy, nurbspy, or turbodash imports.

## Run

From the repository root:

```powershell
.\.venv\Scripts\python.exe development/new_parametrization/parametrization_v2/demo_blade.py
.\.venv\Scripts\python.exe development/new_parametrization/parametrization_v2/demo_blade.py --blade-type radial
```

Edit the single commented [case.yaml](case.yaml) to change the initial inputs, sample count, or slider ranges. Both dimension blocks belong to the same switchable case. Every geometry parameter is supplied explicitly; the selected blade type determines which dimension block is used.

The axial/radial radio buttons show or hide the appropriate dimension sliders, retaining their values when switching back. The other radio buttons choose angle or slope interpolation. Camberline and geometry controls are on the left; thickness controls are on the right. Reset restores all initial inputs. Interactive edits remain in memory; they do not overwrite the YAML.

The panels show the camberline in transformed coordinates, thickness against normalized camberline arc length, and a physical blade row. Thickness inputs still refer to meridional chord. The row view shows three axial blades or the radial sector from zero to pi/2. Solidity is meridional chord divided by mean circumferential spacing; the app reports the achieved value after rounding the blade count.

## Export without opening a window

```powershell
.\.venv\Scripts\python.exe development/new_parametrization/parametrization_v2/demo_blade.py --no-show --blade-type axial
.\.venv\Scripts\python.exe development/new_parametrization/parametrization_v2/demo_blade.py --no-show --blade-type radial
```

PNG/SVG figures and reference-blade CSV files go to `output/`, or the path supplied with `--output`. Physical CSV coordinates are in metres. Axial coordinates are axial position and unwrapped circumferential distance; radial coordinates are Cartesian positions in the radial plane. Station CSVs also include meridional distance, transformed distance, radius, azimuth, arc length, metal angle, and normalized full thickness.

## Files and conventions

- [blade_geometry.py](blade_geometry.py): ordinary NumPy functions with descriptive names and NumPy-style docstrings. `build_blade` takes an integer blade count; no solidity or loading rule is hidden inside the geometry module.
- [demo_blade.py](demo_blade.py): the solidity-to-count calculation, three views, sliders, and export commands.
- [theory_notes.md](theory_notes.md): conventions, camberline equations, Denton thickness, local metric offsets, and blade spacing.
- [questions_temp.md](questions_temp.md): the discussion and your answers; the later meridional-only and solidity-only decisions supersede the earlier alternatives.

The local thickness offset is exact on the unwrapped axial surface and only first-order accurate when radius varies. The nominal Denton maximum need not be the actual maximum. Invalid mathematical inputs leave the last valid plot visible with an explanation; geometrically unattractive or intersecting blades remain visible. TODOs for the requested literature comparisons are preserved without performing those investigations.
