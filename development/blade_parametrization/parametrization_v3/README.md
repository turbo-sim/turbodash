# Meridional blade parametrization v3

A standalone NumPy geometry module and Matplotlib slider app. V3 adds local leading-edge asymmetry to the Denton thickness construction while preserving total thickness. The camberline uses transformed meridional distance and azimuth; Denton thickness uses physical meridional distance. Earlier versions remain unchanged. There are no JAX, SciPy, nurbspy, or turbodash imports.

## Run

From the repository root:

```powershell
.\.venv\Scripts\python.exe development/new_parametrization/parametrization_v3/demo_blade.py
.\.venv\Scripts\python.exe development/new_parametrization/parametrization_v3/demo_blade.py --blade-type radial
```

Edit the single commented [case.yaml](case.yaml) to change the initial inputs, sample count, or slider ranges. Both dimension blocks belong to the same switchable case. Every geometry parameter is supplied explicitly; the selected blade type determines which dimension block is used.

The axial/radial radio buttons show or hide the appropriate dimension sliders, retaining their values when switching back. The other radio buttons choose angle or slope interpolation. Camberline and geometry controls are on the left; thickness controls are on the right. Reset restores all initial inputs. Interactive edits remain in memory; they do not overwrite the YAML.

The panels show the reference camberline in transformed coordinates, total thickness and individual side offsets against normalized camberline arc length, and a physical blade row. Thickness inputs still refer to meridional chord. The row view shows three axial blades or the radial sector from zero to pi/2. The first two plots retain their zero-coordinate conventions and equal aspect ratios. Solidity is meridional chord divided by mean circumferential spacing; the app reports the achieved value after rounding the blade count.

## Leading-edge control

The case starts from the reported leading-edge example with `leading_edge_asymmetry: 0.15`. Move the new right-hand slider to compare:

- **Zero:** recover the symmetric v2 construction with the same other inputs.
- **Positive:** reduce the plus-side offset and increase the minus-side offset.
- **Negative:** reverse the transfer.

The allowed range is `[-1, 1]`. The sign is manual: plus/minus do not automatically mean pressure/suction or concave/convex. The transfer vanishes at the tip and returns smoothly to zero by twice the leading-edge blend fraction, capped at the full chord. With no leading-edge blend it is disabled. An ordinary rounded tip retains its tangent and limiting radius. The local midpoint shifts, so the input camberline becomes a reference curve in the affected region.

## Compare geometry and curvature

```powershell
.\.venv\Scripts\python.exe development/new_parametrization/parametrization_v3/demo_asymmetry.py
.\.venv\Scripts\python.exe development/new_parametrization/parametrization_v3/demo_asymmetry.py --no-show
```

This uses the same YAML and overlays symmetric and adjusted blades, with curvature plots for both sides. For the supplied case, the plus-side peak curvature magnitude times chord falls from **23.25 to 6.73**, while the minus-side peak rises from **7.22 to 8.08**. These sampled peaks cover meridional fractions 0.01862 through 0.3724 and exclude the sensitive tip interval. This demonstrates a useful adjustment, not a universal curvature guarantee: excessive or wrong-sign asymmetry can worsen the shape.

## Export without opening a window

```powershell
.\.venv\Scripts\python.exe development/new_parametrization/parametrization_v3/demo_blade.py --no-show --blade-type axial
.\.venv\Scripts\python.exe development/new_parametrization/parametrization_v3/demo_blade.py --no-show --blade-type radial
```

PNG/SVG figures and reference-blade CSV files go to `output/`, or the path supplied with `--output`. Physical CSV coordinates are in metres. Axial coordinates are axial position and unwrapped circumferential distance; radial coordinates are Cartesian positions in the radial plane. Station CSVs also include meridional distance, transformed distance, radius, azimuth, arc length, metal angle, normalized full thickness, both side-offset ratios, and the transfer window. The comparison demo also exports a curvature CSV. Both scripts accept `--config` for another YAML path.

## Files and conventions

- [curvature_method_review.md](curvature_method_review.md): review of the three supplied Korakianitis papers, v3 curvature diagnostics, and recommendations for a meanline-driven parametrization suitable for later optimization.
- [blade_geometry.py](blade_geometry.py): ordinary NumPy functions with descriptive names and NumPy-style docstrings. `build_blade` takes an integer blade count; no solidity or loading rule is hidden inside the geometry module.
- [demo_blade.py](demo_blade.py): the solidity-to-count calculation, three views, sliders, and export commands.
- [demo_asymmetry.py](demo_asymmetry.py): before/after shape and curvature comparison.
- [theory_notes.md](theory_notes.md): conventions, reference camberline, Denton thickness, new redistribution equations and their provenance, physical offsets, and blade spacing.

The polynomial transfer is a modelling choice developed here, not a Denton prescription. The local thickness offset is exact on the unwrapped axial surface and only first-order accurate when radius varies. The nominal Denton maximum need not be the actual maximum. Invalid mathematical inputs leave the last valid plot visible with an explanation; geometrically unattractive or intersecting blades remain visible. Existing TODOs for literature comparisons are preserved.
