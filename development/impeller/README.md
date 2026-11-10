# Centrifugal impeller geometry

The implementation is in [turbodash.geometry.impeller](../../turbodash/geometry/impeller/). It builds curved hub/shroud channels, independent blade-section laws and a closed, rounded blade surface from the [approved brief](impeller-geometry-answers.json), now with explicit exit-rake control. Examples, copied meanline output and studies live here; MastersThesis can import the same package.

Start with the [zero-rake interactive geometry](output/reversible_impeller/geometry.html), [overview figure](output/reversible_impeller/overview.png) and [rake comparison](output/rake_study/rake_comparison.png). The [independent-section study](output/section_study/comparison.png) retains the original unconstrained example. Outputs are generated locally and ignored by Git; the commands below reproduce them.

## Run

From the Turbodash repository root in the existing environment:

```powershell
.venv/Scripts/python.exe development/impeller/generate_impeller.py
.venv/Scripts/python.exe development/impeller/compare_rake.py --export
.venv/Scripts/python.exe development/impeller/compare_sections.py --export
```

Edit [examples/reversible_impeller_zero_rake.yaml](examples/reversible_impeller_zero_rake.yaml), then rerun. This default retains gradual turning and allows leading-edge lean while imposing zero exit rake. The original [free-rake input](examples/reversible_impeller.yaml) and the earlier [preserved-LE zero-rake input](examples/reversible_impeller_zero_rake_preserve_le.yaml) are preserved. To keep several cases:

```powershell
.venv/Scripts/python.exe development/impeller/generate_impeller.py --config path/to/case.yaml --output development/impeller/output/my_case
```

The generator writes an offline Three.js HTML report with Turbodash's orientation view cube, a simple disk, row/single-blade selection and shroud visibility controls. The report also includes meridional, metal-angle, thickness and edge-stacking plots. Other outputs include PNG/SVG overview, closed blade, full-row and separate disk STL, float64 NPZ arrays, section CSV, resolved YAML, diagnostics and a units/fingerprint manifest. STL coordinates are **metres**; confirm this when importing into a viewer that assumes millimetres. Matplotlib is imported only for visualization. The geometry kernel does not import nurbspy or a meshing runtime.

HTML generation reuses the pinned Three.js browser assets from the companion nurbspy checkout and the original orientation cube from TurboMoC, together with Turbodash's shared cube styling. It reads these files without importing either geometry/solver package. The existing sibling checkouts are discovered automatically; elsewhere, set `NURBSPY_WEB_ASSETS` to `nurbspy/src/nurbspy/web_assets` and `TURBOMOC_SOURCE` to the `method_of_characteristics` checkout. The generated HTML embeds all dependencies and needs neither these repositories nor internet access when opened. Use `--no-html` for geometry exports without the browser assets.

## Python API

```python
from turbodash.geometry.impeller import build_impeller, export_geometry, load_config

config = load_config("development/impeller/examples/reversible_impeller_zero_rake.yaml")
config["rake"]["angle_deg"] = 10.0
config["sections"][-1]["camberline"]["outlet_metal_angle_deg"] = -35.0
config["sections"][-1]["thickness"]["maximum_thickness_m"] = 0.005
geometry = build_impeller(config)
middle = geometry.section(0.5)
manifest = export_geometry(geometry, "development/impeller/output/edited")
```

`geometry.plus`, `.minus` and `.camber` have shape `(span, streamwise, 3)` in XYZ metres. `geometry.u` is normalized conformal distance; `.v` runs from hub to shroud. Each section records its corresponding meridional coordinate, angles, target/measured thickness and conformal charts. `geometry.mapping.point(u,v)` uses the **footprint** streamwise coordinate; see the [theory note](impeller_geometry_theory.md) for the conversion.

Treat evaluated results as snapshots. Edit a configuration and call `build_impeller` again before evaluating sections or exporting; those entry points reject a changed configuration paired with an older result.

For use from MastersThesis, install the local Turbodash checkout into the selected thesis environment with `python -m pip install -e ../turbodash` from the MastersThesis root, then use the same imports and an explicit configuration path. No thesis source code was changed for this implementation.

## Implemented controls

| Input | Meaning |
| --- | --- |
| `dimensions` | Inlet radii, outlet radius/width, axial outlet-centre position and station inclinations |
| `hub`, `shroud` | Independent inlet/outlet wall slopes, tangent reaches and bow |
| `control_point_displacements` | Existing sparse, baseline-relative interior wall refinements |
| `blade_edges` | Independent hub/shroud LE/TE positions and edge bow |
| `sections` | Ordered controls at `v=0`, `v=1`, and optional interior spans |
| `camberline` per section | Inlet/outlet metal angles, turning location/concentration, angle or slope interpolation |
| `thickness` per section | Dimensional nominal maximum and shoulder thicknesses, location, fullness and rounded-end controls |
| `stacking_offset_deg` per section | Circumferential LE position in `free`/`preserve_le`; TE stacking replaces the resulting LE position |
| `rake` | Explicit exit angle and choice of preserving LE stacking or the original angle laws |
| `inlet_interpolation` | Default `tangent_radius` preserves the baseline meanline inlet law; `angle_span` is explicit linear angle interpolation |
| `numerics` | Output stream/span sampling and integration-table resolution |

Thickness uses an explicit local-metric offset and reports the finite paired 3D distance afterwards. It performs no iterative correction. The existing maintained turning/thickness functions are imported; the meridional prototype has been promoted with backward-compatible development imports. The core has been exercised with all nurbspy imports blocked.

## Explicit exit-rake control

The default example uses:

```yaml
rake:
  mode: trailing_edge
  angle_deg: 0.0
  reference_azimuth_deg: null
```

Positive rake points toward increasing azimuth while following the meridional trailing edge from hub to shroud. At the usual radial outlet, zero rake means an axial trailing edge. For a swept or bowed meridional edge, it means a constant-azimuth curve in one meridional plane. Rake is separate from outlet metal angle/backsweep.

| Mode | Preserved quantities | Dependent quantities |
| --- | --- | --- |
| `preserve_le` | LE azimuths, inlet/outlet metal angles, monotonicity, thickness inputs | Interior angle distribution adjusts explicitly to satisfy the requested rake |
| `trailing_edge` | Entire original angle and thickness laws | Each section rotates; LE alignment changes |
| `free` | Original LE stacking and turning law; default when the group is absent | Rake remains an outcome, as in the first implementation |

`reference_azimuth_deg` specifies the hub trailing-edge azimuth. With `null`, `preserve_le` selects the midpoint of the sampled common feasible interval. `trailing_edge` instead retains the original hub trailing-edge azimuth. A target is evaluated at every span, including `geometry.section(v)` between output stations.

The `preserve_le` mode uses an integral-constrained monotone slope law with an analytic integral. It requires no root solve. `turning_concentration` controls its sharpness; the original `turning_location` and `interpolation` controls are inactive in this mode and are listed as such in diagnostics. Infeasible combinations of rake, LE placement and endpoint angles raise an error instead of introducing angle overshoot. The other two modes retain the original scalar turning law.

The default keeps the original gradual angle distributions and selects the original hub TE azimuth, approximately **-54.902593 degrees**, for every section. The sampled rake error is below **3e-12 degrees**. The resulting shroud LE azimuth is approximately **-21.253940 degrees** relative to the hub LE. This is an azimuth displacement, not a single prescribed lean angle; the local lean follows the curved edge and may vary along the span. Both the turning-location and interpolation controls remain active. No nonmonotonic angle law is introduced.

The archived `preserve_le` example selects a common TE azimuth of approximately **-50.628061 degrees** in the interval **[-52.029825, -49.226298] degrees**. Its aligned leading edges and monotonic endpoint-constrained angles force early hub turning and late shroud turning. It remains available for comparison, but is no longer the default. The [comparison](output/rake_study/rake_comparison.png) and [theory](impeller_geometry_theory.md#8-explicit-exit-rake-and-circumferential-stacking) explain the tradeoff.

## Illustrative disk and report controls

The disk follows the full hub contour, including the rounded blade-root extent. The current example extends it upstream with a **35 mm rounded nose** and has a **solid centre with no shaft bore**. The nose joins the existing hub tangentially and closes smoothly on the axis; it does not change the blade or the existing channel walls. The back face is flat. The meanline did not supply these mechanical dimensions.

The optional `disk` configuration group stores `nose_length_m`, `bore_radius_m` and `back_thickness_m`, all in metres. A zero bore radius means a solid centre; a zero nose length gives a flat front. Positive nose length requires a solid centre. Legacy files without this group retain the earlier automatic bore and back thickness. `build_disk` in [disk.py](../../turbodash/geometry/impeller/disk.py) also accepts explicit keyword overrides.

The standard geometry export keeps the disk and blades as separate closed surfaces. The optional printable export below fuses them into one solid for slicing; mechanical cover thickness and structural fillets remain undefined. The Three.js view supports disk visibility/transparency, shroud visibility and an orientation cube for inspecting the complete shape and the leaned leading edge.

For example, pass `disk_options={"nose_length_m": 0.04, "bore_radius_m": 0.0, "back_thickness_m": 0.03}` to `export_geometry` to override dimensions for that export. Prefer storing settings in the configuration for reproducibility across the app, HTML and printable STL. The configuration fingerprint includes the disk group when present; the manifest also records the resolved disk dimensions and topology.

## Single-file STL for 3D printing

The combined export is [impeller_printable_mm.stl](output/reversible_impeller/impeller_printable_mm.stl), with a neighbouring [validation record](output/reversible_impeller/impeller_printable_mm.json). It contains the rounded nose, solid disk and all 20 blades as **one connected, watertight solid**, with no shaft bore and the shroud side open. The original `blade.stl`, `row.stl` and `disk.stl` remain separate engineering exports in metres.

The printable file uses **millimetre coordinates at full scale**: approximately **637.10 × 637.10 × 188.36 mm**, including the nose. Select millimetres in the slicer; STL does not standardize units. Choose any reduction for your printer explicitly, and inspect thin leading/trailing edges and supports at that scale. This export prepares a geometric model for slicing; it does not qualify a printed part as an operating rotor.

Install the optional dependency once, then regenerate from the repository root:

```powershell
.venv/Scripts/python.exe -m pip install "manifold3d>=3.5.4,<4"
.venv/Scripts/python.exe development/impeller/generate_impeller.py --printable --no-html
```

The package installation extra is `turbodash-core[print]`. The reusable API also accepts an edited configuration from the app:

```python
from turbodash.geometry.impeller import build_impeller, load_config, export_printable_stl

geometry = build_impeller(load_config("my_design.yaml"))
record = export_printable_stl(geometry, "impeller_printable_mm.stl")
```

The implementation reuses the existing blade and disk tessellators and uses [Manifold](https://github.com/elalish/manifold) for the solid union. Each root extends 0.5 mm inward into the disk to give finite overlap across independently triangulated hub surfaces. Original blade vertices above the hub are retained before union. The printable disk uses 320 angular samples. Intersection slivers are simplified at 0.001 mm tolerance, snapped to a 0.0001 mm coordinate grid, and simplified again; these settings are recorded in the JSON file. No voxel reconstruction is used.

The written float32 STL is read back and exactly welded before validating one connected component, edge closure, consistent winding, no degenerate triangles, positive volume and the topology matching the requested solid or bored centre. The default export has **519,542 triangles** and passes these checks. Configurations that fail validation raise an error without replacing a previous successful STL. Override `root_overlap_m` or `disk_options` in the API when needed; insufficient overlap relative to disk faceting and excessive overlap relative to disk thickness are rejected.

## What the baseline represents

The copied inputs come from an actual single evaluation of `MastersThesis/meanline/examples/example_reversible.yaml` at its starting design variables. No matching saved result existed, so [export_meanline_example.py](export_meanline_example.py) generated [source_meanline_result.json](examples/source_meanline_result.json). The [source record](examples/README.md) includes commit, file hashes, evaluation status and regeneration instructions.

This is a starting example, **not an optimized or adopted thesis design**. In particular, its meanline constraint targets are not all satisfied. The original design is open and has nonzero clearance. The approved geometry overrides this to a shrouded, zero-gap impeller; its performance has not been recomputed for that change. The maximum thickness, wall shaping and chordwise turning controls are recorded geometric assumptions. The meanline edge thicknesses seed unrounded shoulders; the requested rounded physical tips close to zero thickness.

The source convention places the outlet hub at `L_z`; the channel stores its outlet centre, so the adapter uses `axial_length = L_z - b_2/2`. The baseline preserves the inlet angle of -47.28 degrees at the actual RMS radius, which is not geometric midspan.

## Verification and limits

The default zero-rake example has 20 blades and 17 sampled span sections. Maximum measured paired-thickness error is approximately **0.00843%** (maximum absolute error **0.483 micrometres**), unchanged by the section rotations. The blade export has **16,704 triangles**, shared rounded seams, consistent orientation and positive enclosed volume. The earlier free-rake example's integration/output refinement changed common-point coordinates by at most about **1.46e-10 m**. That historical result does not bound interpolation or triangulation error between points. The archived preserved-LE variant gave **0.00918%** thickness error.

The geometry package checks channel orientation, scalar-law domains, sampled section self-intersections and adjacent-blade intersections. Export checks cover topology, degeneracy, orientation, periodic correspondence and round-trip data. The [next-steps document](next_steps.md) records stronger continuous/3D validation and adaptive sampling as remaining work.

The disk/viewer update passed **199 tests**: 123 impeller, disk and browser-asset tests plus 76 maintained blade-law regression tests. The tests include independent quadrature, Cartesian derivative measurements, signed rake, arbitrary spans, monotonicity, infeasibility, disk topology and known solid volume. The default-example regression verifies that the complete angle and thickness laws are retained and that zero rake holds between sampled spans while LE lean results from section wrap. To run the checks:

```powershell
.venv/Scripts/python.exe -m pytest tests/test_impeller_geometry.py tests/test_impeller_mapping.py tests/test_impeller_meanline.py tests/test_impeller_convergence.py tests/test_impeller_output.py tests/test_impeller_turning.py tests/test_impeller_rake.py tests/test_impeller_disk.py tests/test_impeller_assets.py tests/test_blade_parametrization_kernel.py tests/test_blade_parametrization_review.py -q
```

The generated report was also opened directly from disk in an isolated headless Edge browser. Checks covered offline loading, row/single selection, visibility and opacity, cube alignment, mouse rotation, reset, fullscreen and mobile resizing, with no JavaScript errors or external requests. Reproduce this optional check with `.venv/Scripts/python.exe development/impeller/check_report.py` (requires Playwright and a Chromium-family browser). Screenshots and its validation record are written under `output/reversible_impeller/viewer_checks/`.

The exported passage includes named inlet, outlet, periodic and endwall patches, but endwalls remain **untrimmed at the blade**. It is not yet a watertight fluid volume. The blade itself and the illustrative disk are individually closed. Mechanical disk/cover design, fillets, splitters, vaned diffuser, STEP/CAD solids, central turbine-workbench integration and CFD meshing remain outside this delivery. The standalone impeller app is implemented below.

## Documentation

- [Theory, equations and stacking alternatives](impeller_geometry_theory.md)
- [Implementation record and original repository assessment](impeller_implementation_plan.md)
- [Prioritized next steps](next_steps.md)
- [Approved questionnaire answers](impeller-geometry-answers.json)
- [Meanline snapshot and provenance](examples/README.md)
- [Previous meridional explorer](../mixed_flow/README.md)

## Interactive app

The reusable Dash app is in [turbodash.apps.app_impeller](../../turbodash/apps/app_impeller/). Launch it from the repository root:

```powershell
.venv/Scripts/python.exe development/impeller/app.py --open-browser
```

The equivalent package command is `.venv/Scripts/python.exe -m turbodash.apps.app_impeller`. The default address is [http://127.0.0.1:8052](http://127.0.0.1:8052). Use `--port` for another port and `--config path/to/case.yaml` for another starting configuration; omit `--open-browser` to start only the local server.

The app follows Turbodash's workspace layout. A scrollable engineering sidebar provides sliders and precise numerical inputs for the independent hub/shroud angle and thickness laws, channel dimensions, wall engineering controls, blade footprint, blade count and rake. Slider release or a committed numerical edit updates the geometry. The angle diagram shows hub and shroud together, as does the thickness diagram; these appear at the top, followed by meridional and edge-stacking diagrams, the draggable wall editor, and the Three.js assembly view at the bottom.

Select the hub or shroud in the wall editor, then drag the editable interior points. The native editor supplies point picking, local preview, coordinate edits and undo/redo. **Apply wall edits** commits the refinements to the shared geometry configuration; **Clear wall refinements** restores the current engineering wall shapes. The inlet/outlet station points and tangent control points remain fixed. The editor's x/y display coordinates correspond to axial z/radius r in millimetres. This interface reuses nurbspy's native editor; its local source checkout is discovered automatically, or can be selected through `TURBODASH_NURBSPY`. The impeller geometry kernel remains independent of that editor dependency.

The current default preserves gradual turning, sets zero trailing-edge rake and allows the leading-edge lean to result from the section wraps. Controls that do not affect the selected rake mode are disabled. Invalid draft inputs retain the last accepted plots and 3D geometry with a visible error; saving YAML is disabled while the draft is invalid or awaiting evaluation. **Load YAML**, **Save YAML** and **Reset example** use the same configuration schema as the script workflow. The 3D camera and display settings remain in place as accepted geometry changes.

The **Hub nose and disk** controls set the nose extension, back thickness and optional bore. The bore input is disabled while a rounded nose is present. The meridional diagram shows the nose, and the Three.js preview uses the same disk configuration. Click **Download full STL** below the 3D view to build, fuse, validate and download the current complete impeller in millimetres. The filename includes its configuration fingerprint. Export shows a busy message; if the design changes before completion, the old result is discarded and the app asks you to export again. Invalid drafts cannot be exported. Unapplied wall drags remain previews: use **Apply wall edits** before exporting those changes.

The rounded-nose/download update passed **205 impeller tests** and seven browser checks, including an actual downloaded STL, saved disk settings, export cancellation after design edits, invalid-input recovery and mobile layout. The browser check and screenshots are in `output/app_stl_checks/`; reproduce with `.venv/Scripts/python.exe development/impeller/check_stl_download.py` while the app is running.

The integrated geometry/app/shared-wall regression run passed **358 tests**. Browser checks exercise rapid independent edits under delayed responses, exact typed precision, invalid-input recovery, YAML round trips and persistent 3D camera/display state. Native-editor checks additionally exercise actual pointer dragging, applying and clearing refinements, resetting unapplied edits, unchanged-precision Apply, and rejection of locked station edits. The validation records and screenshots are in `output/app/` and `output/app_wall_checks/`.

With the app running, reproduce the optional browser checks using Playwright and an installed Edge browser:

```powershell
.venv/Scripts/python.exe development/impeller/check_app.py
.venv/Scripts/python.exe tests/browser_impeller_walls.py --url http://127.0.0.1:8052
```
