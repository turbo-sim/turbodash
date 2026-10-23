# Blade geometry

See the [illustrated blade profile controls](blade_geometry_controls.md) for
one-at-a-time reaction/impulse geometry sweeps and reproducible figures.

## Production modules

`turbodash.geometry.blade_parametrization` owns the conformal camberline,
rational Denton thickness, edge rounding/relief, thickness correction, sampled
diagnostics, plots, and the TurboMoC CAD adapter. There is one geometry kernel.
Development modeller and sensitivity studies import this package. Its
`ParameterState` stores an inherited source and sparse, independently editable
overrides. `turbodash.geometry.meanline` constructs per-row baselines and
projects effective geometry into downstream result copies. Compatibility
coordinate entry points in `turbodash.geometry.blade` also use this kernel.
The former JAX laws are archived in
`turbodash/geometry/legacy_parametrization/blade.py` and are not used by the active geometry workflow.

## Meanline mapping

Meanline metal angles are signed degrees relative to the downstream meridional
direction: absolute `alpha` for stators and relative `beta` for rotors. They
map directly to the conformal inlet/outlet metal angles. Dimensions remain in
metres. The inherited row count uses the existing integer realization policy,
`ceil(blade_count_continuous)`, respecting an explicit realized integer count.
Meanline performance and its continuous pitch/count remain unchanged.

| Meanline quantity | Geometry input |
| --- | --- |
| Arithmetic average inlet/outlet radius | Axial mean radius |
| Mean blade height | Blade height / radial axial extrusion height |
| `chord_meridional` | Axial camberline endpoint separation |
| `radius_in`, `radius_out` | Radial camberline endpoint radii |
| `metal_angle_in`, `metal_angle_out` | Signed inlet/outlet camberline angles |
| `maximum_thickness / chord_meridional` | Nominal full maximum-thickness ratio |
| `maximum_thickness_location`, when supplied | Nominal conformal thickness location |
| `trailing_edge_thickness / chord_meridional` | Unrounded trailing shoulder thickness ratio |
| Leading-edge radius | Leading unrounded ratio derived as described below |

The meanline parameter named `maximum_thickness_to_chord` is implemented in the
solver against **meridional** chord, even though true chord is separately used
for solidity. The mapping preserves that definition. Nominal thickness and
location are body controls, not guaranteed locations or values of the actual
maximum after rounding, relief and mapping.

For an axial cylinder, the LE unrounded ratio is
`sqrt(8 R_LE delta_LE / (c_m k_LE cos(beta_in)))`, where `delta_LE` is the
rounding fraction and `k_LE` its exponent. The regular rounded-tip limit thus
retains the meanline radius; the input is not incorrectly treated as a finite
LE thickness. Radial initialization substitutes the local conformal scale
`r_in abs(log(r_out/r_in))` for `c_m`. It is a local-scale target there, not an
exact radius guarantee after nonlinear mapping. Radial thickness correction is
initially off; users may enable it explicitly.

The new closed rounded-tip family does not reproduce the old finite-width
trailing cap. Its inherited TE quantity supplies the **unrounded shoulder
scale**, and the finished rounded tip closes to zero width. A row warning
explains this distinction. Set the trailing rounding fraction to zero for an
open shoulder with a straight closing cap; this is not an automatic attempt to
fit the old cap. Legacy wedge angles remain compatibility metadata and do not
control the new geometry.

## Classification and presets

Physical blade classification uses `z = -beta_in / beta_out` with the same
signed row-frame angles used by the losses. The loss-model helper
`_blade_loading.angle_ratio` agrees in its ordinary domain but continues the
outlet coordinate below 10 degrees to stabilize optimizer trial points. That
continuation is deliberately not used to classify an accepted physical blade.
The preset interpolation weight is `clip(z, 0, 1)`. Nonfinite ratios, undefined
angles or `abs(beta_out) <= 1e-8` degrees select the reaction endpoint with an
explicit diagnostic. Nonfinite metal angles themselves remain invalid inputs.

| Shape control | Reaction (weight 0) | Impulse (weight 1) |
| --- | ---: | ---: |
| Turning concentration | 1.2 | 1.8753 |
| Half-turning location | 0.52 | 0.5 |
| Nominal maximum-thickness location if absent from meanline | 0.3 | 0.5 |
| LE fullness | 0 | -0.25 |
| TE fullness | 0 | -0.25 |
| LE rounding fraction | 0.1 | 0.005 |
| TE rounding fraction | 0.025 | 0.005 |
| LE and TE rounding exponents | 3 | 3 |
| Body exponent | 2 | 2 |
| LE pressure-side relief | 0.15 | 0 |
| TE pressure-side relief | 0 | 0 |

The endpoints follow the thickness sensitivity study: the reaction profile with
zero fullness (original Denton), and the impulse profile with -0.25 fullness at
both shoulders. Turning, thickness location, rounding and body exponent use the
same study presets. Reaction leading relief is 0.15; impulse leading relief and
all trailing relief are zero. For a zero-turning meanline row, leading relief is
disabled because an automatic pressure side is undefined. These are geometric
shape defaults, not aerodynamic requirements. The study's absolute dimensions
and thickness/edge ratios are not copied over meanline values, so a calculated
row need not reproduce the example outline exactly.
Every numeric shape value is interpolated linearly, then validated, including
the rational fullness admissibility condition. LE and TE remain independent;
the app controls span `[-1, 1]`. Valid profiles require values strictly between
the endpoints, which are degenerate limits of the thickness law. Some strongly
opposed pairs within those individual limits are also rejected because their
shared rational denominator would become singular.

Meanline-provided dimensions, angles and thickness values take precedence.
In particular an explicitly supplied thickness location is retained, even at
the impulse endpoint. Symmetry is possible with symmetric inherited inputs;
the application never silently forces unequal meanline edge dimensions to be
equal or links their edits.

## Supported surfaces and diagnostics

Axial blades are sections on a cylinder with the same developed geometry and
angles across span. Radial blades are mapped in the radial plane, inward or
outward, and extruded between parallel endwalls. Meanline flaring and
free-vortex span twist are not represented; a warning states when station
geometry has been averaged. TurboGrid and the turbine assembly/mesh workflow
remain axial-only. The row modeller supports radial geometry and preview.

The new TurboMoC adapter transfers sampled sides with shared rounded-tip
tangents, checks spline-to-polyline deviation, and retains edge protrusions.
The nominal camberline chord is not replaced by the complete profile envelope.
Shape diagnostics are geometric checks at the requested resolution, not proof
of aerodynamic feasibility. Base sampling is 601 stations plus additional edge
stations; Plotly line simplification is disabled for blade contours.

## Workbench workflow and state

Launch the optimization workbench from the repository root:

```powershell
poetry run app_turbine_optimization --open-browser
```

The experimental edition (`app_turbine_experimental`) uses the same modeller.
The older meanline and stage views also construct their profiles with the
production kernel through its coordinate adapters.

1. Calculate a valid axial or radial design in **Meanline**. Each successful
   calculation creates fresh stator and rotor baselines for every stage.
2. Open **Geometry**, choose the stage and row, and expand a numbered section:
   **01 Meridional channel** (including blade count), **02 Camberline and turning**,
   **03 Thickness distribution**, or **04 Flow domain**. The Geometry inspector
   replaces the Meanline inputs while retaining their values. Search stays within
   the current workspace. Base sampling points live under **Flow domain → Advanced:
   resolution**. Slider endpoints show the current minimum and maximum; turning
   concentration initially spans 1–5, expanding for valid typed values outside it.
   Dimensions are displayed in millimetres, angles in degrees, and thickness
   controls are full-thickness ratios. Number boxes accept arbitrary floating
   point values; slider increments do not round the stored input.
3. Adjust turning, independent LE/TE fullness, thickness, rounding, leading-edge relief or
   physical-thickness correction. **Inherited** and **Edited** distinguish
   baseline values from overrides. Reset a field or use **Restore preset defaults**
   to clear the selected row's overrides while retaining its meanline dimensions
   and angles. A stage dropdown and Stator/Rotor toggle stay visible above the
   numbered sections.
   Thickness has separate menus for body/shoulders, leading edge, trailing-edge
   rounding, and physical thickness correction. Leading relief uses the automatic
   pressure-side convention; trailing relief is disabled. Row YAML imports with
   explicit leading-side selection or nonzero trailing relief are rejected with
   an explanation. The standalone kernel still supports these advanced settings.
   The information buttons show the parameter guide's tooltips on hover or focus.
4. Inspect the blade, camberline and thickness plots and 3D row preview. The
   preview uses the shared TurboMoC VTK renderer. **Configure display** controls
   row visibility, displayed blade copies, surface visibility and colors. These
   are display preferences; the physical blade count stays in the geometry
   parameters. **Fit view** frames the geometry. Optional **Passage crosscuts**
   measure the current axial passage; they do not change the blade shape.
   Radial physical passage crosscuts are not implemented.
5. **Row YAML** opens a dialog with the selected row's source and sparse overrides.
   Copy or download the text, paste a replacement, or choose a YAML file.
   Uploading a file fills the editor without changing the blade. **Apply to row**
   validates the draft and applies its effective values as overrides of the current
   row, preserving the current meanline source for reset. Errors stay visible in
   the dialog and leave the last valid geometry intact. Close discards unapplied
   text; reopening shows the current row. In the optimization edition, the main
   header's **Save design** saves meanline inputs, optimization settings and all
   stages' stator and rotor geometry together. **Load YAML**, or supplying the file
   as the app's startup case, restores each row's saved source and overrides after
   the matching meanline calculation. **Row YAML** is useful for transferring a
   single row. In the experimental edition, use **Row YAML** to save geometry
   separately from the header's meanline design file.
6. Download the current axial geometry as independent TurboGrid rows or a
   positioned TurboGrid assembly. This ZIP export does not require optional
   mesh dependencies. The Geometry inspector has no separate 3D-model ZIP button.
   With `poetry install -E mesh`, generate its mesh in **Mesh**. Geometry edits
   invalidate an existing mesh. The mesh
   extra also supplies `dash_vtk` and the TurboMoC browser renderer required for
   the 3D preview. Without it, the viewer explains how to install the extra;
   the geometry editor and 2D plots remain available. Both axial and radial
   previews use sampled geometry without building CAD. The application's
   turbine CAD/mesh export remains axial-only.

Geometry editing does not change the thermodynamic solution or rerun meanline.
Editor responses are accepted only for the current calculation and latest edit
revision, so delayed responses cannot replace a newer row edit or YAML import.
The next **successful** meanline calculation replaces all row baselines and
clears their overrides, including when its resulting geometry is unchanged.
Failed calculations preserve the last successful geometry. Tabs, themes,
view options and row selection preserve edits. Optimizer trial evaluations and
progress snapshots do not initialize geometry; only the accepted result does.

The optimization design YAML stores geometry under
`geometry: {schema_version: 1, stages: [...]}`. Each stage contains `stator` and
`rotor` parameter-state mappings with their own `source` and `overrides`. Saved
inputs include the physical integer blade count, dimensions, camberline and
thickness parametrization, edge rounding and relief, physical-thickness
correction, flow-domain extensions and integer sampling-point count. Continuous
values retain their full precision, and loading preserves which fields were
inherited or edited. Files without a `geometry` section retain the usual meanline
presets. Geometry rows must match the design's stage count and axial/radial type;
invalid geometry is rejected before the uploaded design is applied.

`apps/shared/components/blade_modeller.py` owns the editor and its state reducer.
`geometry_state` contains the live baseline-plus-overrides state for all rows.
`geometry_committed` is the last committed snapshot of that same state, used
for 3D and exports; it is not a separately editable model. The shared workflow
projects it into `geometry_results_store` for existing CAD/mesh consumers.
`result_store` remains the unchanged meanline solution. The meanline blade plot
reads live effective geometry when visible and refreshes when opened; geometry
edits are not inputs to its solver callback. Invalid parameter edits retain the
previous valid state and explain the error. Self-intersection and neighboring
blade checks run on committed geometry. A contour that fails those checks can
still be inspected in the preview with its diagnostic; TurboMoC CAD/mesh
generation rejects it rather than silently changing its shape. TurboGrid curve
exports preserve the sampled contour, so review its diagnostics before meshing.

Slider movement is sampled at 100 ms intervals for lightweight 2D updates;
release or a committed numeric edit refreshes 3D. Sections are cached by their
effective parameters. Preview responses carry the accepted-calculation epoch,
row and edit revision so older results cannot replace a newer view.
`visualization/blade_preview.py` converts the kernel's physical coordinates to
the shared VTK surface payload in millimetres, preserving the sampled blade
contour and its edge protrusions. It supplies the blade and six passage boundary
surfaces without building CAD. The preview's VTK host remains mounted while
geometry data changes; TurboMoC owns rendering and camera interaction. Display
preferences are separate from geometry edits and use the same controls as the
application's other VTK views.
Pending edits travel together in a client transaction journal, so rapidly
editing two fields does not depend on the first response arriving. Acknowledged
edits are compacted into the last valid snapshot to bound replay work during
sustained dragging. Invalid edits are acknowledged without changing geometry.
Saving row YAML or an optimization design includes a pending numeric edit.
CAD/TurboGrid exports, mesh
generation and mesh downloads wait until pending edits have reached the
committed downstream geometry; an older mesh cannot be downloaded as though it
represented the newly edited row.

In the integrated workbench, blade sampling and inlet/outlet domain extensions
belong to the shared geometry state. The mesh pane controls mesh resolution and
inflation without maintaining another editable copy of those geometry inputs.

## Validation

Focused production tests are `tests/test_blade_parametrization_kernel.py`,
`tests/test_blade_parametrization_review.py`, `tests/test_blade_modeller_ui.py`,
`tests/test_blade_preview.py`, `tests/test_visualization_vtk.py`,
`tests/test_geometry_workflow.py`, and `tests/test_geometry_pending_exports.py`.
Meanline, plotting, mesh and export tests also exercise the production adapter.
The two sensitivity scripts under `development/blade_parametrization` rebuild
their 30 camberline and 10 thickness cases using the production kernel.

For browser regression, run the workbench on port 8067, then:

```powershell
poetry run python tests/browser_blade_modeller.py --url http://127.0.0.1:8067
```

The check requires Playwright and Chrome. It writes its report and screenshots
under `development/blade_parametrization/output_integration`. Historical
development test counts are not a substitute for these integration checks.
