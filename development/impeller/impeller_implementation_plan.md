# Centrifugal impeller parametrization: implementation record and remaining plan

Updated 2026-11-10. The geometry implementation includes explicit exit-rake control in `turbodash.geometry.impeller`, with the reversible example, executable studies and documentation in `development/impeller`. The current default imposes zero rake by TE stacking, retains gradual monotonic angle laws and lets leading-edge lean result from the section wraps. A separate illustrative disk and Three.js report with the existing orientation cube support shape review. This section records the implemented scope and approved choices; the original repository assessment is retained below as a clearly historical study. Meshing remains deferred.

Use the [workflow README](README.md) to run and inspect the example, the [theory note](impeller_geometry_theory.md) for the implemented equations and stacking discussion, and [next_steps.md](next_steps.md) for the outstanding work. The [reviewed answers](impeller-geometry-answers.json) are the decision record; the [questionnaire](impeller_geometry_questionnaire.html) remains a record of those choices, not an unresolved implementation gate.

## Approved decisions and implemented scope

| Topic | Current decision and implementation |
| --- | --- |
| Location | Reusable code lives in [turbodash/geometry/impeller/](../../turbodash/geometry/impeller/). Examples, copied meanline outputs, configuration, studies and documentation live here in `development/impeller`. MastersThesis can import the package; its checkout was used read-only. |
| Phase | Geometry first, with executable scripts, exports and a standalone interactive impeller app. CFD volume meshing remains deferred. |
| Machine | One physical reversible impeller, initialized from an actual evaluation of the committed `MastersThesis/meanline/examples/example_reversible.yaml` starting design. It is not an optimized or adopted thesis design. |
| Shroud and blades | Shrouded main-blade impeller, zero modeled tip gap, 20 full blades in the seed and no splitters. The original open meanline machine and its nonzero clearances are preserved in the source snapshot; the shrouded override is recorded explicitly. |
| Channel and footprint | Curved independent hub/shroud walls retain all existing engineering and interior control-point displacement controls. Blade LE/TE endpoints and quadratic edge bow remain independent of wall shape. |
| Spanwise geometry | Independent hub/shroud angle and dimensional thickness controls, with optional intermediate sections. Scalar laws are interpolated and evaluated on the curved stream surfaces; the skins are not a straight XYZ loft between hub and shroud. |
| Angle and stacking | The default `rake.mode: trailing_edge` rotates the original gradual angle laws to zero exit rake; LE lean is dependent and may vary along the span. The earlier `preserve_le` mode remains available to maintain leading-edge positions and endpoint angles with an analytically constrained monotonic law. Missing rake settings retain legacy `free` behavior. Tangent-versus-radius inlet interpolation preserves the seeded RMS condition. Nonmonotonic angle extensions are deferred. |
| Thickness and edges | Reuse the maintained Denton/fullness/rounding laws. Apply explicit symmetric local-metric offsets and report their finite paired 3D-distance residual. There is no iterative thickness correction. Both reference edges are rounded and close to zero paired thickness; finite source edge scales seed the unrounded body. A cutoff edge is outside the chosen scope. |
| Coordinates and dependencies | New parameter-space interfaces and equations use $(u,v)$. The core uses metres, `(z,r)` in the meridional plane, radians internally and a Z rotation axis. NumPy/SciPy suffice; nurbspy is not required for evaluation. |
| Delivered surfaces | Closed sampled blade boundary, repeated blade row, and named reference hub/shroud/inlet/outlet/periodic passage patches. These passage patches remain untrimmed at the blade and do not yet form a watertight fluid-domain boundary. |

The present deliverable covers the selected aerodynamic blade and channel construction plus a configurable disk following the hub. The current example has a 35 mm tangent-continuous rounded nose, solid centre and no shaft bore. The optional `disk` YAML group persists the nose, bore and back thickness. Standard exports keep separate disk/blade surfaces; the printable export Boolean-unions them into one validated millimetre STL, with a small buried root extension to establish overlap. Structural design, cover thickness, fillets and a fitted STEP solid remain outside this implementation. The distinction between a closed rotor surface and a watertight fluid passage remains explicit.

The app now provides **Download full STL** beside the 3D view and **Hub nose and disk** controls. It exports the current accepted configuration through the same reusable printing code; invalid drafts and results made stale by edits cannot download. The rounded-nose integration passed 205 impeller tests and seven browser checks, including a real STL download and verification of its one connected, bore-free surface. See the [theory note](impeller_geometry_theory.md#14-rounded-upstream-hub-nose-and-solid-centre) for the explicit nose construction and [workflow](README.md#single-file-stl-for-3d-printing) for regeneration.

## Code, examples and reproduction

The standalone workbench is now in [apps/app_impeller/](../../turbodash/apps/app_impeller/). Launch with `python development/impeller/app.py` (port 8052). It places combined hub/shroud angle and thickness diagrams at the top, keeps the engineering channel/stacking plots and draggable wall editor, and puts the persistent Three.js assembly at the bottom. Geometry edits use the same evaluator as the scripts. A revision guard rejects late results, invalid drafts retain the clearly marked last valid geometry, and configuration saving requires the current validated revision. The existing mixed-flow explorer and this app now share one native wall-editor adapter; no picking or curve-rendering engine was copied.

| Responsibility | Authoritative files |
| --- | --- |
| Walls, edge controls and channel correspondence | [meridional/](../../turbodash/geometry/impeller/meridional/), promoted from `development/mixed_flow/meridional`; the former modules are compatibility imports, not duplicate kernels. |
| Footprint derivatives, distances and inverse mapping | [mapping.py](../../turbodash/geometry/impeller/mapping.py): `StreamSurfaceMap`, cached fixed-span sections, full-channel continuation and endpoint-tangent extensions. |
| Validated portable inputs and geometry identity | [config.py](../../turbodash/geometry/impeller/config.py): `normalize_config`, YAML read/write, scalar-law adaptation and fingerprinting. |
| Independent sections and 3D skins | [geometry.py](../../turbodash/geometry/impeller/geometry.py): `build_impeller`, section evaluation, shared sampling, stacking, angle integration and thickness diagnostics. |
| Explicit rake and monotone constrained turning | [rake.py](../../turbodash/geometry/impeller/rake.py): physical trailing-edge target and compatible reference azimuth. The reusable [turning.py](../../turbodash/geometry/blade_parametrization/turning.py) scalar law provides monotone endpoint-preserving turning with an analytic slope integral. |
| Plain meanline dictionary adapter | [meanline.py](../../turbodash/geometry/impeller/meanline.py): `from_meanline_geometry`, including the outlet-centre conversion $L_z-b_2/2$. The geometry library does not import the thesis solver. |
| Blade/row closure, reference passage and exports | [output.py](../../turbodash/geometry/impeller/output.py): shared-edge surfaces, orientation/manifold checks, periodic replication, STL/CSV/NPZ/YAML/JSON export and offline HTML/Matplotlib views. |
| Illustrative impeller disk | [disk.py](../../turbodash/geometry/impeller/disk.py): a closed revolved hub-following profile, explicit bore/back-thickness assumptions and separate-component semantics. |
| Editable machine example and provenance | The default [reversible_impeller_zero_rake.yaml](examples/reversible_impeller_zero_rake.yaml) adds the explicit rake choice to the unchanged [original seed](examples/reversible_impeller.yaml). Evaluated [source_meanline_result.json](examples/source_meanline_result.json), [source_snapshot.yaml](examples/source_snapshot.yaml), and [source_provenance.json](examples/source_provenance.json) preserve the actual meanline source. [The example notes](examples/README.md) separate copied outputs and subsequent geometry assumptions. |
| Reproducible source evaluation | [export_meanline_example.py](export_meanline_example.py), which reads the thesis checkout with bytecode writes disabled and writes artifacts here. Overwriting an edited geometry seed is an explicit command-line choice. |
| Geometry and control studies | [generate_impeller.py](generate_impeller.py), [compare_rake.py](compare_rake.py), and [compare_sections.py](compare_sections.py), all importing the same production API. |

From the Turbodash repository root:

```powershell
.venv/Scripts/python.exe development/impeller/generate_impeller.py
.venv/Scripts/python.exe development/impeller/compare_rake.py --export
.venv/Scripts/python.exe development/impeller/compare_sections.py --export
```

The first command now uses the zero-rake `trailing_edge` example and writes overview figures, an offline Three.js report with a disk and orientation cube, closed blade and repeated-row STL, section CSV, sampled arrays, resolved YAML and diagnostics under `development/impeller/output/reversible_impeller`. The earlier default is preserved as [reversible_impeller_zero_rake_preserve_le.yaml](examples/reversible_impeller_zero_rake_preserve_le.yaml). The second command produces a [rake comparison](output/rake_study/rake_comparison.png) and four interactive cases: [original free rake](output/rake_study/free/geometry.html), [preserved-LE zero rake](output/rake_study/zero_rake/geometry.html), [preserved-LE positive rake](output/rake_study/positive_rake/geometry.html), and [TE stacking](output/rake_study/te_stacked/geometry.html). The third deliberately retains the original free-rake seed to isolate independent hub/middle/shroud turning and thickness, footprint sweep/bow and stacking edits under `output/section_study`. These are geometric demonstrations, not aerodynamic improvements.

The source artifacts have already been generated. Re-evaluating the thesis starting point is available through `export_meanline_example.py`; use `--output-dir` to preserve edited inputs, and consult the example README before intentionally using `--overwrite`. The source meanline returned finite results and success flags, but did not satisfy all target constraints. Its recorded performance does not validate the subsequently shrouded geometry.

## Reuse and the revised mapping decision

The initial assessment recommended directly extending TurboMoC's general mapper. Inspection during implementation showed why that is not an equivalent drop-in path for this geometry:

- The promoted wall kernel pairs hub/shroud points at each wall's **own normalized physical arclength**. TurboMoC's current `GeometryDefinition` evaluates its input B-splines at a common raw spline parameter. Passing exact exported wall splines would preserve the walls but change the interior stream surfaces and footprint derivatives.
- The interfaces differ in units, coordinate ordering and azimuth/chart convention: the impeller uses metres and `(z,r)`, while the existing TurboMoC geometry contract uses millimetres and `(r,z)`. Conversion alone cannot repair the different station correspondence or bowed blade edges.
- TurboMoC's current general mapping uses adaptive `quad` evaluations and a `solve_ivp` inverse. The approved construction instead calls for a clean once-through geometry path with cached distance tables and monotone interpolation, including explicit analytic endpoint extensions.

The implemented solution is therefore a small **channel-protocol/table adapter** in `mapping.py`, reusing the promoted wall/derivative kernel and the maintained angle/thickness scalar laws. It does not copy TurboMoC's geometry engine or create a second wall implementation. Each span section integrates a uniform full-channel grid once; physical and conformal distances are measured from the blade's LE, and the inverse uses monotone interpolation. Uniform quadrature grids avoid the numerical instability of nearly coincident inserted stations. Outside the footprint, evaluation follows the actual channel until its endpoints, then continues along its endpoint tangents; the extension integral and inverse use explicit `log1p`/`expm1` expressions and reject nonpositive radius.

This decision supersedes the earlier requirement to use TurboMoC's mapper directly. Future TurboMoC integration should consume this same authoritative surface/section contract, with tested coordinate conversion and serialization, instead of regenerating a different impeller from raw spline parameters or scaled template sections. Mesh topology and mesh deformation remain future TurboMoC work. Nurbspy can still support a concrete future fitting/editor operation, but no such dependency is needed by the current geometry.

## Explicit rake extension and current baseline results

The rake angle is measured locally from the hub-to-shroud meridional trailing-edge tangent toward positive azimuth. The target edge satisfies $\theta_{TE}(v)=\phi+\tan\gamma\,Q(v)$, with $Q(v)=\int_0^v\|E_v(v)\|/r_{TE}(v)\,dv$. Thus zero rake gives constant exit azimuth, independently of backsweep. The general definition also includes swept/bowed meridional trailing edges.

`preserve_le` computes a common admissible hub-TE reference from a fixed 257-span screen plus control spans, then uses a Beta-CDF slope law with an analytically specified mean. The selected reference and every requested section are validated; no nonlinear solve adjusts turning or thickness. The original `turning_location` and angle/slope interpolation selector are inactive in this mode, while `turning_concentration` controls the new family's sharpness. `trailing_edge` retains the entire original angle law and computes the necessary angular shift at every span. The theory note gives the equations and mode-specific tradeoffs.

The generated **zero-rake** artifacts report the following values at the default resolution. Values are geometry checks, not aerodynamic acceptance criteria.

| Quantity | Recorded baseline |
| --- | --- |
| Number of blades | 20; angular pitch $18^\circ$ |
| Sampled blade-side arrays | 17 span sections by 210 shared streamwise stations by 3 coordinates |
| Hub / shroud meridional blade length | Approximately 0.275062 m / 0.160838 m |
| Hub / shroud integrated wrap | Approximately $-54.902593^\circ$ / $-33.648652^\circ$; complete original angle laws retained |
| Target rake / maximum observed interval error | $0^\circ$ / approximately $2.31\times10^{-12}$ degrees |
| Automatic hub-TE reference / resulting shroud LE azimuth | Approximately $-54.902593^\circ$ / $-21.253940^\circ$ relative to the hub LE; no preserved-LE feasibility interval applies |
| Maximum target thickness | Approximately 6.371 mm, an explicit initial shaping assumption |
| Maximum relative paired thickness residual | Approximately 0.00843%; measured 3D side-pair distance versus target |
| Maximum absolute paired thickness residual | Approximately $4.83\times10^{-7}$ m, or 0.483 micrometres |
| Closed blade tessellation | 8,354 vertices and 16,704 triangles; no boundary edges, nonmanifold edges, orientation conflicts or degenerate triangles in the recorded checks |
| Repeated row | 20 closed blade components, 334,080 triangles; the visualization disk is a separate component, not a Boolean union |
| Reference passage | Exact sampled periodic rotation pairing; sampled blade containment passes; passage walls are untrimmed and `fluid_domain_watertight` is false |

The [generated diagnostics](output/reversible_impeller/diagnostics.json) and [manifest](output/reversible_impeller/manifest.json) carry the complete sampled results and limitations. They are reproducible local outputs, excluded from version control.

For comparison, the [original free-rake case](output/rake_study/free/geometry.html) has the same wraps and thickness residual as the current default but a $21.25394^\circ$ exit-azimuth mismatch. The current default removes that mismatch by section rotations and allows the LE to move. The earlier [preserved-LE case](examples/reversible_impeller_zero_rake_preserve_le.yaml) retains aligned leading edges and selects a common wrap of approximately $-50.628061^\circ$ within $[-52.029825^\circ,-49.226298^\circ]$. Its pronounced early hub turning and late shroud turning motivated the present TE-stacked choice. Both configurations remain geometric starting examples requiring aerodynamic review.

Verification is represented by [test_impeller_mapping.py](../../tests/test_impeller_mapping.py), [test_impeller_meanline.py](../../tests/test_impeller_meanline.py), [test_impeller_geometry.py](../../tests/test_impeller_geometry.py), [test_impeller_output.py](../../tests/test_impeller_output.py), and [test_impeller_convergence.py](../../tests/test_impeller_convergence.py). It covers analytic cylinder/radial limits; independent adaptive-quadrature references used only in tests; bowed-edge chain-rule derivatives; distance/inverse round trips; dimensional scaling; the source RMS inlet law; explicit paired thickness; shared boundary/orientation and periodicity checks; serialization/STL round trips; and numerical refinement at common surface points.

The rake extension adds [test_impeller_turning.py](../../tests/test_impeller_turning.py), covering analytic integral accuracy, monotonicity and endpoint preservation, and [test_impeller_rake.py](../../tests/test_impeller_rake.py), covering signed rake recovered from Cartesian derivatives, both controlled modes, intermediate spans, swept/inclined edges, infeasibility, scaling and closed surfaces. The original common-point convergence numbers belong to the free-rake angle law; they are not presented as a complete chordal-error audit of the new constrained-turning blade. Refer to the workflow README for the final combined test result.

The integrated disk/viewer version passed **199 tests**: 123 impeller, disk and browser-asset checks plus 76 maintained blade-law regression tests. The default-example regression verifies unchanged angle/thickness laws, the wrap-derived LE displacement and zero rake at intermediate spans. Disk checks cover a known annular-frustum volume, closed topology, rounded-root coverage and rejected invalid profiles. The HTML passed offline browser checks for cube alignment, rotation, reset, row/single selection, opacity/visibility, fullscreen and mobile resizing, with no page errors or network requests. The [browser validation record](output/reversible_impeller/viewer_checks/validation.json) and screenshots are generated by [check_report.py](check_report.py).

The mapping/integration tests and the existing mixed-flow regression passed during implementation. The legacy tests exercise the promoted implementation through the compatibility imports. A subprocess smoke test also built the reversible example with every nurbspy import blocked. Some Windows pytest temporary-directory fixtures required a run outside the restricted sandbox; their affected tests passed there. Refer to the workflow README for the final integrated test command and result, rather than mixing the historical assessment counts below with the new implementation suite.

These checks establish sampled geometric consistency. They do not prove continuous global intersection freedom, a global tessellation tolerance, watertight fluid-domain trimming, CAD manufacturability, CFD mesh quality or reversible aerodynamic performance.

## Active remaining plan

The next action is **review of the generated geometry and its recorded assumptions**, followed by the outstanding work in [next_steps.md](next_steps.md). The questionnaire is complete; reopening its resolved choices is not a prerequisite.

1. Review the zero-rake TE-stacked baseline in the Three.js report: gradual turning, resulting LE lean, hub attachment to the illustrative disk, wall visibility, inlet/outlet placement, maximum thickness, rounded-edge extents and the deliberate shrouded override. Use the preserved-LE and free-rake cases as comparisons. Nonmonotonic angle laws are deferred. Adopt or edit a labeled geometry configuration before treating it as the thesis machine.
2. Complete the geometry representation needed by the next consumer: trim hub/shroud patches at blade footprints and assemble a watertight periodic fluid boundary; strengthen cross-span and full periodic-intersection checks and error-controlled tessellation. Keep the already closed blade boundary separate from these passage tasks.
3. Explicit exit rake with preserved LE and explicit TE stacking are complete. Remaining candidates include fixed-station/centroid stacking, continuous spanwise feasibility bounds, adaptive chordal sampling and smoother span laws. Add them or another finite-thickness treatment only when evidence demonstrates a need, preserving explicit construction and declared thickness semantics.
4. If required, fit/export CAD with measured deviation and closure checks. STEP was not selected for this delivery. Mechanical wheel features need their own dimensions and acceptance criteria.
5. Add a Turbodash app once the geometry controls and exports are accepted. The app should call this API and persist this schema. Vaned-diffuser geometry, splitters/open-tip extensions and mesh generation are separate later milestones.

MastersThesis remains the consumer and research record. The reusable implementation, its development examples and the copied evaluation data currently live in Turbodash as requested; no source implementation was placed in or duplicated into the thesis checkout.

## Historical repository assessment: before implementation, 2026-11-10

The numbered study below preserves the repository findings, source references, original proposed milestones and pre-implementation checks. Its descriptions of future source work, possible open-tip/cutoff geometry and repository ownership are historical, not active requirements. The approved decisions and implemented record above take precedence. In particular, direct TurboMoC mapper reuse, unresolved questionnaire choices, an open tip, and a finite cutoff trailing edge are not requirements of the delivered implementation.

### 1. Recommendation

**There is enough material to build the complete aerodynamic impeller geometry without starting a new geometry library. There is not yet an integrated, validated centrifugal impeller geometry-and-mesh pipeline.** The main ingredients exist in different places:

- Curved, independently controlled hub and shroud walls in `development/mixed_flow`.
- A maintained camberline/thickness section kernel in `turbodash.geometry.blade_parametrization`.
- General curved-channel conformal mapping, passage construction, mesh generation, inspection and exports in TurboMoC.
- Meanline dimensions and metal angles for one shared reversible impeller in `MastersThesis/meanline`.
- An existing span-section-to-TurboGrid CFD pipeline in `MastersThesis/cfd`.
- Curve/surface evaluation, fitting, tessellation and interactive editing in `nurbspy`.

**Yes: provide independent hub and shroud angle distributions and thickness distributions.** They should be sections of one spanwise geometry model, with explicit stacking and common correspondence. Allow intermediate span controls without changing the schema. Independent controls need not all be free optimization variables initially: initialize the outlet angle from the meanline's single value and initialize thickness from its scalars, then introduce additional freedom deliberately.

**Develop the reusable feature in `turbodash.geometry.impeller`, and import it from thesis machine definitions and research drivers.** Reuse the maintained blade laws and suitable TurboMoC functions without duplicating their implementations. Nurbspy is available if a particular fitting, surface or editor operation needs it; it is not a required dependency of the geometry core. Mesh extensions remain future TurboMoC work.

**Implement source code and executable visualization/export examples first; integrate the app after geometry and export validation.** A browser drawing is an insufficient acceptance test for section correspondence, thickness, surface closure or export fidelity. Mesh validation belongs to the deferred phase.

### 2. What the code provided at the original assessment

#### 2.1 Blade parametrization and demo/application integration

The maintained implementation is [the production blade package](../../turbodash/geometry/blade_parametrization/), imported by the standalone modeller and sensitivity studies in [development/blade_parametrization](../blade_parametrization/README.md). Its current formulation is **conformal angle-law camber plus rational Denton thickness**. The older curvature/NACA implementation is archived under `geometry/legacy_parametrization`; `geometry/blade.py` now adapts callers to the maintained kernel.

| Existing code | Capability | Reuse or change |
| --- | --- | --- |
| [geometry.py](../../turbodash/geometry/blade_parametrization/geometry.py), `turning_angle` | Endpoint metal angles, turning location/concentration, angle or slope interpolation | Reuse per span section. Add a richer angle-law option only when required. |
| Same file, `denton_thickness`; [thickness.py](../../turbodash/geometry/blade_parametrization/thickness.py), `denton_body` | Body thickness, independent LE/TE fullness, rounding and optional one-sided relief | Reuse scalar laws; generalize the physical thickness measurement. |
| `build_section`, `Section` | Paired sides, closed sampled boundary, sampled self/neighbor intersection checks and diagnostics | Separate section laws from the axial/radial-specific metric and mapping. |
| `map_section_coordinates` | Direct sampled XYZ geometry | Currently only axial cylinders and radial planes between parallel endwalls. A radial blade here is not an axial-to-radial impeller. |
| [parameters.py](../../turbodash/geometry/blade_parametrization/parameters.py), `ParameterState` | Baseline, sparse overrides, validation, serialization and fingerprints | Extend to shared channel data and a section family. Preserve provenance and reproducibility. |
| [turbomoc_adapter.py](../../turbodash/geometry/blade_parametrization/turbomoc_adapter.py), `_boundary_splines`, `build_design` | Side interpolation, edge caps, transfer-error checks and passage construction | Generalize from one reference section to a serializable family. |

The existing angle law is monotone between its endpoints. It is a good initial family, but cannot reproduce every impeller angle distribution. Romei's hub distribution can have an interior maximum and is adjusted to satisfy trailing-edge rake. An optional spline/polynomial angle law may therefore become necessary; it should feed the same integrator and mapper rather than create another blade implementation.

The app architecture is useful: [blade_modeller.py](../../turbodash/apps/shared/components/blade_modeller.py) separates fast 2D evaluation from 3D preview, while [the experimental app](../../turbodash/apps/app_turbine_experimental/app.py) passes effective geometry state into meshing. Reuse its YAML/state handling, committed geometry updates, mesh invalidation and subprocess jobs. Its present meanline integration and presets are turbine-specific; they do not constitute a compressor design interface. The production [meshing adapter](../../turbodash/meshing/turbomoc.py), `_row_inputs`, explicitly rejects non-axial turbine results; its current cylindrical passage also flattens endwall variation.

The current [meanline adapter](../../turbodash/geometry/meanline.py), `parameters_from_meanline`, treats finite meanline trailing-edge thickness as an **unrounded shoulder value**, while rounding closes the actual tip. For the impeller, define whether an input is actual edge thickness, edge radius or shoulder thickness and preserve the intended quantity. Otherwise geometry and meanline blockage/loss calculations can disagree despite sharing the same named input.

Documentation needs a small reconciliation during implementation: [the earlier mesh integration note](../../docs/development/turbomoc_meshing.md) describes retired curvature/NACA and free-vortex behavior; current [sections.py](../../turbodash/geometry/sections.py) exports cylindrical, untwisted axial sections. The blade development README also gives an older fullness range; current validation accepts strict $(-1,1)$ subject to coupled admissibility. Code was used as the authority for this assessment.

#### 2.2 Curved hub and shroud channels

The [meridional package](../../turbodash/geometry/impeller/meridional/) is already a substantial starting point:

- [parameters.py](../../turbodash/geometry/impeller/meridional/parameters.py): station dimensions, inlet/outlet inclinations, area adapters and independent `WallShape` controls.
- [curves.py](../../turbodash/geometry/impeller/meridional/curves.py): smooth engineering walls, derivatives, curvature, normalized physical arclength and exact B-spline export. The quintic baseline plus bow is exported as degree six.
- [channel.py](../../turbodash/geometry/impeller/meridional/channel.py): span interpolation, area diagnostics and sampled positive-Jacobian checks.
- [blades.py](../../turbodash/geometry/impeller/meridional/blades.py): independently swept/bowed main-blade LE/TE extents and splitter meridional footprints.
- [editing.py](../../turbodash/geometry/impeller/meridional/editing.py) and [the editor adapter](../mixed_flow/explorer/control_point_editor.py): reproducible control-point offsets and direct reuse of nurbspy's editor.

Promote this implementation into `turbodash/geometry/impeller/`; leave the development explorer and examples importing it. Preserve the useful separation between wall shape and blade extent. Its present output is a **2D meridional footprint**, without circumferential camber, thickness, blade count or clearance geometry.

The current channel follows normalized wall arclength, not solved streamlines. Its area is an unblocked geometric station area, not blade throat area. Sampled Jacobian screening does not prove global intersection freedom. These distinctions should remain explicit in the eventual app.

#### 2.3 TurboMoC mesh and CAD capabilities

The local TurboMoC repository is [method_of_characteristics](../../../method_of_characteristics/), package `turbo_moc`; there is no sibling folder named `turbomoc`.

[meridional_channel.py](../../../method_of_characteristics/src/turbo_moc/geometry/meridional_channel.py) already accepts arbitrary hub/shroud SciPy B-splines through `GeometryDefinition`. Its general `MeridionalChannel` computes $m'=\int dm/r$ and maps to XYZ. The original recommendation was to extend this generic machinery directly. **That implementation recommendation is superseded by the protocol/table decision above:** the current impeller must preserve normalized-wall-arclength correspondence and once-through evaluation, which the existing raw-spline/`quad`/`solve_ivp` path does not supply. The existing TurboMoC mapper remains useful evidence and a future consumer integration point.

The important limitation is [blade_row.py](../../../method_of_characteristics/src/turbo_moc/geometry/blade_row.py), `Design.section`: sections are scaled copies of one template. [mesh_3d.py](../../../method_of_characteristics/src/turbo_moc/meshing/mesh_3d.py), `CoreDeformation.deform` and `volume_mesh`, inherit this assumption. The latter even bypasses deformation when section scale equals one; equal conformal chord does not mean equal geometry once twist and thickness vary independently.

An arbitrary section family requires changes to section generation, serialization, boundary correspondence, inflation/collar deformation and volume validation. Adding a callback to an in-memory `Design` alone is insufficient: saved designs, subprocess workers and reloaded mesh artifacts must reconstruct the same sections.

Reuse reference meshing, periodic node correspondence, CGNS export/audit and inspection. Extend the existing passage topology for splitters or tip gaps when required. A full-span single-blade sweep cannot resolve fluid passing over an open blade tip merely by changing boundary names.

The checkout also contains [cad_export.py](../../../method_of_characteristics/src/turbo_moc/geometry/cad_export.py): CadQuery-based blade/passage/disk assembly and STEP checks. Reuse it after section-family support is established and tested. Existing STEP/STL capability does not establish that a new twisted impeller is a valid solid. An aerodynamic blade/passsage model also does not specify a manufactured wheel's bore, backdisk, fillets or mechanical design.

#### 2.4 Nurbspy and the thesis teaching material

Nurbspy is an optional source of existing capabilities, not a reason to introduce a dependency. The current wall and section calculations already use NumPy/SciPy; retain those where they are sufficient. Use nurbspy only if an accepted requirement calls for its fitting, surface representation, tessellation or editor. Its [ruled-surface implementation](../../../nurbspy/src/nurbspy/numpy/nurbs_surface_ruled.py) and Coons patches require compatible boundary representations; they do not automatically loft unrelated section arrays. [Surface fitting](../../../nurbspy/src/nurbspy/fitting/_api.py) can approximate mapped patches with an explicit error tolerance. [Tessellation/STL export](../../../nurbspy/src/nurbspy/mesh.py) produces surface meshes, not CFD volume meshes. Keep any chosen nurbspy import in the relevant optional adapter; the base impeller API must import and evaluate without it.

Keep the analytic/sampled blade authoritative. A conformal/trigonometric map is not generally an exact NURBS surface just because its input walls are splines. Measure fitting deviation, closure and normals after CAD conversion.

[MastersThesis/turbo_parametrization](../../../MastersThesis/turbo_parametrization/README.md) is valuable student onboarding and provides analytic axial/radial checks. Its completed exercises deliberately use archived NACA code and simplified mappings. Preserve that educational role; research scripts should import the maintained production functions.

### 3. Coupling the reversible meanline machine

Use the geometry dictionaries from [centrifugal_compressor/design.py](../../../MastersThesis/meanline/centrifugal_compressor/design.py), `partial_geometry`, together with [geometry.py](../../../MastersThesis/meanline/centrifugal_compressor/geometry.py). Do not infer geometry from screenshots or plotting arrays: [plotting.py](../../../MastersThesis/meanline/centrifugal_compressor/plotting.py) explicitly draws assumed quarter-ellipse walls, and [reversible_machine/figures.py](../../../MastersThesis/meanline/reversible_machine/figures.py) constructs illustrative plan-view profiles.

#### 3.1 Explicit initial adapter

For the current axial-inlet/radial-outlet drawing convention:

| Meanline quantity | Meridional input |
| --- | --- |
| `D_1_h`, `D_1_s` | `inlet_hub_radius = D_1_h/2`, `inlet_shroud_radius = D_1_s/2` |
| `D_2` | `outlet_radius = D_2/2` |
| `b_2` | `outlet_width = b_2` |
| `L_z` | **`axial_length = L_z - b_2/2`** |
| Existing axial/radial station convention | `inlet_flow_angle_deg = 0`, `outlet_flow_angle_deg = 90` |
| `beta_1b_h`, `beta_1b_s`, `beta_2b` | Hub/shroud inlet metal angles and initial common outlet metal angle |
| `t_b_1`, `t_b_2` | Dimensional edge-thickness constraints, after their measurement convention is defined |
| `Z_r_full`, optional splitter fields | Explicit integer blade-set and blade-within-set definitions |
| `is_shrouded`, `eps_a`, other clearance fields | Casing/tip construction and mesh requirements, with a declared distance direction |

The axial-length conversion matters: the prototype places the **outlet centre** at `axial_length`; the thesis convention places the outlet hub at `L_z` and shroud at `L_z-b_2`. Passing `L_z` unchanged shifts both outlet endpoints by half the exit width. The station inclinations above are geometric conventions, not independently computed flow-field information.

The meanline already defines an inlet span law:

$$
\tan\beta_{1b}(r)=\tan\beta_{1b,\mathrm{rms}}\frac{r}{r_{\mathrm{rms}}},
\qquad r_{\mathrm{rms}}=\sqrt{\frac{r_h^2+r_s^2}{2}}.
$$

The committed reversible example's starting inputs imply approximately $-32.65^\circ$ at the hub, $-47.28^\circ$ at RMS radius, and $-54.29^\circ$ at the shroud. These are example-derived values, not a selected thesis optimum. Linear interpolation of the two endpoint **angles** at geometric midspan will not preserve this RMS condition. Initialize from the tangent law or enforce an explicit RMS-radius constraint.

What remains unspecified includes wall turning controls, complete chordwise angle laws, independent thickness laws, stacking, detailed edges, and the actual blade-tip surface. The design builder presently sets equal scalar inlet/outlet thicknesses; that does not determine maximum body thickness or spanwise taper.

#### 3.2 One physical impeller for both operating modes

[reversible_machine/design.py](../../../MastersThesis/meanline/reversible_machine/design.py), `turbine_partial_geometry`, maps the same compressor impeller into turbine station conventions. The current [decision record](../../../MastersThesis/meanline/DECISIONS.md) removed `same_blades`: the impeller is always shared. Keep one canonical solid and one geometry fingerprint.

Use stable eye/rim and plus/minus-side identifiers. Compressor LE/TE labels are geometric reference names; the inflow edge changes in turbine operation. Do not infer a universal physical pressure side from the sign of camber turning, or mirror/regenerate the rotor from a turbine plot.

Turbine operation changes inlet/outlet roles and signed rotation in the CFD frame. [setup_case.py](../../../MastersThesis/cfd/pipeline/setup_case.py) already contains this mode handling. Meanline rotational speeds are positive magnitudes and require explicit conversion to signed Cartesian rotation.

Blade/vane counts are continuous optimization variables in the meanline. Require integer realization and record/re-evaluate its consequences before creating periodic geometry; do not silently inherit rounding from plotting. The reversible examples use 20 full blades and nonzero clearance (`clearance_ratio = 0.02`), while their vane count can be noninteger. The reversible design builder currently creates full blades only.

The current record does **not** adopt an S-90 design. Begin with a labeled, reproducible example or explicitly selected candidate. That selection is needed for the final machine result, but does not block developing the reusable kernel.

### 4. Original proposed geometry contract

#### 4.1 One meridional map, with declared station correspondence

Use $u\in[0,1]$ in the streamwise direction and $v\in[0,1]$ from hub to shroud. Let $C(u,v)=(z,r)$ describe the channel. Here the channel's $u$ pairs the walls by their respective normalized physical arclengths. Define the blade footprint using its own local $u$:

$$
P(u,v)=C\left(u_{LE}(v)+u[u_{TE}(v)-u_{LE}(v)],v\right),
\qquad 0\leq u,v\leq1.
$$

The two uses of $u$ are local to their maps: $u=0$ and $u=1$ denote the channel endpoints in $C$, and the blade reference edges in $P$. The composition explicitly relates them. This is the prototype's construction expressed with the chosen notation. Extend a generic mapping interface to expose points, derivatives, meridional/conformal distances and their inverses, supported extensions, and serializable definition data. The new impeller API uses `u` and `v`; adapters may translate legacy argument/field names without changing the existing APIs.

**Do not simply pass `WallCurve.as_bspline()` into TurboMoC and assume equivalence.** The prototype pairs walls at normalized physical wall arclength. Its exported spline uses a polynomial parameter, and TurboMoC pairs the splines at common polynomial parameter values. That parameter is not the channel's arclength-based $u$; using the same variable name does not make the maps equivalent. The walls would remain identical, but the interior surfaces would change. Moreover, TurboMoC's current end edges are straight interpolants of wall endpoints, whereas `BladeEdges` permits bowed spanwise edges.

Preserve the chosen channel and blade map through a richer generic mapping contract. A fitted common-station representation is another possible adapter, but must carry measured point/derivative error and represent edge bow. A restricted first mapping experiment may use compatible unswept geometry, explicitly labeled as such; silently dropping sweep or bow is unacceptable for the final implementation.

#### 4.2 Reuse section laws on each curved stream surface

At a fixed $v$, let $P_u=\partial P/\partial u$ and define the derived conformal distance

$$
m'(u,v)=\int_0^u\frac{\|P_u(u,v)\|}{r(u,v)}\,du.
$$

The existing angle law $B(u,v)$ takes a normalized conformal argument. Evaluate its physical distribution and integrate the wrap as

$$
\beta(u,v)=B\left(\frac{m'(u,v)}{m'(1,v)},v\right),
\qquad
\theta(u,v)=\theta_0(v)+\int_0^u\tan\beta(u,v)\frac{\|P_u(u,v)\|}{r(u,v)}\,du.
$$

The integration variable is local to each integral; using $u$ in both the integrand and upper limit is intentional. Then map camber/side samples into the canonical Z-axis frame:

$$
\mathbf{x}=(r\cos\theta,r\sin\theta,z).
$$

Here $m'$ starts at the local blade reference edge and $\theta_0(v)$ is the circumferential stacking law. The angle is signed from the local meridional tangent toward positive mathematical azimuth. Meridional distance, conformal distance and azimuth are derived geometric quantities, not additional independent parameter-space coordinates. A distribution specified versus physical meridional distance, as in Romei, is not the same distribution versus normalized $m'$ on a variable-radius surface.

The single-valued camber construction requires $-\pi/2<\beta(u,v)<\pi/2$ throughout each section. Any new nonmonotone angle law must enforce this interior condition, not just validate its endpoints.

Independent endpoint curves need stacking; otherwise their relative circumferential offset is undetermined. Also, endpoint angles plus a complete angle law and LE stacking already determine the TE location. Prescribed rake adds an integral constraint and requires solving for a free shape/stacking parameter rather than setting all quantities independently.

Evaluate thickness-offset side points at **their own** $m'$ coordinates. Define upstream/downstream extensions or explicit cutting surfaces for protruding edges; do not clamp offsets to the camber interval. A radial cutoff TE is a distinct edge construction, not automatically the existing rounded-tip family.

#### 4.3 Thickness, span interpolation and closure

Reuse Denton/fullness/rounding laws for hub and shroud, with linked defaults and independent overrides. Initially retain the current paired-side measurement convention, generalized and measured on the actual curved surface. The local metric $r$ only gives an infinitesimal thickness conversion; finite paired separation, geodesic distance, minimum side separation and 3D surface-normal thickness differ. A physical-thickness target needs a declared measurement and a residual check, potentially an offset solve.

Use common normalized stations and patch identifiers across span. Current `build_section` adds samples at each section's own thickness peak, edge shoulders and relief locations, so independent outputs cannot safely be stacked by array index. Resample/evaluate to an agreed common correspondence before fitting or meshing.

Start with an explicit span interpolation of laws/stacking, preserving the meanline inlet constraint. Inspect intermediate sections, not just hub and shroud. Straight interpolation of XYZ boundary points creates a ruled surface but generally differs from interpolating radius/azimuth on the prescribed channel. Choose and document the construction; a claim of ruled manufacturing geometry would require a separate constraint.

Produce plus/minus skins, LE/TE closures and appropriate hub/tip closures with consistent shared edges and normals. Validate the blade and the fluid passage separately. For an open impeller, the casing is not the blade tip: define and construct clearance in the intended physical direction. A zero-gap code test is useful but does not represent the current nonzero-clearance meanline machine.

#### 4.4 Units and coordinate adapters

| Consumer | Length and coordinate contract |
| --- | --- |
| Canonical impeller / thesis meanline | Metres; explicit degree inputs, radian internal angular arrays; canonical Z-axis XYZ |
| Mixed-flow engineering walls | Metres, `(z,r)` |
| TurboMoC geometry/native mesh | Millimetres, meridional `(r,z)`; reference chart $q=(-R_{ref}\theta,-R_{ref}m')$ for the Turbodash mathematical azimuth |
| TurboMoC CGNS export | Metres; Z rotation axis |
| Existing thesis TurboGrid input | Declared file units; `(X,R\theta,R)` sections; X rotation axis after explicit frame conversion |

Centralize these conversions and test them using known points, signed tangents and periodic rotation. Store units, axis, handedness, integer counts, schema version, source case and dependency revisions in the artifact manifest.

#### 4.5 Theory note as a required deliverable

Maintain [impeller_geometry_theory.md](impeller_geometry_theory.md) with the implementation. The initial note defines the construction and explicitly marks unresolved modeling choices; each implemented feature must update its corresponding derivation and verification example. It must cover:

1. The $(u,v)$ convention, coordinate frames, units and the distinction between a function's local parameter and derived arclength/conformal coordinates.
2. Wall station correspondence, independent edge placement, the chain rule and channel/footprint Jacobians.
3. Physical and conformal arclength, inversion, camber angle integration and stacking constraints.
4. Spanwise angle/thickness laws, the RMS-radius inlet constraint and optional intermediate sections.
5. Side offsets, the selected finite-thickness measurement, edge closure and clearance construction.
6. Surface derivatives/normals, periodic replication, CAD approximation and geometric validation limits.
7. The correspondence between equations, public functions, configuration fields and runnable examples, including retained legacy adapters.

Use `$...$` and `$$...$$` for Markdown math. Use $u$ and $v$ consistently for parameter-space variables; repeated integration symbols and limits are allowed as above. Do not introduce different dummy variables merely to express a function of $u$. The note is complete only when its equations, implemented options and example outputs agree.

### 5. Original repository-placement proposal

| Repository | Responsibility and proposed placement |
| --- | --- |
| **Turbodash** | **Chosen package: `turbodash/geometry/impeller/`**, i.e. `git_projects/turbodash/turbodash/geometry/impeller/`. Put the feature's channel, blade-footprint, spanwise geometry, diagnostics and public API here; reuse `geometry/blade_parametrization/` rather than copy its laws. Development examples and the app import the package. No separate sibling `geometry/meridional/` package is needed for this work. |
| **TurboMoC** | Reuse or extend suitable generic mapping and CAD functions through thin adapters. Keep a single authoritative mapping implementation and the chosen station correspondence. Future passage topology, mesh deformation and mesh quality/export remain here. |
| **nurbspy** | Optional existing spline/fitting/editor capability only when required. Do not add it to the base impeller requirements or introduce a NURBS fitting step merely because it is available. |
| **MastersThesis** | Own selected machine/case data, the meanline-to-geometry adapter, parameter studies and research evidence. A proposed `cfd/pipeline/parametrize.py` and `cfd/reversible/` fit the existing pipeline/data convention. Keep generic geometry out of the meanline solver and teaching skeletons. |

The public import boundary is `MastersThesis -> turbodash.geometry.impeller`. The package reuses the maintained section laws and the existing NumPy/SciPy numerical machinery. Optional CAD/editor adapters can call TurboMoC or nurbspy when a requested operation needs them; they must not become mandatory merely through package import. TurboMoC must not import the thesis project or Turbodash to regenerate a saved geometry. Keep one canonical mapping implementation and thin conversions instead of parallel copies. Keep headless geometry imports light: package initializers currently pull in broader application/solver dependencies even where the equation modules themselves are independent.

Use editable installs for development and record exact revisions for thesis runs. Turbodash pins its optional published dependency to `turbo_moc==1.2.0`. TurboMoC initially identified itself as `1.2.1` with uncommitted changes; a concurrent update advanced the checkout to `1.2.2` (`cd96056`) before the final check. The geometry/mapping/mesher implementation files assessed here were unchanged by those commits. Validate the chosen implementation against a recorded dependency state, then update the published pin when a compatible release is available. Local source discovery and development nurbspy APIs are not a reproducible release specification.

Choose Python **3.12 or 3.13** for the shared development environment: the companion TurboMoC checkout requires `>=3.12,<3.14`, whereas Turbodash alone also permits 3.11.

### 6. Mesh strategy

Use two consumers of the **same canonical geometry**, rather than implementing two geometry generators.

**First thesis CFD route: reuse the existing TurboGrid pipeline.** [cfd/pipeline/geometry.py](../../../MastersThesis/cfd/pipeline/geometry.py), `write_row`, already reads ordered hub-to-shroud sections, multiple blade types, blade-set counts, units and pitch shifts. [mesh.py](../../../MastersThesis/cfd/pipeline/mesh.py), `mesh_row`, supports inlet/outlet opening stations, shroud tip clearance and first-element offsets. Add the export adapter and a reversible-machine configuration; do not copy HECC dimensions into the new case. Confirm access to the existing licensed runtime when executing this milestone; it was not exercised in this assessment.

**Reusable in-app route, deferred: extend TurboMoC.** Make `Design.section` return the actual section at a supplied span coordinate $v$, with stable side/edge correspondence; persist its definition and deform the reference mesh onto each section. Update blade collar/inflation handling so it follows the new boundary geometry instead of isotropic scaling. Preserve periodic pairing, check Jacobians throughout the volume, and fail visibly when the mapping cannot maintain mesh quality.

Complete impeller meshing must address:

- Upstream/downstream channel extensions and physical periodic surfaces.
- Span clustering and blade/endwall boundary-layer resolution, with first-cell size related to the selected CFD operating point and turbulence treatment.
- Actual open-tip clearance, if the chosen impeller remains unshrouded; this changes topology and needs dedicated gap cells.
- Splitter passages if the chosen design includes splitters: one repeating blade set may contain both a main blade and a splitter. Its periodic angle is set by the number of sets, not simply total blade count.
- Export/reimport, named boundaries and solver interpretation. Retain `blade`, `hub`, `shroud`, `inlet`, `outlet`, `periodic_left`, `periodic_right` where applicable, adding explicit tip/other wall distinctions when topology requires them.

Positive Jacobians and a successful CGNS audit are necessary geometry checks, not evidence of CFD mesh independence. Refinement and solver checks belong to the subsequent thesis evaluation in both operating modes.

### 7. Original implementation sequence and acceptance gates (superseded)

| Step | Work and owner | Concrete acceptance result |
| --- | --- | --- |
| **0. Define the theory and interface** | Theory note plus `turbodash.geometry.impeller` schema: $(u,v)$, labeled meanline snapshot, dimensions, signs, count realization, clearance and edge semantics | The note states equations and unresolved choices; the manifest reproduces its inputs; no candidate is mislabeled as the adopted design. Retain an illustrative `code_test_...` fixture separately. |
| **1. Promote and connect walls** | Channel/footprint code inside `geometry/impeller/`; thin thesis adapter | Existing wall tests still pass; exact eye/outlet dimensions and areas match the snapshot, including `L_z-b_2/2`; scripts plot both walls and actual blade edges; note and API agree on $u,v$. |
| **2. General curved section mapping** | Impeller API reusing/extending suitable generic mapping code | Canonical interior points, derivatives, LE/TE bow and inverse coordinates agree across consumers; recover axial-cylinder and radial-plane analytical cases. No second mapping implementation and no nurbspy requirement for core evaluation. |
| **3. Independent span laws** | Turbodash section family, stacking and thickness metric | Hub/shroud controls work independently; the RMS inlet constraint and endpoint metal angles hold; intermediate sections remain valid; measured thickness and edge geometry meet declared tolerances. |
| **4. Closed blade and passage** | Turbodash assembly plus existing CAD/export adapters | Scripts generate one blade, one periodic passage and a repeated row; matched patch edges, outward normals, no sampled intersections, positive passage gaps, measured CAD fit error and export/reimport checks. Start with a labeled zero-gap fixture; complete actual clearance geometry before claiming the target impeller. |
| **5. Deliver the documented geometry workflow** | Theory note, reusable API, examples and thesis import driver | A standalone command reproduces the blade/row and agreed exports; equations link to implemented functions and independent checks; importing/evaluating core geometry works without nurbspy. Record any optional export dependency. |
| **6. Optional app integration** | Turbodash shared components, after geometry/export acceptance | Channel, hub/shroud sections, stacking and diagnostics edit the same API used by scripts; YAML round-trip works; previews and exports share a fingerprint; invalid edits are handled. No mesh prerequisite. |
| **Later: vaned diffuser and meshing** | Same row model and thesis data; deferred mesh strategy in Section 6 | Extend geometry to the pivoting vane row and interfaces when requested; schedule mesh topology, clearance cells, export and refinement as a separate phase. |

Steps 0-5 define the current geometry phase. Deliver the theory note alongside a headless command that reads a labeled meanline case, draws its meridional channel and hub/mid/shroud sections, generates a 3D blade and writes the agreed geometry formats. It should have no dependency on a running Dash application or nurbspy for core evaluation.

Keep stable section correspondence, explicit surfaces and reproducible exports so that later meshing can consume the geometry. Do not implement or require a mesh experiment during the current phase. The existing TurboGrid route and TurboMoC extensions remain options for that later work.

#### Verification cases to retain

1. Cylinder, radial plane and straight/constant-angle limits with independently computed positions and metal angles.
2. A smooth 90-degree channel; unequal hub/shroud angle and thickness laws; an equal-chord case with different section shapes to catch the old scale-one shortcut.
3. Swept and bowed LE/TE, common station correspondence and a single global scale change.
4. Finite actual TE thickness, rounded edges, edge protrusion/cutting, thin-tip limits and invalid crossing inputs.
5. Channel/section serialization and source-to-preview-to-export geometric agreement within declared tolerances.
6. Complete blade-set periodicity, surface regularity/orientation, minimum gaps, closure, clearance and splitter geometry as they are added. Volume-cell and mesh-boundary checks are deferred.
7. Reversible operation: unchanged geometry hash and physical surfaces, with explicit mode metadata; CFD validation is deferred.
8. Theory-to-code consistency: $(u,v)$ notation, chain-rule derivatives, angle/thickness coordinate definitions, equation-to-function links and reproducible limiting examples. Core evaluation without nurbspy must pass.

Numerical tolerances should be dimensionally scaled and declared before accepting a result. Report sampling and fitting errors; do not use a visually smooth surface as a substitute for these checks.

### 8. How Romei informs this plan

The supplied [Romei et al. paper](<../mixed_flow/papers/Romei et al. - 2023 - Guidelines for the aerodynamic design of sCO2.pdf>), Section 2 and Fig. 1 (printed pp. 111014-2/3), supports separating meridional walls, hub/shroud angle laws, stacking and thickness. It uses a ruled construction, separate angle laws, LE lean, a hub-law adjustment for TE rake, and a trimmed/rethickened splitter. This is useful architectural evidence, not a requirement to replace the current camber/thickness families.

Its case is a shrouded near-critical CO2 impeller with a **vaneless** diffuser. It does not establish the target machine's open-tip geometry, reversible blade behavior or vaned diffuser design. Its wall/splitter optimization also affects operating range, so geometric refinement should not be judged only by design-point efficiency. See the existing [detailed local review](../mixed_flow/romei_design_guidelines.md), the [publisher DOI](https://doi.org/10.1115/1.4063311), and the [authors' institutional record](https://re.public.polimi.it/handle/11311/1257091).

Use the paper to motivate controls and validation questions. Do not copy its dimensions, angle/thickness coordinate definitions, splitter fraction or near-critical performance recommendations into thesis defaults without checking their meaning and applicability.

### 9. Original assessment evidence and limits

Source snapshot inspected:

| Repository | HEAD inspected | Working-tree context |
| --- | --- | --- |
| Turbodash | `e16161c` | Clean before this assessment document |
| TurboMoC / method_of_characteristics | Initially `74434ed`; final `cd96056` | Existing edits were committed concurrently and version advanced from 1.2.1 to 1.2.2; final working tree clean |
| nurbspy | `ec79b88` | Clean at initial inspection |
| MastersThesis | `255e6694` | Clean at initial inspection |

Checks performed for this assessment:

- `python -B -m pytest development/mixed_flow/tests -q -p no:cacheprovider`: **125 passed**, using Turbodash's environment and a workspace output directory. Initial sandbox runs had Windows temporary-directory permission errors; the authorized run outside that sandbox passed.
- `python -B -m pytest tests/test_blade_parametrization_kernel.py -q -p no:cacheprovider`: **46 passed**.
- TurboMoC's existing `tests/test_design_meshing.py::test_actual_reference_volume_and_cgns_export[curved]` was exercised directly with its fixture and a normally created output directory after the same pytest temporary-directory issue. It produced **28,233 hexahedra, 4,803 prisms and 41,612 nodes**, with positive minimum determinant, 260 periodic pairs and a passing CGNS boundary-partition assertion. Its minimum SICN was about **0.016**, so successful topology is not evidence of good resolution/quality everywhere. This is a mildly curved, similar-section fixture, **not** a full 90-degree impeller or independent hub/shroud geometry. Local artifacts are in `method_of_characteristics/output/assessment_direct_mesh_20261110`.
- TurboMoC's numerical checks `test_physical_chords_and_radial_endpoints` and `test_general_curved_meridional_mapping` also passed.
- Read the supplied PDF text and visually inspected the complete geometry-generation page and Fig. 1 page. Verified bibliographic identity against the authors' institutional record.

The TurboMoC mesh check used the editable 1.2.1 runtime with Gmsh 4.15.2. It was not repeated as a released-package compatibility test for 1.2.0 or 1.2.2. The concurrent changes primarily concerned the nurbspy editor dependency and release metadata; they do not remove the section-family or tip-gap limitations identified above.

The checks above belong to the original assessment. No new centrifugal impeller, licensed TurboGrid run, CFD simulation or manufactured CAD validation was completed by it. Existing tests establish the present components, not the proposed integrated pipeline. At that point, the questionnaire and theory note were planning artifacts and source implementation was future work. The implemented status, accepted choices and reproducible geometry workflow at the top of this document now supersede that status.

This work directly supports thesis SQ1's meanline-to-geometry-to-CFD objective in [research/questions.md](../../../MastersThesis/research/questions.md). It does not require moving the research record into Turbodash: thesis cases, adopted choices and comparisons remain in MastersThesis.

**Historical next-step proposal, now completed or superseded:** the assessment originally proposed resolving the questionnaire and implementing the meanline adapter and wall geometry before adding 3D controls. The questionnaire is now fully reviewed, the adapter and 3D geometry exist, and the approved scope is a shrouded main-blade impeller with both edges rounded. There is no pending questionnaire gate. The active next step is review of the generated geometry and the remaining geometry work listed above and in [next_steps.md](next_steps.md).

The original assessment left candidate selection, integer realization, clearance and splitter semantics open. For this implementation, the committed reversible example supplies an explicitly labeled starting seed with 20 full blades; the user selected shrouded geometry, zero modeled tip gap and no splitters. Selecting and aerodynamically validating an adopted thesis design remains separate research work.
