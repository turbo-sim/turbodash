# Impeller geometry: next steps after the first implementation

2026-11-10. The reusable [geometry package](../../turbodash/geometry/impeller/) and [runnable examples](README.md) now construct the approved shrouded main-blade geometry, including explicit exit-rake control. The following priorities follow from the implementation and its measured results. Mesh generation remains a later phase.

The current app also exposes a rounded hub nose, solid-centre/optional-bore disk settings and **Download full STL**. This export fuses and validates the whole rotor for slicing; the default has a 35 mm nose and no bore. These controls and the automated export are implemented, while mechanical sizing and fluid-domain closure remain separate work below.

## 1. Review and adopt the geometric design

Review the [zero-rake baseline HTML](output/reversible_impeller/geometry.html), editable [zero-rake YAML](examples/reversible_impeller_zero_rake.yaml), and [rake comparison](output/rake_study/rake_comparison.png). The four interactive cases show [original free rake](output/rake_study/free/geometry.html), [zero rake with preserved LE](output/rake_study/zero_rake/geometry.html), [positive rake with preserved LE](output/rake_study/positive_rake/geometry.html), and [TE stacking](output/rake_study/te_stacked/geometry.html). The original [source-derived seed](examples/reversible_impeller.yaml) remains unchanged. The [independent-section study](output/section_study/comparison.png) deliberately continues using that legacy free-rake seed to isolate its original control edits.

Select wall turning, blade footprint, hub/shroud thickness, rake and chordwise turning controls suitable for the thesis machine. The current default uses zero-rake TE stacking, preserving the original gradual monotonic angle laws while allowing leading-edge lean to result from the section wraps. Inspect this leaned edge with the illustrative disk and Three.js orientation cube. Nonmonotonic angle laws are deferred. The earlier [preserved-LE case](examples/reversible_impeller_zero_rake_preserve_le.yaml) remains available to show the concentrated early hub/late shroud turning caused by its simultaneous constraints. Existing values remain traceable starting choices, and the source meanline run did not satisfy all design constraints or represent an optimized design.

Before calling a case the adopted reversible machine, record its meanline candidate and decide whether its shrouded/no-clearance assumptions and rounded edge shoulders should also be used in a fresh meanline calculation. Geometry alone does not update the inherited performance. Preserve one physical rotor for both operating modes and stable plus/minus side labels.

**Acceptance:** an adopted case with provenance and explicit shaping assumptions, sensible wall/section plots, and documented departures from the original meanline dimensions or inlet-angle law.

## 2. Finish fluid-boundary topology when required

The single blade is closed and manifold, including endwall-following caps and shared rounded seams. The current passage endwalls are untrimmed reference patches, so combining exported surfaces does not produce a watertight fluid region.

Add exact/shared blade-footprint boundaries to the endwall patches, trim the blade region out of the fluid walls, and sew inlet/outlet/periodic/endwall/blade interfaces. Use the authoritative mapper and section data; avoid independently refitting shared edges. Retain the full channel and any genuine rounded-edge protrusions. Prefer an existing CAD/surface-topology facility if it fits the consumer's actual requirements; do not introduce nurbspy merely to store sampled arrays.

**Acceptance:** a consistently oriented fluid boundary with no gaps, duplicate internal faces or nonmanifold edges, plus exact periodic correspondence and a measured geometric tolerance. This is a geometry task even though its first consumer may be a mesher.

## 3. Strengthen geometric validity and sampling

The present checks sample the channel and blade sections. Adjacent periodic section contours are checked; arbitrary between-span collisions, non-adjacent copies and extreme multi-turn geometry are not certified. Topological closure and positive volume do not establish global nonintersection.

Add broad-phase spatial bounds and robust triangle/patch intersection checks, then periodic checks covering every relevant copy. Check side-surface Jacobians and extreme wall/stacking combinations. Add adaptive station refinement based on chordal deviation, curvature and normal variation, including hub/shroud caps and rounded tips.

The original free-rake numerical audit gives about $1.46\times10^{-10}$ m common-point coordinate change on refinement, but this is a quadrature/map convergence result, not a bound on facet error or a full refinement audit of the new constrained turning law. Measure tessellation error against newly evaluated intermediate points and report it in the manifest. Use common shared-edge sampling for all patches, with particular attention to the sharper constrained-turning transitions.

In `preserve_le`, the rake reference is screened on 257 fixed spans plus control spans, independently of visualization resolution. Every subsequently evaluated section is checked again, but this remains sampled feasibility. Add continuous or conservatively bounded spanwise feasibility checks when aggressive preserved-LE designs demand them; do not confuse a passing screen with a proof between its stations. TE stacking does not use this feasibility interval because LE placement is dependent.

**Acceptance:** intentionally folded/intersecting fixtures fail, valid difficult shapes pass, and each export meets a declared physical chordal tolerance.

## 4. Review the completed rake controls and extend stacking only as needed

Explicit rake control is implemented. `rake.mode: preserve_le` retains leading-edge positions and endpoint metal angles while using a monotone slope law with an analytically prescribed integral. `trailing_edge` preserves the complete original angle laws and rotates every evaluated section to the target edge, allowing the LE to move. `free` retains legacy behavior. The generator now defaults to the zero-rake `trailing_edge` case; the comparison script keeps the earlier preserved-LE examples available.

The constrained scalar law lives in [blade_parametrization/turning.py](../../turbodash/geometry/blade_parametrization/turning.py); [impeller/rake.py](../../turbodash/geometry/impeller/rake.py) defines the physical edge target and mode-specific stacking. Neither controlled mode needs nonlinear fitting or thickness iteration. The current default's automatic TE reference is about $-54.902593^\circ$, retaining the original hub position. Its shroud LE azimuth relative to the hub is about $-21.253940^\circ$; this is not a prescribed constant lean angle. Turning-location, concentration and interpolation remain active. The archived `preserve_le` example uses approximately $-50.628061^\circ$ inside the screened interval $[-52.029825^\circ,-49.226298^\circ]$; its turning-location and interpolation selectors are inactive.

Remaining stacking extensions are fixed-section-station alignment and centroid-based alignment. The latter needs a precise centroid definition and quadrature. Prescribing both edge curves while also retaining the entire original angle law remains overconstrained; it should not be added as an apparently independent control. The [theory note](impeller_geometry_theory.md) derives the implemented rake law and compares the remaining choices.

**Acceptance:** chosen stacking locus is reproduced, endpoint metal angles remain controlled, resulting sections are valid, and the theory/API state which variables are dependent.

## 5. Refine span laws and thickness only where evidence calls for it

Independent hub/shroud parameters and optional intermediate controls work now. Piecewise linear span interpolation can introduce derivative kinks at intermediate controls. Add an explicitly selected shape-preserving smooth span interpolation if surface-normal continuity is needed, preserving positive thickness and angle bounds. Document that an independently edited middle inlet angle can depart from the source RMS law; optionally add an explicit RMS diagnostic/constraint mode.

The once-through thickness method gives a maximum paired-distance residual of about **0.00843%** in both the current TE-stacked zero-rake baseline and its original free-rake seed, **0.00918%** in the archived preserved-LE case and **0.00708%** in the legacy independent-section study. Rigid section rotations preserve the measured thickness. There is no evidence that these cases require a complex iterative correction. Stress-test thicker blades and tighter channels first. If a different accuracy requirement emerges, consider an explicit higher-order local correction or a different declared measurement. Keep paired Euclidean thickness distinct from full-surface-normal or minimum skin distance.

**Acceptance:** geometry meets a chosen physical thickness tolerance over the actual design envelope; requested smoothness is demonstrated with derivatives rather than only pictures.

## 6. Optional geometry extensions

| Extension | Reuse and necessary additions |
| --- | --- |
| Splitters | Reuse the meridional edge model and section kernels; add shortened independent laws, pitchwise position and parent-passage checks. The approved baseline uses main blades only. |
| Vaned diffuser | Reuse angle/thickness laws and the section-family architecture; introduce its stationary channel, dimensions, vane count and impeller-to-diffuser gap. Do not duplicate the impeller evaluator. |
| Mechanical wheel/cover | Replace the illustrative disk assumptions with justified back thickness, bore, attachments and fillets; add cover thickness if required. Shrouded aerodynamic walls and a separate visualization disk do not specify a manufactured rotor. |
| STEP export | Reuse a suitable existing fitting/CAD adapter only when needed, with fit tolerance, sewing, orientation and reimport checks. The mapped blade is not automatically an exact NURBS surface. |

## 7. Extend the app and integrate downstream consumers when required

The standalone [impeller app](../../turbodash/apps/app_impeller/app.py) now edits the same validated YAML and calls the script evaluator. It reuses Turbodash's controls and diagrams, the native shared wall editor, and the report's Three.js renderer with a persistent camera. Validated hub/shroud drag refinements are applied to the complete geometry; stale responses cannot replace newer edits. YAML saving is disabled while the current draft is invalid or pending. Remaining UI work can extend this app to intermediate span controls, disk dimensions and the eventual central turbine workbench without introducing another geometry implementation.

For later TurboMoC integration, extend its consumer contract from a scaled reference section to an independently controlled section family. Preserve normalized wall-arclength correspondence, curved footprints, full side coordinates and explicit metre-to-millimetre/frame conversion. Its existing raw-spline channel and ODE inverse are not interchangeable with this protocol without an adapter. Aim to share the authoritative mapper rather than maintaining two impeller paths.

The existing MastersThesis geometry-to-CFD pipeline is another potential consumer. Evaluate its required section ordering and units before writing an adapter. Do not begin CFD volume meshing or mesh-quality work as part of the current geometry-only scope.

## Maintained deliverables

Keep source inputs, provenance, examples, tests and this note in Turbodash. Thesis studies import the package and cite the selected version/case. Keep the [theory](impeller_geometry_theory.md) aligned with every implemented extension, using $(u,v)$ consistently and recording coordinate conversions rather than introducing new independent parameter names.
