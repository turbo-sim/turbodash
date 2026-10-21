# Blade parametrization: status and integration plan

This directory contains the standalone blade geometry implementation and sensitivity studies for its camberline and thickness controls. The model generates complete sections in the conformal $(m',\theta)$ plane and maps them onto axial cylindrical or radial planar stream surfaces.

**The standalone application is implemented. Integration into TurboDash's production meanline and geometry views remains the next development task.** The workflow described below is the agreed integration plan, not current production behavior.

## Documentation and implementation

| Resource | Content |
| --- | --- |
| [Blade construction](parametrization_blades/README.md) | Mathematical formulation, coordinate conventions, physical mapping, numerical evaluation, and limitations. |
| [Parameter guide](parametrization_blades/parameter_guide.md) | Parameter definitions, valid ranges, defaults, and design guidance suitable for application help. |
| [Camberline study](parametrization_camberline/README.md) | Turning-location and concentration sensitivities for impulse and reaction sections. |
| [Thickness study](parametrization_thickness/README.md) | Rational Denton modification and independent LE/TE fullness sensitivities. |
| [Reusable geometry package](parametrization_blades/blade_geometry/) | The single implementation used by the standalone app and both studies. |

From the TurboDash repository root:

```powershell
poetry run python development/blade_parametrization/parametrization_blades/launch.py
```

The launcher opens the browser with the reaction preset and 3D preview enabled. Append `--case impulse`, `--port 8057`, or `--no-3d` as required. Install optional preview dependencies once with `poetry install -E mesh`. An occupied port requires stopping the existing server or selecting another port.

## Current formulation

The parametrization retains **camberline plus thickness** as the design model:

1. Specify inlet/outlet metal angles, half-turning location, and turning concentration; integrate the local slope to obtain the camberline.
2. Evaluate a Denton-based thickness distribution, modified by independent leading- and trailing-edge fullness controls.
3. Apply symmetric rounding families at LE and TE, followed by optional one-sided relief at either edge.
4. Apply normal offsets in the conformal plane, optionally corrected for the local physical thickness target.
5. Map the complete section onto the prescribed stream surface and place neighboring blades using the angular pitch.

There is no passage-first parametrization or automatic passage fitting. Passage crosscuts are optional postprocessing and do not modify the blade.

The rational thickness modification has global support. Negative fullness reduces the shoulder thickness contribution; positive fullness increases it. Both inputs equal to zero recover Denton exactly. Each input lies in $[-0.5,0.5]$, subject to a coupled admissibility condition that rejects some strongly opposed combinations. Independent inputs do not imply spatially isolated effects on the two halves of the section.

LE and TE use the same rounding law, with separately specified parameters. Matching edge settings supports symmetric impulse sections when the remaining camberline and thickness inputs are also symmetric. No automatic symmetry constraint is imposed.

The specified maximum thickness and station are nominal interpolation controls. Unequal endpoint thicknesses, rounding, relief, and variable-radius mapping can shift the actual maximum. The application reports the sampled physical maximum separately.

## Coordinate and physical-thickness conventions

Both camberline and thickness are defined using normalized conformal meridional position. The conformal metric preserves local angles and scales lengths by the local radius. On a cylindrical surface, multiplication by radius gives the developed physical section. On a radial surface, finite physical distances and curvature change under mapping.

Consequently, a fixed conformal thickness ratio does not generally prescribe a constant physical thickness ratio along a variable-radius section. The optional correction targets paired physical separation using a local metric factor; its finite-offset residual is measured explicitly. Paired separation is distinct from minimum surface-to-surface distance, a physical-normal thickness construction, and passage throat.

Uniform scaling of all physical machine dimensions produces uniform scaling of the mapped geometry. Scaling a section on a fixed variable-radius surface does not generally do so. Angular pitch remains determined by blade count.

## Development history and selected approach

| Stage | Main contribution | Subsequent change |
| --- | --- | --- |
| v0 | Examples using the existing TurboDash generator and discussion of alternative thickness constructions. | Established the original comparison baseline. |
| v1 | Independent edge/body thickness construction and a Denton comparison branch. | Explored additional shape freedom and joining conditions. |
| v2 | Conformal camberline integration with thickness still defined through physical-distance conventions. | Motivated a consistent complete-section construction in one plane. |
| v3 | Local leading-edge thickness redistribution. | Demonstrated case-specific curvature improvement without establishing robust general controls. |
| v4, renamed `parametrization_blades` | Complete conformal sections, symmetric edge treatment, relief, physical correction, application, mapping, and diagnostics. | Established the reusable implementation. |
| Current refinement | Global rational Denton correction with independent LE/TE fullness; shared implementation across app and studies. | Provides shoulder-shape control without additional local joining extents. |

Intermediate investigations considered wedge-angle controls with local blending. Their sensitivity to blending extent and localized inflections, particularly for reaction sections, motivated the global rational modification. The selected correction changes the concavity/convexity behavior of the thickness shoulders while retaining the established thickness level and nominal station.

## Application behavior and sensitivity evidence

The application supports precise numeric inputs, shape sliders, YAML persistence, independent scrolling of controls and results, four 2D views, and a 3D preview. Lightweight 2D geometry updates during slider motion; committed edits trigger the downstream 3D update. Display curves use approximately 600 base stations plus additional edge samples, with line simplification disabled.

The camberline study varies concentration at fixed location 0.5 for impulse sections. For reaction sections, it evaluates a location-by-concentration grid and corresponding overlays. Thickness parameters remain fixed within each family.

The thickness study evaluates fullness values -0.5, -0.25, 0, 0.25, and 0.5. The impulse sweep applies equal values at both ends; the reaction sweep varies only the TE input. Each figure compares unrolled thickness and the final section.

Axial passage crosscuts measure wall separation normal to a geometric equidistant center curve. They are not streamlines, guaranteed minimum-clearance measurements, or proof of constant flow area. Ambiguous and unconverged cuts are omitted and reported; radial crosscuts are not implemented.

Recorded validation before documentation consolidation included 189 passing automated tests, 40 passing sensitivity cases, and browser checks of numeric inputs, sliders, YAML persistence, preset switching, and 3D preview. Two opt-in mesh tests were skipped. The validation suite is no longer present in this directory, so these counts are historical evidence. The two sensitivity scripts remain available. Geometric validation does not establish aerodynamic performance or validity for every permitted input combination.

## Production integration and scope

The standalone implementation has not replaced TurboDash's existing production generator. Its equations are independent of Dash and TurboMoC; the adapter supplies downstream coordinate/unit conversion, CAD transfer, preview, and meshing interfaces.

Supported surfaces are axial cylinders and radial planes between parallel endwalls. General mixed-flow geometry, flaring, and arbitrary spanwise section variation require further implementation. Unsupported configurations must be identified explicitly during production integration.

The agreed integration workflow is:

1. A successful meanline calculation initializes each blade row from meanline dimensions and metal angles, supplemented by shape presets for parameters not provided by the meanline model.
2. Presets interpolate between reaction-like and impulse-like endpoints using the existing loss-model classification $z=-\beta_{in}/\beta_{out}$, after verifying its exact conventions. The proposed weight is $w=\operatorname{clip}(z,0,1)$. Undefined ratios require an explicit fallback.
3. The geometry pane edits one shared effective geometry state. Its changes update both the detailed geometry views and the meanline blade display without modifying or rerunning the thermodynamic solution.
4. A subsequent successful meanline calculation replaces the geometry baseline and clears overrides. Failed calculations preserve the previous valid geometry; tab changes and unrelated callbacks preserve edits.
5. The modeller follows the optimization app's existing layout and style. Production code and development studies import one geometry implementation.

The current standalone state model already separates inherited values from explicit overrides. Its **Refresh source** action preserves overrides; the planned reset after a successful meanline solve is a distinct application-level operation.

The superseded production generator should move into `legacy/` outside the active geometry subpackage. The integration should establish one authoritative geometry state across meanline visualization, detailed modelling, and CFD preparation.
