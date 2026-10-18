# Turbodash organization proposal

**Status: proposal for review; no structural refactor implemented.**

Prepared from the repository review and test baseline on 17 October 2026.
Please add comments or edit the decisions section before implementation.
The numerical findings in [Arianna's troubleshooting folder](troubleshooting/arianna_meanline/README.md)
are separate work; their proposed corrections are excluded from this refactor.

## Recommendation

Organize the existing package by responsibility: calculations, geometry,
meshing integration, visualization, reporting, external interfaces, shared
numerical utilities, and applications. Keep the current top-level `turbodash/`
installation layout. Move existing implementations and update imports and
resource references, with compatibility paths for existing users.

The greatest benefit is making ownership clear, rather than introducing a new
application framework or redesigning the numerical models. Keep the stable and
experimental turbine applications independent while the latter is developed.

## Assessment and the TurboMoC comparison

Turbodash already separates many responsibilities into individual files, but
nearly all are peers at the package root. Names such as `app`, `graphics`,
`plotting_mpl`, `geom_blade`, and `reporting_utils` do not make their relationships
obvious. Some modules also span several responsibilities:

- `core_stage.py` includes calculations, console reporting, table construction,
  and Word report generation.
- `turbogrid_assembly.py` includes assembly geometry, a Matplotlib plot, and a CLI.
- Assembly geometry imports section-construction helpers from a TurboGrid
  exporter, putting reusable geometry below an external file-format dependency.
- `app_meshing.py` and `app_turbogrid.py` sound like launchable applications but
  actually provide panels and callback registration.
- Package-root wildcard exports and module aliases form an existing public API
  used by demos and applications; moves must preserve those call paths.

The reviewed TurboMoC checkout is `../method_of_characteristics`, revision
`7587fe1`. Its `src/turbo_moc/` package has identifiable owners including `core`,
`geometry`, `meshing`, `visualization`, `io`, `configuration`, `app`, and `legacy`.
Its app README explains local responsibilities, resources, and launch commands.

Patterns worth adopting are responsibility-based subpackages, an applications
README, explicit launch targets, and attention to installed resources and worker
module paths. Keep TurboMoC responsible for its meshing and rendering algorithms;
Turbodash should own its adapter from solved turbine results.

Do not copy the whole hierarchy. Turbodash does not currently need TurboMoC's
specialized `fluent`, `configuration`, or `workflows` packages. TurboMoC also has
cross-package dependencies, including geometry-to-visualization and current-to-legacy
imports; its folder names are not proof of a strictly layered architecture.
It has named support modules rather than a general `utils` package. Turbodash's
shared utilities should likewise have specific responsibilities.

## Proposed structure

Each new package would have an `__init__.py`; only meaningful exports belong in
those files. Compatibility forwarding modules are omitted from this tree.

```text
turbodash/
    __init__.py                  # existing public API and package metadata
    core/
        stage.py                # existing single-stage model
        turbine.py              # existing multistage driver and stage model
        loss_models/            # existing correlation filenames and functions
    geometry/
        blade.py                # blade coordinates and parametrizations
        sections.py             # existing spanwise section construction
        assembly.py             # positions, interfaces, endwalls, row envelopes
    meshing/
        turbomoc.py             # existing optional TurboMoC adapter
    visualization/
        matplotlib_style.py     # styling, figure saving, animation, scaling
        matplotlib_turbine.py   # turbine figures
        matplotlib_assembly.py  # assembly preview figure
        plotly_stage.py
        plotly_turbine.py
        plotly_mesh.py
        vtk.py                  # reusable payloads, controls and callbacks
        vtk_assets.py           # installed TurboMoC renderer bridge
    reporting/
        stage.py                # existing stage printers, tables, Word report
        turbine.py              # existing reporting_utils implementations
    io/
        turbogrid.py
        turbogrid_assembly.py
        excel_stage.py
        excel_turbine.py
    utils/
        math.py                 # existing shared numerical helpers
    apps/
        README.md
        turbine.py              # stable meanline application
        turbine_experimental.py # meanline, geometry and mesh workspaces
        stage.py                # original single-stage application
        stage_efficiency.py     # dimensionless efficiency exploration
        assembly_preview.py     # existing noninteractive assembly-preview CLI
        launchers.py            # existing launch_app_* functions
        mesh_jobs.py            # dashboard mesh-job lifecycle and worker
        components/
            meshing.py
            turbogrid.py
    assets/                     # keep existing resource locations initially
    docs/
```

`tests/`, `demos/`, `docs/`, and `troubleshooting/` remain at the repository root.
Do not simultaneously reorganize the parametrization research prototypes or
convert the project to a `src/` layout.

`io/` groups the existing external-format and Excel adapters without adding
separate packages for each small integration. An alternative is `exports/` plus
`interfaces/`; that is semantically narrower but adds hierarchy with little
immediate benefit. My recommendation is `io/` for this codebase's current size.

The mesh-job service belongs with applications because it handles dashboard
job admission, subprocesses, polling, cancellation and artifact access. The
meshing adapter remains reusable independently of that service. Visualization
includes the current reusable Dash/VTK bridge; moving it does not require
rewriting it into separate frontend and numerical layers.

## Main moves and selective function relocations

| Current source | Proposed destination or treatment |
| --- | --- |
| `core_stage.py`: calculations through `compute_blade_row_geometry` | `core/stage.py` |
| `core_stage.py`: formatting/reporting block starting at `_fmt`, including printers, tables and `generate_meanline_report` | Move together to `reporting/stage.py`, with their existing dependencies. |
| `core_turbine.py` | `core/turbine.py`; preserve its implementations. |
| `loss_models/` | `core/loss_models/`; preserve correlation module and function names. |
| `geom_blade.py` | `geometry/blade.py`; keep this implementation together initially. |
| `export_turbogrid.py`: `_positive`, `_station_geometry`, `_make_section` | `geometry/sections.py`; reusable section construction belongs below both exporters and assembly geometry. |
| Remaining `export_turbogrid.py` | `io/turbogrid.py`; profile/endwall text serialization stays here. |
| `export_turbogrid_assembly.py` | `io/turbogrid_assembly.py` |
| `turbogrid_assembly.py`: `build_assembly_layout` | `geometry/assembly.py` |
| `turbogrid_assembly.py`: `plot_assembly_layout` | `visualization/matplotlib_assembly.py` |
| `turbogrid_assembly.py`: `main` | `apps/assembly_preview.py`; retain the old module CLI through compatibility forwarding. |
| `meshing.py` | `meshing/turbomoc.py`, with the existing callable API exposed from the package. |
| `mesh_jobs.py` | `apps/mesh_jobs.py` |
| `graphics.py` | `visualization/matplotlib_style.py` |
| `plotting_mpl.py` | `visualization/matplotlib_turbine.py` |
| `plotting_plotly_stage.py`, `plotting_plotly_turbine.py` | `visualization/plotly_stage.py`, `visualization/plotly_turbine.py` |
| `plotting_mesh.py` | `visualization/plotly_mesh.py` |
| `visualization_vtk.py`, `vtk_assets.py` | `visualization/vtk.py`, `visualization/vtk_assets.py` |
| `reporting_utils.py` | `reporting/turbine.py`, including its current dictionary-formatting helper. |
| `excel_interface_stage.py`, `excel_interface_turbine.py` | `io/excel_stage.py`, `io/excel_turbine.py` |
| `math.py` | `utils/math.py` |
| Application and panel modules | The application mapping below. |
| `launch_app_stage`, `launch_app_turbine`, `launch_app_experimental` | `apps/launchers.py`, re-exported from the package root with their current names and behavior. |

Keep the two existing meanline models separate. Their same-named geometry-sizing
functions have different APIs, formulas and output schemas. Merging them would
be a functional redesign. Their sizing routines can remain in their respective
core modules; `geometry/` initially owns coordinate, section and assembly geometry.

Likewise, keep `graphics.py` functionality separate from turbine-specific plots.
Do not move every generic-looking helper into `utils/`; the reporting formatting
helpers belong with reporting and the blade-profile helpers belong with geometry.

## Applications, relationships and naming

Use `turbodash.apps.<purpose>[_<variant>]` for standalone application modules.
The `apps` namespace makes another `app_` prefix unnecessary. Use
`turbodash-<purpose>[-<variant>]` for explicit console commands, while preserving
the existing shorter commands as aliases to the same applications.

| Current module | Proposed module | Purpose and relationship | Proposed explicit command |
| --- | --- | --- | --- |
| `app_turbine.py` | `apps.turbine` | Stable multistage meanline application; remains the default. | `turbodash-turbine`, with existing `turbodash` retained. |
| `app_experimental.py` | `apps.turbine_experimental` | Independent extension of the turbine app with Meanline, Turbine geometry and Mesh workspaces. | `turbodash-turbine-experimental`, with existing `turbodash-experimental` retained. |
| `app.py` | `apps.stage` | Original dimensional single-stage calculator using `core_stage`; a different model, not merely the turbine app with one stage. | Existing `turbodash-stage`. |
| `app_lite.py` | `apps.stage_efficiency` | Dimensionless efficiency exploration versus velocity ratio and reaction, using `compute_performance_stage`; no current console command. | Add `turbodash-stage-efficiency` pointing to its existing `main`. |
| `app_meshing.py` | `apps.components.meshing` | Geometry/mesh panels and callback registration; not a standalone application. | None. |
| `app_turbogrid.py` | `apps.components.turbogrid` | Reusable TurboGrid export panel and callback registration. | None. |

The stable turbine app should not be renamed `legacy` while the experimental
app is being refined. Do not delete or deprecate the stage and efficiency apps
without deciding their intended support status. The Excel interfaces are host
application adapters; demo `run_app` scripts are launch shortcuts, not additional
versions of the dashboard.

The new explicit console aliases are additive naming changes. A more conservative
alternative is to keep only the three current commands and introduce a command
only for the previously unadvertised efficiency app. I recommend the explicit
aliases for discoverability, while retaining all existing commands.

### Planned short `turbodash/apps/README.md`

The applications README would contain:

- A short version of the table above, explaining which app to use and its status.
- Existing Python launchers and both explicit and compatibility console commands.
- Stable deployment target `turbodash.apps.turbine:server` and the separate
  experimental target `turbodash.apps.turbine_experimental:server`.
- The optional `mesh` dependency for experimental geometry/meshing, and a clear
  distinction between standalone apps, reusable components and mesh workers.
- The assembly-preview CLI and a note that demo launch scripts delegate here.

Preserve current startup behavior: console `main()` uses `debug=False`, while
the existing package launch helpers use `debug=True` and schedule browser opening
only in the Werkzeug reloader child. Do not introduce an app factory, consolidate
variant callbacks, change the default port, or add new command-line options.
The original stage/efficiency modules do not currently start a server under
`python -m`; do not advertise that behavior without a separately agreed change.

## Compatibility and behavior-preservation requirements

Keep function signatures, bodies, decorators, defaults, result keys, equations,
units, error behavior and UI behavior intact, apart from import/reference updates
necessary for relocation. Move complete dependency groups rather than extracting
functions whose existing globals would change meaning.

Specifically:

1. Preserve root-level calculation/reporting/graphics exports and aliases such
   as `td.core_turbine`, `td.geometry`, `td.mpl`, `td.plotly`, and
   `td.reporting_utils`. Inventory the current wildcard exports before replacing
   any of them; narrowing the public namespace is outside this refactor.
2. Keep thin compatibility modules for old dotted imports and deployment paths
   for a transition period. Internal code should import canonical owners directly.
   Do not maintain duplicate implementations or construct another Dash app in
   a compatibility module.
3. Module-global state matters. Current tests patch `ctx`, mesh-job globals,
   caches, and adapter helpers. A plain `import *` wrapper omits private names
   and does not forward writes to implementation globals. Preserve single-module
   ownership with intentional aliases where possible, and test legacy paths.
   `geometry` and `meshing` becoming real packages require explicit facades;
   existing callable APIs must continue to work. Tests patching implementation
   globals should address their canonical module rather than an accidental copy.
4. Keep stable and experimental app objects, servers and callback maps separate.
   Although many helper functions are identical, they depend on variant globals
   and decorators. Whole-module moves are safer than sharing those callbacks.
5. Preserve effective documentation and asset paths. Moving Dash modules changes
   implicit asset discovery. Keep current package resources in place initially
   and reference them explicitly. Verify wheel inclusion and HTTP asset routes.
   The original stage apps currently fall back when looking for a missing
   repository-level documentation file; correcting that existing behavior is
   a separate functional change.
6. Update the mesh worker's `python -m` target and its `__file__`-derived working
   directory together. Preserve cache paths, artifact access, worker lifetime,
   progress/status contracts and the `/_turbodash_vtk/` renderer URLs.
7. Preserve Excel UDF names, including the existing `meanline_caculator_full`
   spelling used in workbook formulas. Relative configuration lookup currently
   resolves against the package root and must not shift into `io/`. Preserve old
   import paths rather than modifying binary workbooks during this refactor.
8. Preserve the package's current import-time exception-hook behavior. Changing
   it or introducing a new lazy-loading policy is separate maintenance work.

The temporary compatibility layer leaves more files at the root. That is a
reasonable migration cost; deleting it immediately would make the tree cleaner
while breaking existing scripts. Removing it should be an explicit later decision.

## Test baseline and verification plan

The baseline includes the earlier stable/experimental split already present in
the working tree; it is not a clean checkout of HEAD `eea5aa9`. No code was moved
for this structural proposal.

| Check | Before refactor | After refactor |
| --- | --- | --- |
| Full existing pytest suite, 125 collected | **117 passed, 8 skipped**, exit 0, 8.25 seconds | Pending implementation. |
| Pre-existing pytest failures in the completed run | None. | Report any new failures separately. |
| Optional geometry/meshing integration cases | Six skipped because `gmsh` was not installed; `pyvista` also absent. | Establish matching baseline with the pinned optional stack before claiming coverage. |
| Optional VTK asset case | One skipped because `dash_vtk` was absent. | Run with matching dependencies when available. |
| Windows symlink case | One skipped because symlink creation lacked privileges. | Track platform limitation separately. |

The initial sandboxed attempts hit 44 temporary-directory fixture setup errors;
the permitted run resolved those environment errors without changing test code.
These were not demonstrated calculation failures.

The successful baseline used Python 3.13.2 and pytest 8.4.2 from:

```text
C:/Users/roagr/AppData/Local/pypoetry/Cache/virtualenvs/
  turbomachinery-special-course-fbPpiFVa-py3.13/Scripts/python.exe
```

Its other recorded versions include NumPy 2.5.2, SciPy 1.18.1, JAX 0.6.2,
JAXProp 0.5.6, Dash 3.4.0, and Plotly 5.24.1. This existing environment is not a
fresh lockfile recreation. Its installed TurboMoC is 1.0.1, whereas Turbodash's
optional integration pins revision `65976c6`; do not upgrade that pin during
the refactor. The raw local baseline log is `refactor_baseline_authorized.log`
(ignored by Git); the durable result is recorded here.

The baseline command, from Turbodash, was:

```powershell
python -m pytest -q -ra --tb=short --basetemp=.pytest_cache/refactor-baseline-20261017-authorized
```

Use the recorded interpreter and a fresh task-specific temporary directory when
repeating it. Keep before/after dependency versions and input data the same.

Passing the existing suite is necessary but insufficient: `tests/test_core.py`
is entirely commented out, and the CI pytest steps are also commented out.
The active suite mostly covers meshing/UI infrastructure. Before approved moves:

- Save representative axial/radial single-stage and multistage results, loss-model
  outputs, blade coordinates, and reporting tables from the unchanged code.
- Retain Arianna's independently reproduced outputs as useful additional parity
  evidence, including their currently invalid parameter regions; do not change
  those numerical results while calling the change structural.
- Capture TurboGrid ZIP member names and contents, assembly layout data, and
  report content. Compare archive members rather than timestamp-dependent ZIP bytes.

After each coherent move group, run the relevant existing tests and parity checks;
then run the full suite after the complete refactor. Also verify old and new
imports, app layout/dependency endpoints, stable startup without optional mesh
imports, launch helpers, deployment targets, worker execution, Excel configuration
lookup, and resources in an installed wheel launched outside the checkout.
Report unchanged failures/skips separately from newly introduced failures or
new skips. Do not weaken existing numerical assertions to make a move pass.

## Implementation sequence after feedback

1. Agree on the structure, names, compatibility period and selective extractions.
2. Freeze the working-tree baseline and capture the missing numerical/export
   parity cases before changing implementation locations.
3. Move shared scientific code and the limited reporting/section/assembly
   function groups, updating imports and compatibility paths.
4. Move visualization and external interfaces, preserving resource/config paths.
5. Move applications, components, launch helpers and workers; update packaging,
   deployment references, docs and demos; add the applications README.
6. Complete the full before/after comparison and report regressions separately.

Keep dependency updates, API redesigns, broad formatting, app deduplication and
numerical fixes out of these steps so the structural diff remains reviewable.

## Decisions for your feedback

| Decision | Recommendation and tradeoff | Your comments |
| --- | --- | --- |
| Package layout | Adopt the responsibility-based structure above and retain the current installation layout. A simultaneous `src/` migration improves installation isolation but broadens scope. | |
| Application names | Use `apps.turbine`, `apps.turbine_experimental`, `apps.stage`, `apps.stage_efficiency`; add explicit purpose-based commands while preserving current commands. Keeping only existing CLI names is a smaller alternative. | |
| Stage/efficiency app status | Retain and document both as distinct tools. Move them under `legacy` only if you decide they are historical and no longer maintained. | |
| Compatibility | Preserve old imports and commands for at least one transition period. Removing forwarders immediately reduces clutter but breaks scripts and workbook integrations. | |
| Extraction depth | Move whole modules except the specific reporting, section-construction and assembly responsibilities described above. Sharing stable/experimental app internals can wait. | |
| External integrations | Use one `io/` package now. Choose separate `exports/` and `interfaces/` only if you prefer the sharper distinction despite the extra hierarchy. | |
| Numerical investigation | Resolve or explicitly defer its functional recommendations separately; do not incorporate the continuous-count or momentum-sizing candidate into this refactor. | |

## Functional follow-ups explicitly excluded

The [meanline troubleshooting findings](troubleshooting/arianna_meanline/FINDINGS.md)
identify blade-count quantization, unequal-meridional-velocity Zweifel sizing,
loss-correlation behavior and invalid-domain reporting. They require numerical
decisions and validation rather than file moves.

Other separate follow-ups include reconciling the two meanline model APIs,
correcting stale demos/documentation, completing radial plotting placeholders,
repairing the `IMEGEIO_AVAILABLE` spelling inconsistency, reviewing package import
side effects, and restoring useful core/CI test coverage. None should be silently
fixed as part of the behavior-preserving structural change.
