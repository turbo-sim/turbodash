# Package refactoring verification

The organization refactor uses commit `8a1972f` as its numerical/export baseline.
Implementations live under `core/`, `geometry/`, `meshing/`, `visualization/`,
`reporting/`, `io/` and `apps/`. The user approved removing module compatibility
on 18 October 2026. The package root now contains only `__init__.py` and `math.py`.
The 23 forwarding modules and `_compat.py` have been deleted; geometry and meshing
are ordinary packages. See the [refactor evaluation and migration map](../REFACTOR_EVALUATION.md).

Root scientific functions, plotting-style functions and Python app launchers
remain direct exports. Flat module imports and root module aliases such as
`td.core_turbine`, `td.mpl` and `td.plotly` have been removed. Repository tests,
demos, analysis scripts and documentation import the owning modules directly.
External scripts and Excel module configuration must use those current paths.

## Automated checks

| Check | Result |
| --- | --- |
| Production suite before the organization refactor | 399 passed, 8 skipped. |
| Production suite after relocation with compatibility checks | 425 passed, 8 skipped. |
| Production suite after removing compatibility | 411 passed, 8 skipped. |
| Removed tests | 14 checks of old aliases, forwarding and duplicate worker/import paths. |
| Axial/radial stage calculations, report tables and console text | Match captured pre-refactor results. |
| Loss models, math and commented `tests/test_core.py` | File hashes unchanged. |
| Four YAML startup cases | Layout, calculation HTTP callbacks and adjacent-file demo runners pass. |
| Meshing worker | Canonical module command runs from outside the checkout; job lifecycle checks pass. |
| Built wheel | No removed compatibility files; four apps, CSS, documentation, console targets and Excel configuration lookup pass outside the checkout. |

The command was `python -B -m pytest tests -q -ra`, with a fresh temporary
directory. The sandboxed attempt encountered Windows access-denied errors
creating pytest's temporary directory. The permitted rerun passed without
changing test assertions. The eight skips match the baseline: six native meshing
cases require Gmsh, one renderer case requires Dash VTK, and the Windows symlink
case requires symlink privileges. Excel-host integration is not exercised here.

## Real browser checks

Headless Microsoft Edge loaded each application's YAML case from the rebuilt
wheel, served outside the source checkout. Layouts and calculation requests
completed, plots rendered, and no JavaScript exceptions or Dash error cards were
detected. The experimental workspace's tabs were also exercised.

| Application | Rendered plots | Browser errors |
| --- | ---: | ---: |
| Stable turbine | 5 | 0 |
| Experimental turbine | 8 | 0 |
| Original stage | 4 | 0 |
| Dimensionless efficiency explorer | 2 | 0 |

Use the [four app runners](../demos/apps/README.md) to reproduce these cases.
Each resolves `case.yaml` beside its own file. Startup configuration initializes
controls without replacing equations or calculation callbacks.

## Scientific analyses

KO, Benner and isentropic stage source guards and the KO shock source guard pass
against the unchanged numerical implementation. Individual-function evidence
and interpolation source inventories have been refreshed through their supported
calculation/validation workflow after deleting the forwarding files. The AM fit
figures and provenance use the current plotting-style import. Numerical tables
are compared with their pre-cleanup contents, including signed and invalid
samples; all 158 retain identical decompressed contents.

The analysis contains 214 SVG figures embedded in 27 Markdown reports, including
the three-column stage comparison. Original paper PDFs remain digitization
inputs. Source guards, plot hashes and Markdown equation/link checks remain in
use. No analysis PNGs or ZIP backups are added.

## Retained boundaries

The two meanline APIs and all four applications remain distinct. The stable app
starts without importing the optional experimental meshing/VTK stack. Physical
consumers validate annulus geometry and realize integer blade counts; meanline
studies retain signed geometry and efficiency. Existing stage-app documentation
fallbacks, workbook UDF names, worker
policies, console debug behavior and browser-opening launchers remain unchanged.

For release 0.7.0, the optional TurboMoC dependency was changed to published
version 1.2.0 and the adapter was checked against its renderer and mesher.
The [integration guide](turbomoc_meshing.md) records the updated native mesh
checks and the full result of 418 passed tests with one symlink-permission skip.
