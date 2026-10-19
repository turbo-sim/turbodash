# Applications

Each app package owns its Dash app, server and callbacks in `app.py`. The stable and experimental
turbine applications remain independent.

```text
apps/
    turbine/                 # Stable meanline app
    turbine_experimental/    # Geometry/mesh workspace and assembly CLI
    turbine_optimization/    # Optimization app, panel, model, jobs, assets
    stage/                   # Constant-loss stage calculator
    stage_efficiency/        # Dimensionless efficiency explorer
    shared/                  # Startup, launchers, mesh jobs, panels, assets
```

Each app folder has `app.py`, `__init__.py`, and `__main__.py`. Existing console
commands and `python -m turbodash.apps.<name>` launch commands continue to work.
For Python integrations, use `from turbodash.apps.<name> import app as application`;
for WSGI deployments, target `turbodash.apps.<name>.app:server`.


| Application | Module | Existing launch command | YAML demo |
| --- | --- | --- | --- |
| Turbine meanline, stable/default | `turbodash.apps.turbine` | `turbodash` | [Runner](../../demos/apps/turbine/run.py), [case](../../demos/apps/turbine/case.yaml) |
| Turbine geometry and meshing workspace | `turbodash.apps.turbine_experimental` | `turbodash-experimental` | [Runner](../../demos/apps/turbine_experimental/run.py), [case](../../demos/apps/turbine_experimental/case.yaml) |
| Experimental workspace with optimization | `turbodash.apps.turbine_optimization` | `turbodash-optimization` | Uses the current design or an uploaded YAML case |
| Original constant-loss stage calculator | `turbodash.apps.stage` | `turbodash-stage` | [Runner](../../demos/apps/stage/run.py), [case](../../demos/apps/stage/case.yaml) |
| Dimensionless stage efficiency explorer | `turbodash.apps.stage_efficiency` | Use its demo runner or `main()` | [Runner](../../demos/apps/stage_efficiency/run.py), [case](../../demos/apps/stage_efficiency/case.yaml) |

The existing `turbodash.launch_app_turbine()`, `launch_app_experimental()` and
`launch_app_stage()` helpers still open the selected app with their original
browser/reloader behavior. The console commands run with `debug=False`.

Each application also accepts an optional startup case:

```python
from turbodash.apps.turbine import app as turbine
turbine.main("case.yaml")
```

`configure("case.yaml")` applies the YAML to the initial controls and stage
stores without starting the server. Call it before serving requests. It does
not replace the app's calculations or callbacks. Dimensional apps retain their
existing `inputs` schemas; the efficiency demo documents its displayed parameters.

Run `python -m pip install -e .` after updating a checkout to register the
canonical console targets. Stable deployment uses
`gunicorn turbodash.apps.turbine.app:server`; experimental deployment uses
`gunicorn turbodash.apps.turbine_experimental.app:server` and the optional `mesh`
extra. Stable startup does not import the optional meshing/VTK stack.

`shared/components/` contains reusable meshing and export panels and callback
registration. `shared/mesh_jobs.py` manages isolated worker processes and artifacts;
`python -m turbodash.apps.shared.mesh_jobs REQUEST_JSON` is its worker entry point.
`turbine_experimental/assembly_preview.py` provides the assembly CLI.
`shared/startup.py`, `shared/launchers.py`, `shared/paths.py`, and
`shared/assets/` provide common configuration loading, launch helpers, resource
paths, and styling. Version-specific resources stay with their app. The former
flat implementation modules and the old `components/` folder have been removed.

## Optimization edition

Install `python -m pip install -e ".[optimization]"`, then run
`python -m turbodash.apps.turbine_optimization` or `turbodash-optimization`.
It is a separate app based on the experimental workspace. Both earlier app
entrypoints retain their behavior. Optional meshing still uses the `mesh` extra.

Expand **Optimization** at the bottom of the left sidebar, below the overall
parameters and every stage's initial inputs. Settings select a scalar objective,
maximize/minimize, solver tolerance, iteration limit, and feasibility tolerance.
Add inputs under **Design variables**, then set physical lower/upper bounds.
Leave Initial blank to use the current design, or enter an explicit override;
out-of-bounds guesses are clipped and reported. Every unselected input stays
fixed. Choices include every stage and blade row. Radial aspect ratios and
derived metal angles are excluded. Speed varies in specific-speed units or rpm
according to its fixed selected type.

Under **Constraints**, select a result, comparison, target, and positive scale.
All-stage/both-row selections create one scalar constraint per selected output.
For example, both-row flaring equal to zero with scale 10 is supported; it is
optional. Scale divides the physical residual; feasibility tolerance applies to
the resulting dimensionless violation. Efficiencies are fractions, dimensions
are metres, and angles are degrees. Multiple scalar equalities require at least
as many independent design variables.

**Start optimization** uses the live design. The sidebar stays busy while a
subprocess solves and while the successful design is applied to all controls
and plots. Other panes remain interactive. The expandable **Optimization
terminal** below the meanline results shows the native PSV report, clipping,
physical progress, evaluation counts, and completion/errors. PSV's Func-value
is signed and normalized for minimization; the progress summary shows the
physical objective. The terminal follows new output until you scroll up.

Only converged, feasible results update the inputs, geometry, performance,
tables, and downstream geometry/mesh signatures. Failed solves retain the last
displayed design. Normal manual editing remains available afterward. Loading
YAML also restores its optimization editors and bounds; both flat and older
nested speed schemas are accepted. Saving a successfully optimized design
includes its optimization settings without reapplying old initial overrides.

`turbine_optimization/model.py` owns editor choices and validation;
`turbine_optimization/panel.py` owns the panel/callbacks;
`turbine_optimization/jobs.py` owns isolated processes, session ownership,
progress, and logs. Its CSS/JavaScript live in `turbine_optimization/assets/` and
are loaded only by this version. Numerical code lives in the
`turbodash.optimization` package, independently of Dash. Its optional `logger`
receives clipping and native PSV output without redirecting server stdout.
Public callback snapshots remain in physical units and include the matching
configuration and turbine results for future live geometry rendering.

Jobs use a local temporary cache, signed session cookies, atomic progress files,
a 10 minute worker limit, and at most two concurrent solves per host. Completed
jobs expire after 24 hours. Set `TURBODASH_SECRET_KEY` for a deployment-managed
signing key. Multiple web workers on one machine share the cache; deployment
across separate hosts needs a shared job backend. Progress is currently shown
in the status/terminal; design plots update when the result is accepted.

Validation commands:

```powershell
python -m unittest development.optimization.test_optimization -v
python -m pytest tests/test_app_optimization.py tests/test_app_entrypoints.py tests/test_app_meshing.py tests/test_mesh_jobs.py tests/test_refactored_layout.py -q
python tests/browser_optimization.py
```

The optional browser check needs Playwright plus Chromium (`python -m pip
install playwright` and `python -m playwright install chromium`). It starts and
stops its own local app server and writes screenshots and a JSON report.
