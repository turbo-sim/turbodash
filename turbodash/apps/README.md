# Applications

Each app package owns its Dash app, server and callbacks in `app.py`. The stable and experimental
turbine applications remain independent.

```text
apps/
    app_turbine/                 # Stable meanline app
    app_turbine_experimental/    # Geometry/mesh workspace and assembly CLI
    app_turbine_optimization/    # Optimization app, panel, model, jobs, assets
    app_stage/                   # Constant-loss stage calculator
    app_stage_efficiency/        # Dimensionless efficiency explorer
    shared/                      # Startup, launchers, mesh jobs, panels, assets
```

Each app folder has an `app_` prefix and contains `app.py`, `__init__.py`, and
`__main__.py`. Console command names stay the same; module launches use
`python -m turbodash.apps.app_<name>`. For Python integrations, use
`from turbodash.apps.app_<name> import app as application`; for WSGI deployments,
target `turbodash.apps.app_<name>.app:server`.


| Application | Module | Launch command | YAML demo |
| --- | --- | --- | --- |
| Turbine meanline, stable/default | `turbodash.apps.app_turbine` | `app_turbine` or `turbodash` | [Runner](../../demos/apps/turbine/run.py), [case](../../demos/apps/turbine/case.yaml) |
| Turbine geometry and meshing workspace | `turbodash.apps.app_turbine_experimental` | `app_turbine_experimental` or `turbodash-experimental` | [Runner](../../demos/apps/turbine_experimental/run.py), [case](../../demos/apps/turbine_experimental/case.yaml) |
| Experimental workspace with optimization | `turbodash.apps.app_turbine_optimization` | `app_turbine_optimization` or `turbodash-optimization` | Uses the current design or an uploaded YAML case |
| Original constant-loss stage calculator | `turbodash.apps.app_stage` | `app_stage` or `turbodash-stage` | [Runner](../../demos/apps/stage/run.py), [case](../../demos/apps/stage/case.yaml) |
| Dimensionless stage efficiency explorer | `turbodash.apps.app_stage_efficiency` | `app_stage_efficiency` | [Runner](../../demos/apps/stage_efficiency/run.py), [case](../../demos/apps/stage_efficiency/case.yaml) |

The existing `turbodash.launch_app_turbine()`, `launch_app_experimental()` and
`launch_app_stage()` helpers still open the selected app with their original
browser/reloader behavior. The console commands run with `debug=False`.

All five apps support an optional `--open-browser` flag. After registering the
commands with `poetry install --only-root`, run any of these from the repository
root:

```powershell
poetry run app_turbine --open-browser
poetry run app_turbine_experimental --open-browser
poetry run app_turbine_optimization --open-browser
poetry run app_stage --open-browser
poetry run app_stage_efficiency --open-browser
```

Omit the flag to start the server without opening a browser. The flag also works
with the existing `turbodash*` commands and `python -m turbodash.apps.app_<name>`.
It opens one browser tab after the app's layout endpoint responds successfully.
The browser URL follows the server's `HOST` and `PORT` environment settings,
using a local address when the server binds to all interfaces. Use `--help` to
see the launch options. If no browser is available, the server still runs and
logs its URL for manual opening.

Each application also accepts an optional startup case:

```python
from turbodash.apps.app_turbine import app as turbine
turbine.main("case.yaml")
# Or open a browser after starting the configured app:
turbine.main("case.yaml", open_browser=True)
```

`configure("case.yaml")` applies the YAML to the initial controls and stage
stores without starting the server. Call it before serving requests. It does
not replace the app's calculations or callbacks. Dimensional apps retain their
existing `inputs` schemas; the efficiency demo documents its displayed parameters.

Run `python -m pip install -e .` after updating a checkout to register the
canonical console targets. Stable deployment uses
`gunicorn turbodash.apps.app_turbine.app:server`; experimental deployment uses
`gunicorn turbodash.apps.app_turbine_experimental.app:server` and the optional `mesh`
extra. Stable startup does not import the optional meshing/VTK stack.

`shared/components/` contains reusable meshing and export panels and callback
registration. `shared/mesh_jobs.py` manages isolated worker processes and artifacts;
`python -m turbodash.apps.shared.mesh_jobs REQUEST_JSON` is its worker entry point.
`app_turbine_experimental/assembly_preview.py` provides the assembly CLI.
`shared/startup.py`, `shared/launchers.py`, `shared/paths.py`, and
`shared/assets/` provide common configuration loading, launch helpers, resource
paths, and styling. Version-specific resources stay with their app. The former
flat implementation modules and the old `components/` folder have been removed.
`shared/cli.py` owns the common command-line options and optional browser opening.

## Optimization edition

From the repository root, install the optimization dependencies and register the
launch commands once:

```powershell
poetry install -E optimization
```

Then start the app with:

```powershell
poetry run app_turbine_optimization
```

If dependencies are already installed, `poetry install --only-root` registers
the new command. The existing `poetry run turbodash-optimization` and
`poetry run python -m turbodash.apps.app_turbine_optimization` also work.
For a Linux WSGI deployment, the corresponding target is
`gunicorn turbodash.apps.app_turbine_optimization.app:server`.
It is a separate app based on the experimental workspace. Both earlier app
entrypoints retain their behavior. Optional meshing still uses the `mesh` extra.

The optimization edition uses a light Blueprint / dark Graphite workbench.
The first visit follows your operating-system theme; the switch at the far right
remembers an explicit choice. The left rail switches between **Meanline**,
**Geometry**, and **Mesh** without resetting inputs or jobs. Clicking the brand
returns to Meanline. Load YAML and Save design remain in the header.

Geometry is a blade modeller using the production conformal camberline and
rational Denton thickness. Its per-row baseline comes from each successful
meanline result plus interpolated reaction/impulse presets. Geometry controls
edit sparse overrides shared by the blade views and CAD/mesh exporters;
they never rerun meanline. A successful new calculation clears those overrides,
while failed calculations, tab changes and optimizer progress retain them.
Use the row YAML controls to save/load geometry separately from the header's
meanline design file. See the [blade geometry guide](../../docs/source/guides/blade_geometry.md)
for mapping, presets, state ownership, launch instructions and limitations.

Five numbered sections start collapsed: **Turbine setup**, **Operating point**,
**Shaft speed**, **Stage design**, and **Optimization**. The stage
selector uses saved names; Stator/Rotor switches the visible blade fields while
all stage inputs remain mounted. Continuous inputs accept full floating-point
precision and display at least three decimal places, preserving extra supplied
digits. Formatting does not round stored values or interrupt typing; discrete
counts stay whole numbers. Find a parameter (Ctrl/Cmd+K) searches names and aliases within the current Meanline, Geometry, or Mesh workspace, opens its
section, selects its stage/row, and focuses the field. The inspector can be resized
on desktop and opened as a drawer on narrow screens.

Fluid controls use readable property names and units. Changing an inlet pair
converts the last valid state atomically; an impossible conversion clears both
new values and requests fresh input. Invalid drafts retain the last successful
results and prevent saving a stale design. Save design includes the current
optimization editor, including unsolved settings and precise bounds.

The overview reports total-to-static/total-to-total efficiency, shaft speed in
rpm, and adaptive W/kW/MW power. Detailed reports retain all rows and offer CSV
exports. Meridional and blade-to-blade views appear side by side on desktop;
reset/autoscale restores a zero radial baseline while deliberate zoom is allowed.
The channel has upstream/downstream display extensions and smooth guide
streamlines constrained within the hub/shroud passage. These illustrate the
channel shape, rather than a computed CFD velocity field. Stator/rotor profiles
use orange/blue outlines with matching translucent fills.

Expand **Optimization** at the bottom of the left sidebar. **Configuration**
selects an objective, maximize/minimize, iteration limit, and solver tolerance.
**Show expert settings** reveals the independent feasibility and optimality
limits, grouped with the solver tolerance in scientific notation. It also shows
first step fraction (default `0.25`), step growth factor (`1.5`), step reduction
factor, and trial retry limit inline. Hiding expert settings preserves their
values. Invalid-trial recovery is always enabled in the app; maximum step
fraction and derivative step remain configurable through YAML/Python. The driver
uses one bound-aware derivative probe per variable, without derivative retries.
See the [SLSQP guide](../../docs/source/guides/optimization.md) for details.

Hover or keyboard-focus a tab for 750 ms to see its purpose. The variable and
constraint tabs place their Add buttons directly below the tabs. Picker branches
start expanded, with indented checkboxes and connecting tree lines; users can
collapse branches themselves. Lower/Initial/Upper and Comparison/Target/Scale
remain visible as small labels inside their fields, including while typing.

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
terminal** below the meanline results shows one SLSQP report, plus any
clipping, recovery, or error diagnostics. Its objective is signed and normalized for
minimization; the separate progress summary shows the physical objective.
The terminal follows new output until you scroll up.

Only converged results that pass final feasibility and first-order checks update the inputs, geometry, performance,
tables, and downstream geometry/mesh signatures. Failed solves retain the last
displayed design. Normal manual editing remains available afterward. Loading
YAML also restores its optimization editors and bounds; both flat and older
nested speed schemas are accepted. Saving a successfully optimized design
includes its optimization settings without reapplying old initial overrides.

`app_turbine_optimization/workbench.py` owns the layout and display formatting;
`shared/fluid_inputs.py` owns readable thermodynamic properties and safe pair conversion.
`app_turbine_optimization/opt_model.py` owns editor choices and validation;
`app_turbine_optimization/opt_panel.py` owns the panel/callbacks;
`app_turbine_optimization/opt_selection.py` owns hierarchical variable and
constraint menus and applies selection changes while preserving edits;
`app_turbine_optimization/opt_jobs.py` owns isolated processes, session ownership,
progress, and logs. Its CSS/JavaScript live in `app_turbine_optimization/assets/` and
are loaded only by this version. Numerical code lives in the
`turbodash.optimization` package, independently of Dash. Its optional `logger`
receives the single SLSQP report and diagnostics without redirecting server stdout.
Public callback snapshots remain in physical units and include the matching
configuration and turbine results for future live geometry rendering.

The optimizer uses Turbodash's driver around SciPy 1.17.1; PySolverView remains
a `jaxprop` dependency but does not run the optimization. Recognized meanline
domain failures carry metadata so the driver can reject a trial and retry a
shorter step. Invalid roots are not clipped and no artificial objective penalty
is introduced. Invalid starting designs, exhausted recovery budgets, and
nonrecoverable model consistency errors produce a failed job while retaining
the displayed design.

Jobs use a local temporary cache, signed session cookies, atomic progress files,
a 10 minute worker limit, and at most two concurrent solves per host. Completed
jobs expire after 24 hours. Set `TURBODASH_SECRET_KEY` for a deployment-managed
signing key. Multiple web workers on one machine share the cache; deployment
across separate hosts needs a shared job backend. Progress is currently shown
in the status/terminal; design plots update when the result is accepted.

Validation commands:

```powershell
python -B -m pytest tests/test_optimization_api.py -q
python -m pytest tests/test_app_optimization.py tests/test_app_entrypoints.py tests/test_app_meshing.py tests/test_mesh_jobs.py tests/test_refactored_layout.py -q
python tests/browser_optimization.py
```

The optional browser check needs Playwright plus Chromium (`python -m pip
install playwright` and `python -m playwright install chromium`). It starts and
stops its own local app server and writes screenshots and a JSON report.
