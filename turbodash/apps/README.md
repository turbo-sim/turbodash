# Applications

Each module owns its Dash app, server and callbacks. The stable and experimental
turbine applications remain independent.

| Application | Module | Existing launch command | YAML demo |
| --- | --- | --- | --- |
| Turbine meanline, stable/default | `turbodash.apps.turbine` | `turbodash` | [Runner](../../demos/apps/turbine/run.py), [case](../../demos/apps/turbine/case.yaml) |
| Turbine geometry and meshing workspace | `turbodash.apps.turbine_experimental` | `turbodash-experimental` | [Runner](../../demos/apps/turbine_experimental/run.py), [case](../../demos/apps/turbine_experimental/case.yaml) |
| Original constant-loss stage calculator | `turbodash.apps.stage` | `turbodash-stage` | [Runner](../../demos/apps/stage/run.py), [case](../../demos/apps/stage/case.yaml) |
| Dimensionless stage efficiency explorer | `turbodash.apps.stage_efficiency` | Use its demo runner or `main()` | [Runner](../../demos/apps/stage_efficiency/run.py), [case](../../demos/apps/stage_efficiency/case.yaml) |

The existing `turbodash.launch_app_turbine()`, `launch_app_experimental()` and
`launch_app_stage()` helpers still open the selected app with their original
browser/reloader behavior. The console commands run with `debug=False`.

Each application also accepts an optional startup case:

```python
from turbodash.apps import turbine
turbine.main("case.yaml")
```

`configure("case.yaml")` applies the YAML to the initial controls and stage
stores without starting the server. Call it before serving requests. It does
not replace the app's calculations or callbacks. Dimensional apps retain their
existing `inputs` schemas; the efficiency demo documents its displayed parameters.

Run `python -m pip install -e .` after updating a checkout to register the
canonical console targets. Stable deployment uses
`gunicorn turbodash.apps.turbine:server`; experimental deployment uses
`gunicorn turbodash.apps.turbine_experimental:server` and the optional `mesh`
extra. Stable startup does not import the optional meshing/VTK stack.

`components/` contains reusable panels and callback registration, not launchable
apps. `mesh_jobs.py` manages isolated worker processes and artifacts;
`python -m turbodash.apps.mesh_jobs REQUEST_JSON` is its worker entry point.
`assembly_preview.py` provides the assembly CLI. Import applications, panels and
workers from `turbodash.apps`; the former flat module paths have been removed.
