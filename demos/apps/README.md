# Run the four applications

Each subfolder has `run.py` and `case.yaml`. The runner loads its adjacent YAML
into the real app controls before starting the server. Edit that file to change
the initial design; no separate upload is needed.

From the repository root, using the environment with Turbodash's dependencies:

```powershell
python demos/apps/turbine/run.py
python demos/apps/turbine_experimental/run.py
python demos/apps/stage/run.py
python demos/apps/stage_efficiency/run.py
```

Run one command at a time and open `http://127.0.0.1:8050/`. Stop it with Ctrl+C
before starting another. You can also change into a subfolder and use
`python run.py`; case paths are resolved beside the runner rather than against
the current working directory. Runners use this repository's package.

| Folder | Initial case |
| --- | --- |
| [turbine](turbine/run.py) | Single axial air turbine, Benner losses, 5 bar / 600 K inlet and 1 bar exit. |
| [turbine_experimental](turbine_experimental/run.py) | Same turbine with KO losses, plus geometry and mesh workspaces. Native mesh generation needs the optional mesh dependencies. |
| [stage](stage/run.py) | Original radial constant-loss stage calculator, with dimensional inputs under `inputs`. |
| [stage_efficiency](stage_efficiency/run.py) | Dimensionless efficiency curves for five reaction values. `inputs` contains the displayed `alpha1`, `alpha2`, `radius`, loss coefficients, velocity-ratio bounds and `R_values` list. |

The original stage and dimensionless explorer use prescribed constant losses.
The turbine applications select the production empirical models. All four apps
retain their existing equations and controls.

The production tests exercise each YAML case through Dash's calculation HTTP
endpoint as well as checking layouts, runners and worker
entry points. See the [test workflow](../../tests/README.md).
