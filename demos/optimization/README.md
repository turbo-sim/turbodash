# Turbine optimization examples

These two commented scripts use the public `turbodash.optimization` API and
the same meanline model and PySolverView `OptimizationSolver` frontend as the
optimization app. The frontend selects `library="pysolver", method="slsqp"`.
Turbodash maps turbine inputs and physical reports; PySolverView owns
optimization, finite differences, domain recovery, independent KKT checks,
and the convergence table and final solver report.

PySolverView is a required PyPI dependency, with version `>=0.7.2,<0.8`
(Poetry `^0.7.2`). Install from the Turbodash repository root and inspect the
version and installation used by its environment:

```powershell
poetry install
poetry run python -c "import inspect, pysolver_view as psv; print(psv.__version__); print(psv.__file__); print(inspect.getfile(psv.OptimizationSolver))"
```

The version should satisfy that range; the paths identify the active PySolverView
installation in the Poetry environment. SciPy 1.17.1 is required by the controlled
backend. PySolverView is imported from that installation; there is no separate
example solver implementation.

Run from the repository root, using `poetry run python` or the environment's
Python executable. Both scripts save plots without opening windows by default;
add `--show` for interactive figures. Their default inputs and output folders
are beside the scripts, so execution by absolute script path also works from
another working directory. Relative `--case`/`--output` paths start beside the
script; absolute paths work as supplied.

## 1. Basic API, results and reporting

```powershell
poetry run python -B demos/optimization/optimize_turbine.py --case axial
poetry run python -B demos/optimization/optimize_turbine.py --case radial
poetry run python -B demos/optimization/optimize_turbine.py --case axial --show
```

Read [`optimize_turbine.py`](optimize_turbine.py) from top to bottom. It explains
loading a case, selecting physical design variables and constraints, observing
accepted progress with a callback, running the optimizer, inspecting acceptance
and exporting reusable inputs and reports. The core API call is:

```python
from turbodash.optimization import load_case, optimize_turbine, save_optimization_results

case = load_case("demos/optimization/case_axial.yaml")
result = optimize_turbine(case)
save_optimization_results(result, "output/my_turbine")
```

This short snippet uses paths relative to your shell; the full script resolves
its own paths. Its default output is `output/axial/` or `output/radial/` here.

| Output | Meaning |
| --- | --- |
| `input_case.yaml` | Original inputs and optimization settings. |
| `initial_case.yaml` | Effective starting inputs after bounds clipping and explicit initial overrides. |
| `optimized_case.yaml` | Final inputs/settings, with initial overrides removed so a restart uses this design. |
| `optimization_result.yaml` | Physical performance, variables, constraint report, accepted history, solver history, KKT diagnostics, recovery events and evaluation/time counts. |
| `initial_turbine.png/.svg`, `optimized_turbine.png/.svg` | Initial and final turbine figures. |
| `convergence_history.png/.svg` | PySolverView's signed/scaled objective and constraint convergence against accepted iterations. |
| `convergence_history.txt` | PySolverView's convergence table and final solver status. |
| `optimization_report.txt` | PySolverView's physical and normalized design variables, named constraints, KKT checks, and Lagrange multipliers. |

A result export identifies `pysolver_view.OptimizationSolver` and the `slsqp`
method, so it is traceable to the optimizer actually used.

PySolverView prints the optimization status, normalized feasibility and KKT
optimality, evaluation counts, and timing. After the solve, the basic demo also
prints and saves its full optimization report, including design-variable bounds,
each named physical constraint, KKT conditions, and constraint/bound multipliers.
The final KKT report uses the solve's separate feasibility and optimality limits
and reuses its saved certificate; rendering the report and convergence plot does
not run the turbine model again. The basic demo adds physical turbine performance
and output paths.

The PySolverView convergence plot is saved by default. Run with `--show` to open
it and the turbine figures interactively; omitting that flag keeps the run
headless. Its signed objective is negative when maximizing efficiency; physical
efficiency remains available in the turbine results and YAML exports.
On rerun, the demo removes the retired `physical_convergence_history.png/.svg`
files from its output directory so old plots are not mixed with the native output.

The retained main frontend is available as `result.optimizer` for further
PySolverView plots and reports. For example:

```python
result.optimizer.print_optimization_report(
    include_kkt_conditions=True,
    include_multipliers=True,
    savefile=True,
    filename="optimization_report.txt",
    output_dir="output/my_turbine",
)
result.optimizer.plot_convergence_history(
    savefile=True,
    filename="convergence_history",
    output_dir="output/my_turbine",
    showfig=False,
)
```

The default final report uses saved solve data. Explicitly requesting another
design or a different tolerance is a separate diagnostic request and can perform
additional model or KKT work.

A final file is written even when a solve is rejected. Check `result.success`
before adopting its design. `solver_success` is native termination, `feasible`
is the independent constraint/bound check, and final acceptance also requires
the first-order KKT check. Native `ftol`, feasibility tolerance and optimality
tolerance have different purposes. The CLI exits nonzero for an unsuccessful
basic solve while preserving its reports.

The callback receives independent snapshots of accepted physical designs;
rejected trials and derivative probes do not produce progress rows. Physical
efficiency in `result.history` is positive. The solver minimizes a signed,
normalized objective, so its objective is negative when maximizing efficiency.
Step norms and KKT diagnostics use normalized solver coordinates.

`result.optimizer` is the live PySolverView frontend for plotting and reporting;
it is excluded from YAML and application-worker payloads.
`result.frontend_result` is PySolverView's detached public result: its `x` is
physical, `x_solver` is normalized, and `nfev` includes derivative probes.
The retained `result.solver` contains raw backend diagnostics and normalized
`x`; its `nfev` excludes derivative probes supplied by the frontend.
`result.n_evaluations` separately counts successful uncached turbine evaluations,
including the fresh final check. YAML exports include `frontend_result` and
`frontend_convergence_history` alongside the backend history and recovery events.

### Editing the input

[`case_axial.yaml`](case_axial.yaml) and [`case_radial.yaml`](case_radial.yaml)
include comments about the case schema:

- Only fields listed under `optimization.design_variables` vary. Dotted paths
  start inside `inputs`; bounds and initial values use physical units. Omit a
  variable to hold it fixed. An optional `initial_value` overrides the input.
- `objective_function` selects a scalar result and `maximize` or `minimize`.
  The examples maximize total-to-static efficiency.
- Each wildcard match in a constraint creates a separate scalar constraint.
  The examples require zero flaring in both rows; that illustrates equality
  constraints rather than a universal turbine design rule. A physically valid
  starting point can still be infeasible, and enforcing constraints can reduce
  its original efficiency.
- Constraint normalization sets residual scale. With a 10-degree scale,
  feasibility tolerance `1e-6` permits `1e-5` degrees of residual.
- Specific speed is dimensionless when `rotational_speed_type: specific`;
  `actual` uses rpm. Optimize the numeric `rotational_speed` field while its
  type remains fixed.

The commented cases retain explicit example settings. Current driver defaults
are listed by `pysolver_view.SLSQPOptions` and can differ from those choices.
The turbine adapter preserves the existing `[0, 1]` coordinates, explicit
finite-difference step, and separate feasibility and optimality tolerances when
calling `pysolver_view.OptimizationSolver` with `problem_scale=1.0`. For additional
selectors, variable choices and control details, see the
[calculation guide](../../docs/source/guides/calculations.md) and
[SLSQP guide](../../docs/source/guides/optimization.md).

## 2. Step fractions, failure and recovery

```powershell
poetry run python -B demos/optimization/compare_slsqp_controls.py
```

[`compare_slsqp_controls.py`](compare_slsqp_controls.py) starts every run from
the same [`case_recovery.yaml`](case_recovery.yaml). Only the four step/recovery
settings below change. It uses five policies rather than a large parameter
sweep, and keeps the numerical problem and tolerances identical:

| Policy | First fraction | Maximum fraction | Growth factor | Domain recovery |
| --- | ---: | ---: | --- | --- |
| Full, no recovery | 1.0 | 1.0 | Immediate restoration | Off |
| Full with recovery | 1.0 | 1.0 | Immediate restoration | On |
| Half first | 0.5 | 1.0 | Immediate restoration | Off |
| Half every direction | 0.5 | 0.5 | Immediate restoration | On |
| Gradual restoration | 0.25 | 1.0 | 1.5 | On |

`first_step_fraction` and `max_step_fraction` multiply the proposed QP direction,
not absolute physical movements. Native merit-function line search still runs.
Recovery shortens recognized invalid pending model trials without changing the
equations or discarding the solver state. `step_growth_factor` controls how a
reduced cap returns toward its maximum; `None` restores it immediately for the
next major direction. Finite-difference probes are separate and are not retried.

The default `output_controls/` contains:

- `summary.csv` and `summary.yaml`, listing settings, acceptance, native status,
  efficiency, violation, KKT ratio, iterations, recovery count and model calls.
- A subdirectory per policy with PySolverView's native
  `convergence_history.png/.svg`, `optimization_report.txt`, `solver.log`,
  and the four API YAML exports.
  The plot shows the signed/scaled objective and constraint violation against
  accepted iterations. Each export records actual recovery events and exact
  input settings.

For this supplied case, a PySolverView run reproduced the previous
Turbodash driver's outcomes:

| Policy | Accepted | Major iterations | Final efficiency [%] |
| --- | --- | ---: | ---: |
| Full, no recovery | No: first trial leaves the model domain | 1 | 90.684812 |
| Full with recovery | Yes: one domain rejection | 38 | 89.811945 |
| Half first | Yes | 38 | 89.811945 |
| Half every direction | Yes | 55 | 89.811945 |
| Gradual restoration | Yes | 39 | 89.811939 |

The rejected row retains its infeasible initial design; its higher efficiency
does not make it a successful design. The summary records accepted and rejected
outcomes explicitly. The comparison exits zero when all runs complete with
controlled outcomes, including that expected failure; unexpected exceptions
still fail the process. Changed inputs or solver versions can change the results.

This example shows why always damping can cost iterations and why recovery can
help a full step. It does not establish one universally best policy. Inspect the
KKT diagnostics as well as the efficiency and feasibility curves: small steps
or objective changes alone do not prove convergence. A ratio above one or
unavailable final diagnostics does not pass the independent acceptance check.

## 3. Time KKT checks

```powershell
poetry run python demos/optimization/compare_kkt_timing.py
poetry run python demos/optimization/compare_kkt_timing.py --case radial --repeats 3
```

[`compare_kkt_timing.py`](compare_kkt_timing.py) runs the same case with
PySolverView's `check_kkt` enabled and disabled. It warms up each mode once,
then times three pairs in alternating order. Each run uses a fresh problem
and the same initial design, bounds, constraints, tolerances, and step controls.
Only `solve()` is timed; setup, console output, plots, and reports are excluded.

The console table shows solve time, KKT checker time, iterations, total model
calls, KKT check counts, outcomes, and physical objective. The final lines compare
median solve times. `--repeats 1` gives a shorter comparison. The script produces
no plots or result files and does not change the app's mandatory KKT policy.
With checks disabled, success requires native convergence and feasibility;
optimality is unassessed. Different iteration/evaluation counts can also affect
the measured time difference.
Small wall-time differences can be timing noise; increase `--repeats` for more
samples and inspect the separately measured checker time.

## Development maintenance

The 52-case turbine integration matrix is retained in
[`development/optimization`](../../development/optimization/README.md).
It checks real turbine scaling, physical outputs, callbacks, and exports.
Its KKT checks, optimization reports, and convergence plots are all delegated
to PySolverView. These demos are the shorter starting point for the turbine API.
