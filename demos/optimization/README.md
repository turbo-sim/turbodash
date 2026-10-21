# Turbine optimization examples

These two commented scripts use the public `turbodash.optimization` API and
the same meanline model and SLSQP driver as the optimization app. They require
the normal project environment, including the pinned SciPy 1.17.1 dependency;
there is no separate example-specific installation or solver implementation.

Run from the repository root, using `poetry run python` or the environment's
Python executable. Both scripts save plots without opening windows by default;
add `--show` for interactive figures. Their default inputs and output folders
are beside the scripts, so execution by absolute script path also works from
another working directory. Relative `--case`/`--output` paths start beside the
script; absolute paths work as supplied.

## 1. Basic API, results and reporting

```powershell
python -B demos/optimization/optimize_turbine.py --case axial
python -B demos/optimization/optimize_turbine.py --case radial
python -B demos/optimization/optimize_turbine.py --case axial --show
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
print(result.success, result.message)
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
| `convergence_history.png/.svg` | Physical objective and normalized constraint convergence. |

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
are listed by `SLSQPOptions` and can differ from those choices. For additional
selectors, variable choices and control details, see the
[calculation guide](../../docs/source/guides/calculations.md) and
[SLSQP guide](../../docs/source/guides/optimization.md).

## 2. Step fractions, failure and recovery

```powershell
python -B demos/optimization/compare_slsqp_controls.py
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
- `comparison.png/.svg`, showing physical efficiency, normalized constraint
  violation and KKT-to-tolerance ratio against accepted iterations.
- A subdirectory per policy with `solver.log` and the four API YAML exports.
  Each export records actual recovery events and its exact input settings.

For this supplied case, a real production run observed:

| Policy | Accepted | Major iterations | Final efficiency [%] |
| --- | --- | ---: | ---: |
| Full, no recovery | No: first trial leaves the model domain | 1 | 90.684812 |
| Full with recovery | Yes: one domain rejection | 38 | 89.811945 |
| Half first | Yes | 38 | 89.811945 |
| Half every direction | Yes | 55 | 89.811945 |
| Gradual restoration | Yes | 39 | 89.811939 |

The rejected row retains its infeasible initial design; its higher efficiency
does not make it a successful design. Circles mark accepted endpoints and crosses
mark rejected outcomes. The comparison exits zero when all runs complete with
controlled outcomes, including that expected failure; unexpected exceptions
still fail the process. Changed inputs or solver versions can change the results.

This example shows why always damping can cost iterations and why recovery can
help a full step. It does not establish one universally best policy. Inspect the
KKT diagnostics as well as the efficiency and feasibility curves: small steps
or objective changes alone do not prove convergence. A ratio above one or
unavailable final diagnostics does not pass the independent acceptance check.

## Development maintenance

The independent 52-case KKT/tolerance audit and a record of why the custom
interface was necessary are in
[`development/optimization`](../../development/optimization/README.md).
Those tools are for solver maintenance; these demos are the shorter starting
point for using the API.
