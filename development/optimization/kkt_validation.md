# Turbine integration and PySolverView tolerance study

This experiment runs the same `optimize_turbine` entry point used by the app.
It checks the turbine adapter and records PySolverView's diagnostics and
acceptance flags. It does not require every optimization problem to converge.

From the repository root:

```powershell
poetry run python development/optimization/verify_kkt.py --workers 2
```

Use `--preflight` to evaluate the five initial designs, `--filter axial_nominal`
to run a subset, or `--resume` to reuse completed cases. A custom `--output`
directory keeps independent experiment runs separate. Resume should only be
used with unchanged code and case definitions.
`--workers 1` runs in the current process; larger values use a process pool.

The harness calls the turbine adapter in `turbodash.optimization`, which uses
`pysolver_view.OptimizationSolver` for optimization. It calls the returned
frontend directly for convergence plots, convergence tables, and optimization
reports including design variables, constraints, KKT checks, and multipliers.
The poor axial start is [case_poor_start.yaml](case_poor_start.yaml), while the
radial start is the canonical [radial demo](../../demos/optimization/case_radial.yaml).
Both use the optimizer's numeric shaft-speed input format, with no app imports.
Generated artifacts go to ignored `output/kkt/` by default. Historical result
directories are not bundled with this harness; rerun the cases to assess the
current implementation.

The 52 cases comprise:

- Three axial starting designs: very poor (blade velocity ratio 0.20, stator
  exit angle 40 degrees), nominal (0.60, 70 degrees), and high-speed/high-angle
  (0.85, 78 degrees). All are initially physically evaluable and within bounds.
- Four axial constraint sets: variable bounds alone, flaring within +/-10
  degrees, flaring within +/-5 degrees with maximum Mach <= 1.05, and exactly
  zero flaring in both rows.
- Two radial starting designs with zero-flaring equality constraints.
- Three tolerance profiles for each combination, shown below.
- Four additional axial mixed-constraint and equality-constraint cases which
  tighten only feasibility or only optimality, holding the native solver
  settings fixed.
- Two deliberately impossible constraint sets, requiring each row's flaring
  angle to be both >= +5 degrees and <= -5 degrees.
- Four additional strict poor-start cases repeating the bounds-only and
  flaring-limited optimizations with finite-difference steps of 1e-7 and 1e-8.

| Profile | Native `ftol` | Feasibility tolerance | Optimality tolerance |
| --- | ---: | ---: | ---: |
| Loose | 1e-5 | 1e-3 | 1e-2 |
| Standard | 1e-9 | 1e-6 | 1e-4 |
| Strict | 1e-12 | 1e-9 | 1e-6 |
| Feasibility gate | 1e-5 | 1e-12 | 1e-2 |
| Optimality gate | 1e-5 | 1e-3 | 1e-10 |

The normalized finite-difference step is 1e-6 except in the four derivative
sensitivity reruns. All runs use physical-domain
recovery, a first-step fraction of 0.20, and up to 200 iterations (40 for the
impossible problems). `step_growth_factor: null` restores the maximum cap
immediately after the reduced first step and after accepted recovered steps.
Domain recovery can still shorten an invalid trial. These are
study controls, not live defaults. Current `SLSQPOptions` defaults use
`finite_diff_step: 1e-7`, `first_step_fraction: 0.25` and
`step_growth_factor: 1.5`. Both the study and the current production driver
evaluate each derivative probe once; invalid derivative probes stop the solve
without retrying a different direction or smaller step. Trial-step physical
domain recovery remains enabled. Tolerances apply to normalized residuals: for example,
flaring constraints use a 10-degree scale, so feasibility tolerance 1e-6
corresponds to a 1e-5-degree allowed residual for those components.

## Physical integration validation

The harness reevaluates the final turbine design and compares its physical
objective and constraint data with the returned results. It checks the saved
design inputs, physical variable bounds, accepted-point and iteration alignment,
and physical objective scaling in the callback history. It also checks that the
turbine adapter preserves PySolverView's success, native status, and feasibility
flags. These checks do not reconstruct KKT residuals or acceptance thresholds.

PySolverView supplies all KKT calculations and the reported `kkt_passed`,
optimality, and tolerance ratios. The former local KKT checker, derivative audit,
and custom plotting/report helpers have been removed.

The additional analytic tests in PySolverView
(`tests/optimization/solvers/test_custom_slsqp_kkt_tolerances.py`) cover known
derivatives, equality/inequality and bound multiplier signs, and tolerances
immediately below, equal to, and above the independently computed residual.

## Results

The generated `summary.json` and `summary.csv` distinguish an accepted optimization
from passing physical integration validation. A rejected infeasible or
insufficiently optimal point can pass validation. The script exits nonzero for
an integration mismatch or unexpected experiment error, not for an expected
optimization rejection.

Each case directory contains its exact `case.yaml`, `solver.log`, saved input
and final cases, full `optimization_result.yaml`, physical `validation.json`, and
`summary.json`. PySolverView writes `convergence_history.png`,
`convergence_history.svg`, `convergence_history.txt`, and
`optimization_report.txt` into each case directory. The parent output directory
contains only combined matrix data in JSON and CSV. Failed endpoints are
diagnostic designs and should not be treated as successful optimized designs.

`--retry-validation-failures` reruns failed integration checks with `--resume`;
the old `--retry-audit-failures` spelling remains an alias. Old audit summaries
are rerun because they lack the current validation schema. `--no-summary`
(also available as `--no-report`) preserves combined matrix data while still
saving each case's PySolverView artifacts. Use a fresh output directory to avoid
mixing older custom plots or reports with the current results.

## Historical findings addressed in production

The sweep exposed a history-alignment bug at very small accepted steps. The
turbine adapter previously merged nearly identical physical inputs, then
attached the later step/KKT diagnostics to the earlier iteration. It now tracks
exact normalized solver points, preserving each accepted row. The final refresh
of the identical point remains deduplicated. This changes reporting, not the
optimizer's steps. `tests/test_kkt_history_alignment.py` covers callbacks and
YAML exports when distinct normalized floats map to the same physical float.

Failure messages also now retain enough precision to distinguish a residual
just above its tolerance. For example, 1.0033539e-6 versus 1e-6 is no longer
displayed as "1e-6 exceeds 1e-6". One-ULP differences are covered by regression
tests as well. These historical findings explain the retained integration
coverage; the previous sweep results are not evidence for the current checkout.

Changing the independent acceptance tolerances alone does not make native
SLSQP iterate further. The equality-constraint comparison demonstrates the
same endpoint being accepted under loose tolerances and rejected by either a
stricter feasibility or optimality check. Obtaining a more accurate endpoint
also requires appropriate native stopping settings and derivative accuracy.

In the historical poor-start sensitivity runs, a derivative step of 1e-7 gave more stable
strict stationarity estimates than 1e-8. A smaller step is not automatically
more accurate: subtractive cancellation and model-evaluation precision also
matter. The current production default derivative step is 1e-7. The experiment
retains its explicit 1e-6, 1e-7 and 1e-8 settings for fresh comparisons. Recheck
the current run before drawing conclusions about which step is most reliable.
