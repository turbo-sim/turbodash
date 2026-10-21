# Optimization maintenance

Production code lives in [turbodash/optimization](../../turbodash/optimization/).
This directory retains the independent KKT audit, one difficult starting case
and the reasoning behind the solver interface. User-facing examples belong in
[demos/optimization](../../demos/optimization/README.md).

## Consolidation assessment

| Previous material | Decision |
| --- | --- |
| Original optimization runner and axial/radial cases | Move into the basic demo. Move useful API contracts to tests/test_optimization_api.py, collected by the production suite. |
| Standalone optimization_slsqp driver, turbine adapter and SciPy bridge/license | Remove the prototype copies. Production owns the implementation, license and solver regression tests. |
| Original 14-variable SLSQP comparison | Keep its case in the advanced demo and its compact historical benchmark here. Replace the old runner with the educational comparison. |
| poor_initial_design, step_fraction_sweep and step_ramp studies/results | Remove the separate studies. Retain the poor-start input for the latest KKT matrix; explain a small set of fraction/recovery policies in the advanced demo. |
| optimization_troubleshoot hooks, penalty/root workarounds and saved analyses | Remove superseded experiments; retain their conclusions below. |
| Latest independent KKT validation | Keep the current runner, independent checks, report generation and documented 52-case matrix here. Regenerate results when maintaining the solver. |

The old API tests are relocated, not discarded. The axial/radial demo cases
remain inputs to model-domain and app regression tests. This consolidation
does not change production optimizer or meanline equations.

## Maintained files

| File | Purpose |
| --- | --- |
| [verify_kkt.py](verify_kkt.py) | Production optimizer over the 52-case acceptance matrix; preflight, filtering, workers and resumption. |
| [kkt_checks.py](kkt_checks.py) | Independent residual signs/scales, multiplier complementarity, step norms, history alignment and fresh final finite-difference stationarity. Does not call the production KKT checker. |
| [kkt_report.py](kkt_report.py) | Matrix reports and convergence plots. |
| [case_poor_start.yaml](case_poor_start.yaml) | Physically evaluable poor initial design used by the matrix. |
| [kkt_validation.md](kkt_validation.md) | Matrix definitions, tolerance profiles, audit method and findings addressed by regressions. |
| [historical_benchmark.json](historical_benchmark.json) | Original SciPy 1.17.1 comparison; historical evidence, not a current benchmark or acceptance oracle. |

Generated exports, logs and reports go under ignored output/. No old result
directories are needed. Resume only runs with unchanged code, inputs and settings;
use a fresh output directory after changing the solver.

## Why the SLSQP interface was needed

A turbine can satisfy every variable bound and still have no real velocity
triangles. The public SciPy SLSQP wrapper does not expose the pending trial
state needed to shorten that step consistently and retry while retaining the
solver state. Its iteration callback is too late for that operation.

The production bridge updates the pending point, direction and line-search
quantities together. SciPy still performs SLSQP's quadratic subproblems, merit
line search and BFGS updates. Turbodash handles model evaluation, trial control,
diagnostics and result acceptance. It does not monkeypatch installed SciPy,
clip invalid roots or substitute artificial objective penalties. Only recognized
model-domain failures are recoverable; programming and consistency errors remain
visible.

| Production module | Responsibility |
| --- | --- |
| turbine.py | Physical/configured variables, normalization, responses, callbacks, fresh final reevaluation and YAML exports. |
| slsqp.py | Evaluation loop, trial fractions, recovery, derivatives, stopping/reporting and independent acceptance. |
| _slsqp_backend.py | Version-specific SciPy state/workspace and consistent pending-trial shortening. |
| kkt.py | Feasibility and first-order optimality, including variable bounds. |

## Step fractions and recovery

first_step_fraction scales the first QP direction; max_step_fraction caps
every direction, including the first. These are fractions in normalized coordinates, not
absolute physical movement limits. Smaller fractions can help a particular
start but can also require more iterations.

With domain_backtracking enabled, recognized invalid line-search trials are
shortened by backtrack_factor, up to max_backtracks times. A finite
step_growth_factor restores a reduced cap gradually toward the maximum; None
restores it immediately for the next major direction. Recovery cannot repair
an invalid initial design or make incompatible constraints feasible.

Finite-difference probes are separate. The current driver evaluates each once
in a bound-admissible direction. A physically invalid probe ends the derivative
calculation with a diagnostic; line-search recovery does not retry it. The old
max_fd_backtracks setting is ignored for legacy inputs and omitted from the
maintained examples.

The [advanced demo](../../demos/optimization/README.md) shows a small controlled
comparison. The [SLSQP guide](../../docs/source/guides/optimization.md) documents
every setting. Current defaults come from SLSQPOptions; the maintenance matrix
deliberately supplies explicit study settings.

## KKT acceptance and reporting

Native termination, feasibility and final acceptance are distinct:
result.solver_success records native convergence; result.feasible checks
normalized residuals and bounds; result.success additionally requires the
independent first-order assessment. Small steps or objective changes alone
do not establish stationarity.

The KKT assessment checks primal feasibility, Lagrangian stationarity, multiplier
dual feasibility and complementarity. Constraint signs, scales and variable
normalization must agree throughout the solver, multiplier reconstruction and
exports. When diagnostics are available, kkt_ratio <= 1 summarizes passing the
configured acceptance limits. Unavailable derivatives must not appear as zero
optimality.

The audit reconstructs these quantities independently from physical outputs and
fresh probes. An audit may pass when an optimization is correctly rejected.
It also checks callback/export history alignment: distinct normalized points
must not be merged merely because their physical values are nearly equal.
Focused regressions include tests/test_kkt_history_alignment.py and
tests/test_kkt_tolerances.py.

## Running maintenance checks

From the repository root, in the project environment:

~~~powershell
python -B -m pytest tests/test_slsqp.py tests/test_optimization_api.py tests/test_kkt.py tests/test_kkt_tolerances.py tests/test_kkt_history_alignment.py tests/test_kkt_reporting.py -q
python -B development/optimization/verify_kkt.py --preflight
python -B development/optimization/verify_kkt.py --filter axial_nominal__flaring__standard --workers 1
python -B development/optimization/verify_kkt.py --workers 2
~~~

The full matrix is a maintenance study, not a quick unit test. Its report
distinguishes audit failures from rejected optimization outcomes. Follow with
relevant app regressions when changing callback or export contracts.

The private bridge requires SciPy 1.17.1. Upgrades require reviewing its
interface and workspace layout, then rerunning numerical, recovery, KKT, turbine
and app checks. The historical benchmark cannot establish compatibility with
another release.

## Retained historical findings

The original problem used 14 variables, four flaring inequalities and no
equalities. A full unrecovered first trial left the physical domain. Adaptive
halving and a half-sized first trial both converged to about 89.811945% efficiency
in 38 major iterations and 586 valid model evaluations, including final
derivative verification. Halving every direction took 55 iterations and
840 evaluations in that recorded comparison.

Those observations motivated recovery and showed the cost of unnecessary
damping; they are not promises for other inputs or current revisions. The
advanced demo reruns a small comparison through the current API and reports its
actual outcomes. The KKT matrix later exposed history-alignment and tolerance
message issues now covered by production tests. Large old result sweeps and
private experimental hooks are not needed to retain those lessons.
