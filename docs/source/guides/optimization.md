# Turbine optimization

Turbodash uses PySolverView's controlled SLSQP optimizer to improve a selected
objective while respecting design-variable bounds and configured constraints.
The Python API and optimization workbench use the same public
`pysolver_view.OptimizationSolver` frontend with `library="pysolver"` and
`method="slsqp"`, including its shared reporting and acceptance checks.
Turbodash supplies the turbine problem mapping and physical reports;
PySolverView supplies derivatives, step controls, recovery, KKT diagnostics,
and the convergence table and final solver report.

Start with the [calculation workflow](calculations.md) to choose the objective,
variables, and constraints. For a complete Python example, see
[demos/optimization](https://github.com/turbo-sim/turbodash/tree/main/demos/optimization)
in the repository:

```python
from turbodash.optimization import load_case, optimize_turbine

case = load_case("demos/optimization/case_axial.yaml")
result = optimize_turbine(case)
```

Run this example from a repository checkout. The basic demo explains reporting
and exports; the advanced demo compares step fractions and recovery policies.

PySolverView is a required PyPI dependency with version `>=0.7.2,<0.8`
(Poetry `^0.7.2`). Run `poetry install` from Turbodash before using
`poetry run python` for the demos. See the [installation guide](../installation.md)
for checking the installed version and import location.

In the app, **Stop** beside **Start optimization** cancels the running worker.
The current design and convergence log are retained; the unfinished result is
not applied. Stop is enabled only while a numerical run is active. You can
adjust the settings and start a fresh run after stopping.

## Account for exhaust recovery

After the `stages` section, turbine inputs include a prescribed diffuser recovery:

```yaml
inputs:
  # Operating conditions...
  stages:
    # Stage definitions...
  diffuser:
    kinetic_energy_recovery_fraction: 0.60
```

In **Meanline > 05 Diffuser**, enter **Exhaust kinetic-energy recovery [%]**
between 0 and 100. The default is 60%, including for older cases without this
section. Saving a design writes the fraction between 0 and 1 after `stages`.

The fifth headline result, **Efficiency with exhaust recovery**, uses

$$
\eta_{\mathrm{recovery}} =
\frac{P_{\mathrm{shaft}}}
{\dot m\left(\Delta h_{\mathrm{is,ts}} - f_{\mathrm{rec}} V_{\mathrm{exit}}^2/2\right)}.
$$

The absolute velocity at the final rotor exit includes meridional and tangential
components. At 0% recovery this equals total-to-static efficiency; at 100% it
equals TurboDash's total-to-total efficiency. Intermediate values represent
partial recovery of exhaust kinetic energy and provide an efficiency estimate
between those limits. This prescribed assumption does not solve a diffuser
outlet state or modify shaft power, turbine flow, or the exit-pressure boundary.

The overall result key is `efficiency_ts_with_recovery`. It is available as an
optimization objective or constraint under **Overall**. The recovery fraction
is an assumption held fixed during optimization, not a design variable. The
original total-to-static and total-to-total metrics remain available.

## Constrain flow angles and blade-row turning

Under **Constraints > + Add constraint**, each stage's **Station 1–4** menus
offer **Absolute flow angle [deg]** (`alpha`) and **Relative flow angle [deg]**
(`beta`). These are calculated quantities, so they constrain the velocity
triangles without introducing independent angle design variables.

| Station | Boundary | Frame relevant to the blades |
| --- | --- | --- |
| 1 | Stator inlet | Absolute; equals relative for the stationary stator |
| 2 | Stator outlet | Absolute; equals relative for the stationary stator |
| 3 | Rotor inlet | Relative |
| 4 | Rotor outlet | Relative |

**+ Add groups** offers **Stator inlet relative flow angle** and **Rotor inlet
relative flow angle**, for all stages or an individual stage. They start with
an editable signed inlet-angle constraint **≥ 0°**, using `beta` at station 1
or 3 respectively. Negative angles retain their sign; adjust the comparison
and target for the intended velocity triangle.

To prevent almost straight rows, prefer **Minimum blade-row turning**. This
adds separate stator and rotor lower bounds, each initially **≥ 50°**:

$$
\Delta\theta_S=|\alpha_2-\alpha_1|,\qquad
\Delta\theta_R=|\beta_4-\beta_3|.
$$

An inlet-angle limit alone cannot guarantee either difference. The turning
group applies to each stage separately, so high turning in one row cannot
compensate for insufficient turning in another. The group is off until
selected, and both thresholds are independently editable. The 50° default
is a design starting point, not an aerodynamic validity limit.

Turning appears under row geometry in the constraint picker and geometry
report, with result key `geometry_meanline.<row>.flow_turning`. Angle constraints
use normalization scale 90°. They are saved and loaded with the optimization
settings. These constraints govern meanline flow turning; they do not restrict
subsequent geometry-only camberline edits, and are not incidence constraints.

## Constrain hub-to-tip ratios

For axial turbines, open **Meanline > Optimization > Constraints > + Add groups**.
Under **All stages**, activate **Hub-to-tip ratios** to add four limits: stator
inlet, stator outlet, rotor inlet, and rotor outlet. Each starts at a minimum of
0.8 with comparison **≥** and applies to every stage. This is an editable starting value, not a
mandatory design limit; the group is off until selected.

The same group is available under each individual stage. Using **+ Add constraint**,
you can also select **Inlet hub-to-tip ratio** or **Outlet hub-to-tip ratio** under the stator or
rotor geometry quantities. Change each comparison and target independently to
set a minimum, maximum, or equality constraint. The ratio is hub radius divided
by tip radius at that row boundary, with normalization scale 1.

These are calculated meanline quantities, so they constrain optimization rather
than directly setting blade geometry. They are omitted for radial turbines,
where the current model fixes both ratios at 1. Constraints are retained when
saving and loading the design's optimization settings.

## Configure the solver

The default workbench objective is **Efficiency with exhaust recovery**, listed
first in the objective menu. Loading a design with an explicit objective
preserves that selection.

In the workbench, **Optimization > Configuration** shows the iteration limit
and solver tolerance (`ftol`). **Show expert settings** reveals feasibility
and optimality tolerances, the first step fraction, growth factor, reduction
factor, and retry limit. Hiding these settings preserves their values.

The defaults are a useful starting point. If a run reaches the iteration limit,
inspect its convergence history before increasing `maxiter`. If it repeatedly
encounters invalid designs, review its bounds and try a smaller first step.
Tightening tolerances alone does not fix either problem.

### YAML and Python controls

Set the following fields under `optimization` in a case YAML. These are the
defaults; objectives, design variables, and constraints occupy their usual
neighboring fields.

```yaml
optimization:
  feasibility_tolerance: 1.0e-6
  solver_options:
    ftol: 1.0e-9
    maxiter: 200
    finite_diff_step: 1.0e-7
    first_step_fraction: 0.25
    max_step_fraction: 1.0
    step_growth_factor: 1.5
    domain_backtracking: true
    backtrack_factor: 0.5
    max_backtracks: 20
    optimality_tol: 1.0e-4
```

| Setting | Meaning |
| --- | --- |
| `ftol` | Native SLSQP stopping tolerance, including objective/step changes and aggregate feasibility and optimality tests. It does not guarantee a component-wise KKT bound. |
| `maxiter` | Maximum number of major iterations. |
| `finite_diff_step` | Absolute derivative-probe distance in normalized variable coordinates. |
| `first_step_fraction` | Fraction of the first major iteration's proposed QP direction to try initially. |
| `max_step_fraction` | Maximum fraction of each major iteration's proposed QP direction to try initially. |
| `step_growth_factor` | Growth of the starting step fraction after a reduced first step or domain recovery; defaults to `1.5`. `null` disables gradual restoration; a finite value at least `1` enables it. |
| `domain_backtracking` | Retry recognized invalid model trials at shorter distances. |
| `backtrack_factor` | Factor multiplying a rejected trial's displacement; `0.5` halves it. |
| `max_backtracks` | Maximum additional trial reductions for each native function request. |
| `optimality_tol` | Independent acceptance tolerance for KKT stationarity, dual feasibility, and complementarity in normalized coordinates. |
| `feasibility_tolerance` | Independent acceptance tolerance for normalized constraint and bound violations. This field belongs directly under `optimization`. |

The optimizer scales each selected design variable to `[0, 1]` using its bounds.
It signs the objective for minimization and divides objectives and constraints
by their configured normalization scales. Inputs and app results retain their
physical units. Solver tolerances and derivative steps refer to the scaled
problem, so choose meaningful scales before tightening them.

### Step fractions and recovery

Step fractions multiply SLSQP's proposed search direction. A fraction of `0.5`
tries half that direction; it is not a fixed distance in physical units.
Both fractions must be greater than zero and at most one. SLSQP's own line
search may shorten a trial further.

With the defaults, starting fractions grow as `0.25, 0.375, 0.5625, 0.84375, 1.0`
until reaching the maximum, provided no domain recovery occurs. After recovery,
growth resumes from the fraction actually accepted. A small fraction at every
iteration can make convergence slower. Set `step_growth_factor: null` to return
to the maximum at the next iteration, or `1.0` to hold the reduced fraction.

The app always enables invalid-trial recovery. `max_step_fraction` and
`finite_diff_step` are YAML/Python controls, preserved when loading and saving
a design but not shown in the editor. The advanced demo illustrates why these
policies can produce different histories even for the same starting design.

## When the model cannot evaluate a design

A design can satisfy all variable bounds and still have no physical meanline
solution. For example, a combination of flow angles and velocity ratios can
require a negative squared velocity component.

For recognized invalid line-search trials, the solver moves back toward the
last accepted design by `backtrack_factor` and tries again, up to
`max_backtracks` times. Rejected trials do not appear as accepted history
points. Recovery preserves the model equations; it cannot make incompatible
physical requirements feasible.

A run needs a physically evaluable starting design. That design may violate
optimization constraints: SLSQP can work toward satisfying them. Recovery
cannot repair an unevaluable start because there is no accepted point to move
toward. Python callers receive `ModelDomainError` for an invalid initial model
evaluation; the app reports the failure and retains its displayed design.

Derivative probes are separate from line-search trials. Each uses one
bound-admissible finite-difference probe; an invalid probe stops the solve
with a diagnostic. Reducing the first step fraction does not change that probe.
Review the reported variable, starting design, bounds, and `finite_diff_step`
if this occurs. Consistency errors and unexpected programming errors also stop
the run rather than triggering recovery.

If recovery is exhausted after a solve begins, inspect the unsuccessful result
at the last accepted design. Use the failure message and rejection history to
identify the physical limit, then reconsider the start, bounds, or constraints.

## Read the convergence report

The terminal reports the signed, normalized objective that SLSQP minimizes.
The app and public callback history show physical objectives and variables.
Step and KKT diagnostics use solver normalization in both places.

The Python API preserves both frontend and backend views of the solve:

| Result | Coordinates and evaluation counts |
| --- | --- |
| `result.frontend_result` | Detached PySolverView public result. `x` is physical, `x_solver` is normalized, and `nfev` includes frontend model and derivative-probe calls. |
| `result.solver` | Raw backend result. `x` and history use normalized coordinates and the driver's KKT convention; `nfev` excludes derivative probes supplied by the frontend. |
| `result.n_evaluations` | Successful uncached turbine evaluations, including the independent fresh final evaluation. |

The YAML report saves `frontend_result` and `frontend_convergence_history` as
well as the backend diagnostics. These counters measure different work and
need not be equal.

| Diagnostic | How to interpret it |
| --- | --- |
| `step_norm` | Distance between consecutive accepted designs in normalized coordinates. A small step alone does not establish convergence. |
| `objective_change` | Signed change in the normalized minimized objective. |
| `primal_feasibility` | Largest normalized constraint or bound violation. Compare with `feasibility_tolerance`. |
| `optimality` | Independent first-order optimality residual, including stationarity, multiplier signs, and complementarity. Compare with `optimality_tol`. |
| `kkt_ratio` | Largest ratio of feasibility or optimality residual to its acceptance tolerance. A value at most one passes the independent check. |
| `diagnostics_available` | Whether the derivatives and multiplier assessment needed for the check were available. Missing values do not count as a pass. |

A successful result requires native SLSQP convergence, a fresh final meanline
evaluation, and the independent feasibility and first-order optimality checks.
`ftol` controls SLSQP's stopping rules; `feasibility_tolerance` and
`optimality_tol` control independent acceptance. Changing one does not silently
change the others. A feasible design at the iteration limit is still reported
as unsuccessful.

The KKT check includes variable bounds. It provides numerical evidence of
first-order optimality, subject to derivative accuracy; it does not prove a
global optimum. Compare multiple physically plausible starting designs when
assessing the quality of a solution. Tightening `optimality_tol` below the
accuracy of the finite differences does not produce more accurate derivatives.

Accepted history points contain matching configurations and turbine results;
rejected trials and derivative probes are excluded. Saved diagnostics include
evaluation counts, timing, rejection counts, and recovery events. Python callers
can inspect `result.solver.events` when investigating repeated recovery.
Recovery events and counts remain available in callbacks and exports; the
frontend's terminal report does not print the old low-level trial-rejection log.

The solver implementation, independent KKT mathematics, and generic numerical
audits are maintained in
[PySolverView](https://github.com/turbo-sim/pysolver_view). Turbodash's integration
tests check turbine variable mapping, accepted physical snapshots, exports,
and the app's optimization workflow.
