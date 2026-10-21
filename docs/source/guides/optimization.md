# SLSQP optimization and model-domain recovery

Turbodash runs SLSQP through its own driver around SciPy's compiled solver.
The same driver serves the Python API and the optimization workbench. It can
shorten a trial step when the meanline model identifies a physically invalid
design, while preserving the model's equations and valid numerical results.
PySolverView remains an installation dependency through `jaxprop`; it does
not drive these optimization runs.

Use the [calculation workflow](calculations.md) to choose variables and
constraints. The [standalone examples](../../../demos/optimization/README.md)
in `demos/optimization/` use the same production API as the app:

```python
from turbodash.optimization import load_case, optimize_turbine

case = load_case("demos/optimization/case_axial.yaml")
result = optimize_turbine(case)
print(result.success, result.message)
```

The optimizer maps each selected physical variable to the interval `[0, 1]`
using its lower and upper bounds. The objective is signed for minimization
and divided by its optional normalization. Constraint residuals are divided
by their configured scales. The model and the app continue to use physical
input units.

## Why use a dedicated SLSQP driver?

A turbine can satisfy every individual variable bound and still have no real
velocity triangle. At such a trial point, the model cannot supply a valid
objective and constraints. The driver needs to reject that evaluation, shorten
the pending step, and retry without losing the solver's accumulated state.
After recovery, it can restore the step fraction gradually to reduce the chance
of immediately crossing the model domain again.

In the supported SciPy version, `scipy.optimize.minimize(method="SLSQP")` does
not expose this pending-trial recovery or the step-fraction controls. Its
iteration callback cannot intercept and retry an undefined pending model
evaluation. Simply evaluating a shorter step inside an objective
function would also leave the solver's internal step and line-search state
inconsistent with the evaluated point.

Turbodash therefore calls the same compiled SciPy SLSQP algorithm through a
small private bridge that updates those quantities together. SciPy still
performs the quadratic subproblems, merit line search, and BFGS updates.
Recognized domain failures trigger another model evaluation at a shorter step;
the driver does not replace invalid equations with clipped roots or invented
objective penalties. Recovery is limited and may still fail.

Scaling, derivative handling, and independent KKT reporting could be built
around the public interface. The reason for the private bridge is control of
the pending trial and its internal state. The tradeoff is a pinned SciPy version
and explicit compatibility work when upgrading it.

The source separates these responsibilities:

| Module in `turbodash/optimization/` | Purpose |
| --- | --- |
| `__init__.py` | Exposes the shared Python API used by scripts and the app. |
| `turbine.py` | Translates configured physical variables and responses into a scaled optimization problem; assembles physical results, histories, and exports. |
| `slsqp.py` | Runs the evaluation loop, controls trial fractions and recovery, computes derivatives, and coordinates reporting and acceptance. |
| `_slsqp_backend.py` | Owns the private SciPy state and workspace and consistently shortens pending trials. |
| `kkt.py` | Independently evaluates first-order feasibility and optimality residuals, including variable bounds. |

## Configure the solver

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

Both step fractions must be greater than zero and at most one. With gradual
restoration disabled, the smaller fraction applies on the first iteration;
later iterations use `max_step_fraction`. These controls multiply the QP direction computed by
SLSQP. A fraction of `0.5` is half that direction, not an absolute limit of
half a physical unit or half a variable's range. SLSQP can shorten the trial
further through its merit-function line search.

The defaults use `first_step_fraction: 0.25`, `max_step_fraction: 1.0`, domain
backtracking enabled, and growth factor `1.5`. The first move is conservative
and subsequent steps grow toward the full fraction. Applying a small fraction at every iteration can
slow convergence considerably. Changing `finite_diff_step` changes derivative
sampling, not the initial line-search step.

For a difficult starting design, gradual restoration can avoid immediately
returning to the full fraction at the second iteration. For example:

```yaml
optimization:
  solver_options:
    first_step_fraction: 0.2
    max_step_fraction: 1.0
    step_growth_factor: 1.5
```

Without domain recovery, at major iteration `k`, starting from `k = 1`, the initial trial fraction is
`min(max_step_fraction, first_step_fraction * step_growth_factor**(k - 1))`.
This example gives fractions `0.2, 0.3, 0.45, 0.675, 1.0`, then `1.0` for the
remaining iterations. After an iteration that needed domain recovery, the
schedule restarts from the fraction actually accepted relative to that
iteration's original QP direction, including further native line-search
reductions. The next iteration starts at that fraction times the growth
factor, capped at `max_step_fraction`. For example, recovery to `0.25` gives
subsequent starting fractions `0.375, 0.5625, 0.84375, 1.0` with growth `1.5`.
Further domain recovery restarts this process again.

Native line search and domain recovery can still shorten any trial further.
A growth factor of `1.0` holds the starting fraction until a domain recovery
reduces it again. Set `step_growth_factor: null` to restore the maximum
immediately on the next iteration instead. Gradual restoration is enabled by
default and does not guarantee physically valid trials. The default iteration
limit is `200` and the default first-step fraction is `0.25`.

The workbench's **Optimization ? Configuration** tab shows the iteration limit
and one solver tolerance (`ftol`). **Show expert settings** expands the same view
inline: feasibility and KKT optimality tolerances appear beside the solver
tolerance, followed by the first step fraction, growth factor, reduction factor,
and trial retry limit. All three tolerances display in scientific notation.
The switch changes visibility only: hiding expert values does not reset them or
change their role in final acceptance. The three tolerances remain independent;
changing `ftol` does not silently rescale the other two.

The app always enables invalid-trial recovery. Maximum step fraction and the
finite-difference step remain YAML/Python options and are preserved when loading
and saving a design, but are not shown in the editor. The Python SLSQP interface
still exposes `ftol`, `feasibility_tol`, and `optimality_tol` independently;
turbine YAML uses the top-level `feasibility_tolerance` field for feasibility. The retired `max_fd_backtracks` key is discarded when
loading legacy YAML, and is not exported. Direct `SLSQPOptions` construction
no longer accepts it.

Older configurations using `tolerance`, `max_iterations`, or
`derivative_abs_step` are accepted as aliases for `ftol`, `maxiter`, and
`finite_diff_step`. Legacy `library: scipy` and `method: slsqp` still select
this driver; they do not invoke PySolverView.

## Recognized model failures

Independent variable bounds cannot guarantee that a combination describes a
valid turbine. For example, reaction, meridional velocity ratios, radius
ratios, and flow angles can individually lie within bounds while jointly
requiring a negative squared rotor-exit velocity component.

The meanline model checks recognized domain conditions before the invalid
operation. It raises `ModelDomainError` carrying the location and numerical
details of the failure. Recognized cases include invalid enthalpy drops or
physical scales, singular divisors, negative squared velocity quantities,
known equation-of-state domain failures, invalid continuous geometry
dimensions, and nonfinite loss-model outputs. Existing loss-model extrapolation
diagnostics remain informational when the model still returns valid values.
The optimization adapter translates that exception
into a recoverable solver evaluation failure. It does not substitute a large
objective penalty, invent constraint results, or silently evaluate another
design under the requested coordinates.

The exception has `recoverable=True` and a JSON-safe `to_dict()` representation.
Its metadata includes `code`, `quantity`, `value`, `margin` when defined,
`stage_index` (zero-based), `stage_name`, and `station` when available. These
fields identify the physical condition and stage without parsing a message.
Nonfinite values are represented explicitly in the serialized diagnostics.

For a rejected line-search trial, the driver moves toward the current accepted
design by `backtrack_factor` and evaluates the shorter trial. It updates the
stored SLSQP search step and corresponding line-search state consistently.
The native solver then judges the valid trial using its usual merit test.
The rejected design never becomes an accepted progress snapshot.

Finite differences choose one probe direction and distance from the available
variable bounds before evaluating the model. There is no derivative retry or
step-halving mechanism. A physically invalid probe stops the solve with its
variable index and failure metadata; no derivative or KKT certificate is
invented. This is separate from recovery of invalid line-search trials.

There must be a valid starting design. Recovery cannot repair an invalid
initial point because there is no accepted design to backtrack toward. An
invalid start or exhausted recovery budget ends the run with a diagnostic;
the app retains the previously displayed design. Adjust the starting inputs
or bounds and run again. A physically valid starting design may still violate
the requested optimization constraints, such as flaring limits: SLSQP can work
toward satisfying those constraints.

For Python callers, an invalid initial turbine evaluation raises
`ModelDomainError` before the solve starts. The app records its structured
metadata in the failed job status. Once a valid solve has begun, exhausting a
recovery budget returns an unsuccessful result at the last accepted design.

Consistency checks raise `ModelConsistencyError` with `recoverable=False`
and diagnostic context.
These checks detect disagreement between quantities that should satisfy the
same equations. They are not automatically treated as recoverable physical
domain failures. Unexpected programming exceptions also propagate as failures.
This distinction prevents retry logic from concealing implementation errors.

The checks do not replace a negative radicand with zero, force an invalid root
to be real, or alter the meanline equations. A design without a real solution
is reported as invalid. Successful evaluations retain the existing numerical
behavior.

## Understand the report and saved result

The terminal displays one SLSQP convergence report, with diagnostics for
rejected trials and termination. The report's objective is the signed,
normalized value minimized by the solver. Its step norm is the Euclidean
distance between consecutive accepted designs in normalized coordinates,
not the proposed QP direction or a line-search fraction. The objective change
is the signed change in the normalized minimized objective. The initial
design has zero step norm and objective change.

The app's separate progress summary and public callback history use physical
objectives and variables. The accompanying step and KKT diagnostics retain
solver normalization, so they match the terminal. Each accepted progress
snapshot contains the matching configuration and turbine results; derivative
probes and rejected trials are excluded. KKT diagnostics use derivatives
evaluated at that snapshot, rather than derivatives from the previous design.

A result is applied only when the native solver reports convergence and the
independent final checks pass. The final design is evaluated through the
strict meanline model. Constraint feasibility uses
`optimization.feasibility_tolerance`; first-order optimality uses
`solver_options.optimality_tol`.

### KKT residuals and tolerances

Write the normalized problem as minimizing $f(x)$ subject to $h(x)=0$,
$c(x)\geq 0$, and $l\leq x\leq u$. Equality residuals retain their configured
sign; inequalities are converted to this nonnegative convention internally.
The Lagrangian is

$$
L=f-\lambda^T h-\mu^T c-z_L^T(x-l)-z_U^T(u-x),
$$

where $\lambda$ is unrestricted and $\mu,z_L,z_U\geq 0$. The check includes
every finite variable bound, even though SciPy's native QP multipliers omit
bound multipliers. Absent bounds contribute neither a residual nor a multiplier.

Four infinity-norm residuals are reported:

| Residual | Definition |
| --- | --- |
| `stationarity` | $\|\nabla f-J_h^T\lambda-J_c^T\mu-z_L+z_U\|_\infty$ |
| `primal_feasibility` | Largest equality absolute residual, inequality violation, or bound violation. |
| `dual_feasibility` | Largest negative part of any inequality or bound multiplier. Equality multipliers have no sign restriction. |
| `complementarity` | Largest absolute product of an inequality/bound multiplier and its corresponding slack: $\mu_i c_i$, $(z_L)_j(x_j-l_j)$, $(z_U)_j(u_j-x_j)$. |

Turbodash estimates multipliers at each reported design by solving a linear
minimax problem: minimize the largest stationarity or complementarity residual,
with unrestricted equality multipliers and nonnegative inequality/bound
multipliers. All constraints and finite bounds participate. There is no
arbitrary active-set distance cutoff; an inactive constraint cannot balance
the objective gradient without also contributing its multiplier-times-slack
residual. The diagnostic multipliers are independent of SLSQP's native QP
multipliers, which remain available separately. Multipliers need not be unique
when constraint gradients are dependent.

`optimality` is the maximum of stationarity, dual feasibility, and
complementarity. `kkt_residual` is the maximum of all four residuals.
The tolerance-aware quantity is

$$
\text{kkt\_ratio}=\max\left(
\frac{\text{primal\_feasibility}}{\text{feasibility\_tolerance}},
\frac{\text{optimality}}{\text{optimality\_tol}}\right).
$$

A ratio at most one passes the independent KKT check. The default feasibility
tolerance is $10^{-6}$ and optimality tolerance is $10^{-4}$. This separates
physical constraint satisfaction from first-order accuracy. The native
`ftol` (default $10^{-9}$) controls SLSQP's own stopping rules; its objective
and step tests do not imply that every KKT residual is below `ftol`. A small
step alone cannot pass the independent check. Native solver success is still
required: a feasible design at the iteration limit is reported as unsuccessful.
These distinctions follow the [SciPy SLSQP documentation](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html)
and the stopping tests in the [pinned SciPy 1.17.1 implementation](https://github.com/scipy/scipy/blob/v1.17.1/scipy/optimize/__slsqp.c).

These residuals describe the scaled problem and depend on derivative accuracy.
The same domain-aware finite differences used by the solver are used for the
check. Reducing `optimality_tol` far below the derivative error does not create
additional accuracy; review scaling and derivative step size when tightening
it. A passing check is numerical evidence of first-order KKT conditions, not
a proof of global optimality or second-order sufficiency. If derivatives or
the multiplier assessment are unavailable, the report marks diagnostics
unavailable, saves null values, and does not claim that the KKT check passed.
Constraint qualifications also matter: at a degenerate constraint, a local
optimum need not admit KKT multipliers. A failed check is therefore not by itself
a proof that a better feasible design exists.

Saved optimization diagnostics include `solver_status`, `step_norm`,
`objective_change`, all four KKT residuals, `optimality`, `kkt_residual`,
`kkt_ratio`, `diagnostics_available`, and the complete `kkt` assessment.
`kkt_multipliers` separates equality, inequality, lower-bound, and upper-bound
multipliers. Constraint multipliers follow their configured component order;
bound multipliers follow design-variable order. The same diagnostics accompany
the accepted history and app job result, alongside evaluation counts, elapsed
time, `domain_rejections`, `derivative_rejections`, and `recovery_events`.
Each rejection
event records solver coordinates in `x` and diagnostic metadata in `failure`;
turbine failures also include the physical `design_variables`. Python callers
can inspect the full event list through `result.solver.events`. Inspect these
when a run needs repeated
recovery. A smaller trial can avoid a domain boundary, but cannot make an
incompatible set of physical requirements feasible.

## SciPy compatibility and maintenance

The public SciPy SLSQP interface does not expose these step controls.
Turbodash therefore isolates its private SciPy integration in
`turbodash/optimization/_slsqp_backend.py`, with generic driver logic in
`turbodash/optimization/slsqp.py` and turbine-specific adaptation in
`turbodash/optimization/turbine.py`. Run state belongs to each solve; the
driver does not monkeypatch SciPy or modify installed packages.

This implementation requires **SciPy 1.17.1**. Its version and expected
internal workspace layout are checked before using the private interface.
Upgrading SciPy requires reviewing that interface and rerunning the numerical,
recovery, turbine, and app regressions. The maintenance harness and retained
recovery/KKT findings live in `development/optimization/`; user examples live
in `demos/optimization/`. Production runs use the modules above.
