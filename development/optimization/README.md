# Turbine optimization example

This development API optimizes axial and radial turbines using the existing
Turbodash meanline model. It follows ThermOpt's problem definition and solves with
`pysolver_view` (PSV), including its convergence history and progress callbacks.
The reusable API lives in `turbodash/optimization/turbine.py` and is exported
by `turbodash.optimization`. This directory contains only examples, inputs, and
tests; it has no copy or compatibility shim of the optimization implementation. The stable and existing
experimental apps are unchanged. A separate optimization edition is available:

```powershell
python -m pip install -e ".[optimization]"
python -m turbodash.apps.turbine_optimization
```

See [the app guide](../../turbodash/apps/README.md) for controls and execution details.

## Running the example

Use the project's Python environment and install the optimization dependency:

```powershell
python -m pip install -r development/optimization/requirements.txt
python -m development.optimization.run
```

Direct execution also works: `python development/optimization/run.py`.
There are no command-line arguments. Set the globals at the top of `run.py`:

```python
CASE_FILES = ("case_axial.yaml",)  # Or ("case_radial.yaml",), or both filenames.
SHOW_PLOTS = True
PLOT_TURBINE_EACH_ITERATION = False
PLOT_CONVERGENCE_LIVE = False
```

`OUTPUT_DIRECTORY` selects the output parent folder. The default is `output/`
beside the runner. Each case has its own subdirectory: `output/axial/` or `output/radial/`.

- `input_case.yaml`: the source case, including its optimization settings.
- `initial_case.yaml`: the effective starting inputs after initial-value overrides.
- `optimized_case.yaml`: final optimized inputs, with initial overrides removed
  so rerunning this file starts from the saved optimized design.
- `optimization_result.yaml`: final turbine results, constraint diagnostics, and
  physical and native solver histories.
- `initial_turbine.png` / `.svg`, `optimized_turbine.png` / `.svg`, and
  `convergence_history.png` / `.svg`.

The runner plots the initial and final turbines and the convergence history.
Live plotting is optional. All numeric output is YAML; no JSON files are written.
The result's progress history uses physical variables. The native solver history
uses normalized coordinates and the signed minimized objective, so maximizing
efficiency produces a negative objective in the convergence plot.
The result YAML records `solver_elapsed_seconds` and `model_evaluations`; the
runner also prints these. Solver time excludes the initial/final figure exports.
Live plots inside solver callbacks add to this time, so disable them when
comparing numerical performance or running batches of cases.

## Editing a case

`case_axial.yaml` and `case_radial.yaml` are standalone files containing `inputs`
and `optimization`. Both maximize total-to-static efficiency with zero stator and
rotor flaring. Zero flaring demonstrates constraints; it is not a general design
rule. The original inputs violate it, so a feasible optimized design can have
lower efficiency than the original, infeasible design.

Add or comment out entries under `optimization.design_variables`:

```yaml
design_variables:
  blade_velocity_ratio:
    lower_value: 0.5
    # initial_value: 0.58
    upper_value: 0.65
  rotational_speed:
    lower_value: 0.5
    upper_value: 0.95
```

Names are dotted paths relative to `inputs`, or labels with an explicit `path`.
For example, `stages.0.blade_geometry.rotor.solidity_chord` addresses the first
stage's rotor. Missing or null `initial_value` uses the current input value.
Commenting out a variable keeps its original input fixed. Bounds and initial
values use the model's native units. Initial values outside the bounds are clipped
to the nearest bound, with a printed message showing the original and new values.
This applies both to current configuration values and explicit initial overrides.

Speed uses two ordinary input fields: `rotational_speed_type: specific` and
`rotational_speed: 0.7`. With type `specific`, the value is dimensionless specific
speed; with `actual`, it is rpm. Optimize the numeric `rotational_speed` field
directly. The type stays fixed during a solve. Older nested speed inputs are still
accepted by the turbine calculation for existing app cases.

`objective_function.variable` selects one scalar output; `type` is `maximize` or
`minimize`. An optional positive `normalize` scales the objective. Constraints are
independent list entries; omit the list or set it to null for an unconstrained
case. The examples include a commented maximum-Mach constraint.

Selectors accept an optional leading `$` and `*` over lists or dictionaries.
Each wildcard match becomes a separate constraint. Supported types are `=`, `<=`,
and `>=` (also `<` and `>` as non-strict aliases). Targets can be numeric values,
compatible arrays, or result selectors. Python callers can also supply callables
for objective variables, constraint variables, and targets; selectors themselves
do not evaluate algebra.

Constraint `normalize` is `False` for native residual units, a positive scale, or
`True` to divide by the absolute target. Use an explicit scale for zero targets.
The separate `optimization.feasibility_tolerance` applies to normalized residuals;
with scale 10 degrees, tolerance `1e-6` permits `1e-5` degrees of flaring.
`solver_options` passes PSV options such as `library`, `method`, `update_on`,
derivative settings, tolerance, and iteration limit. The example uses SciPy SLSQP
through PSV, with numerical derivatives and bounded normalized design variables.

## Python API and progress callbacks

`turbodash/optimization/turbine.py` contains `TurbineOptimizationProblem`, `TurbinePlotCallback`,
and the optimization, plotting, and export helpers:

```python
from turbodash.optimization import (
    load_case, optimize_turbine, save_optimization_results,
    plot_turbine, plot_convergence,
)

config = load_case("development/optimization/case_axial.yaml")

result = optimize_turbine(config, callback=lambda p: print(p["objective"]))
geometry = result.geometry
save_optimization_results(result, "development/optimization/output/custom")
```

`optimize_turbine(config, optimization=None, *, evaluator=None, callback=None)`
uses `config["optimization"]` unless given a separate dictionary. Results expose
`config`, `turbine_results`, `geometry`, physical `design_variables`, objectives,
constraint diagnostics, and `history`. `solver` is the actual PSV solver instance.
`solver_success` records convergence; `feasible` records constraint acceptance;
`success` requires both. Check these before using the design. Invalid model trials
raise an error with the offending variables rather than artificial penalties.

Callbacks receive independent copies of `iteration`, `objective`,
`max_constraint_violation`, `design_variables`, `config`, and `turbine_results`.
Physical values are recovered from PSV's normalized vector, and duplicate
gradient updates are filtered. The runner uses this hook for optional live turbine
plots. `plot_turbine(results, title=None, figure=None)` can update an existing
figure; `plot_convergence(result, show=False)` plots the recorded solver history.

A future app could run the optimizer in a background worker and use callbacks to
publish geometry. A button could release a waiting callback to continue execution.
PSV does not expose a native step iterator; repeatedly restarting with one allowed
iteration would discard the optimizer's accumulated state. No app stepping is
implemented here.

## Choosing variables and interpreting results

Useful variables include blade velocity ratio, specific speed, reaction, stator
exit angle, meridional velocity ratios, radius ratios, and row solidity. Axial
aspect ratio, maximum thickness/chord, and trailing-edge thickness/opening affect
losses. Use realistic thickness bounds; local bound optima are not finished designs.
Keep radial row radius ratios away from one to preserve nonzero meridional chord.
For multistage work-split weights, fix one because uniform rescaling is redundant.

Metal angles follow velocity triangles; independent angles require incidence or
deviation modeling. Radial aspect ratio is derived. First-stage `radius_ratio_01`
has no effect in a single stage. Thickness location and trailing-edge wedge angle
affect blade construction rather than the production losses; leading-edge geometry
principally affects incidence, currently zero with triangle-matched metal angles.
Rotor clearance is fixed at 1% of height and stator clearance at zero.

Zero flaring requires equal inlet/outlet heights. Changing aspect ratio alone
cannot enforce it; the examples vary both row meridional velocity ratios.
The model reconstructs isentropic flow and deducts empirical losses, so results
are preliminary meanline designs and SLSQP finds local optima. Blade counts remain
continuous during optimization; realize integer geometry separately for export.

Run the API and real-model checks with
`python -m unittest development.optimization.test_optimization -v`.
