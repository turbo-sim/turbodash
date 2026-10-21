# Cyclopentane: tenth design and constrained optimization

`cyclopentane_10.yaml` is the retained tenth design, in the current input format.
It preserves its operating point, velocity ratios, reaction, and blade shape.
The former Zweifel targets (0.7 stator, 0.9 rotor) were converted with the current
backend at that design point to true-chord solidity (`chord / pitch`):
1.1209982569657784 stator and 2.1539034913192463 rotor.
Saved historical performance/geometry snapshots were removed; scripts recalculate
from inputs. Earlier YAML designs, including the extra axial snapshot, were
removed, together with their obsolete ZIP exports.

Run from the repository root with `poetry run python`, or use your environment's
Python interpreter. Paths below are resolved from each script, so the default
case and output folders also work when launched from another working directory.

## Evaluate and export the retained design

```bash
poetry run python demos/cyclopentane_case/main.py
poetry run python demos/cyclopentane_case/export_rotor_turbogrid.py
poetry run python demos/cyclopentane_case/export_rotor_3d_free_vortex.py
```

`main.py` saves four plots, current solved YAML, and stator/rotor coordinate CSVs
under `output/design_10/`. Add `--show` to open figures. The two rotor exporters
also recalculate the current inputs, then realize integer blade counts through
the backend's geometry helper. They do not round a continuous blade count while
retaining an incompatible pitch or opening. They write their separate default
folders under `output/`. See [TurboGrid options](turbogrid_export.md).
All three commands accept `--yaml-path` and `--output-dir`.

## Optimize the pure-impulse axial stage

```bash
poetry run python demos/cyclopentane_case/optimize.py
```

`cyclopentane_optimization.yaml` contains both the initial inputs and the full
optimization settings. It can be loaded directly into the optimization app.
The script calls the same production meanline model and SLSQP driver as the app.
It starts from the unoptimized tenth design, not from a saved optimum.

Eight effective variables are free: blade velocity ratio, specific speed,
stator exit flow angle, the three meridional velocity ratios, and stator/rotor
true-chord solidity. Their bounds are in the YAML; the meridional ratios can go
down to 0.01. The stator and rotor solidity bounds are 0.5 to 3.0.

The following stay fixed:

- Degree of reaction is 0.0 (pure impulse).
- Stator and rotor aspect ratios are 2.0.
- Every blade thickness and shape input retains the tenth design's value,
  including maximum-thickness/chord and trailing-edge-thickness/opening ratios.
  Dimensional thickness can change when chord or opening changes.
- Axial radius ratios are 1.0, and the single stage's reference work weight is 1.0.
- Fluid, inlet state, exit pressure, mass flow, and inlet flow angle are fixed.

Both row flaring angles are equality-constrained to 0 degrees. Both row outlet
hub-to-tip ratios are equality-constrained to 0.90. With constant mean radius and
zero flaring, each row's inlet ratio is also 0.90. These are four independent
scalar constraints; imposing the inlet ratios again would duplicate the same
conditions. Constraint `normalize` is the residual divisor: 10 degrees for
flaring and 1 for the dimensionless hub-to-tip ratio. Feasibility tolerance is
1e-6 in normalized units, and the independent KKT tolerance is 1e-4.

`output/optimization/` contains:

- `input_case.yaml` and `initial_case.yaml`: source and effective starting cases.
- `optimized_case.yaml`: final inputs and settings, directly loadable in the app.
- `optimization_result.yaml`: acceptance flags, constraint residuals, KKT
  diagnostics, evaluation history, and current turbine results.
- Initial/final turbine figures and objective/constraint convergence plots.

The script prints acceptance and the achieved row geometry, and returns a
nonzero exit status if the final design fails acceptance. Inspect the acceptance
flags before adopting a result; outputs are saved for unsuccessful runs too.
Use `--yaml-path`, `--output-dir`, or `--show` as needed.

## Checked with the current backend

All three existing scripts ran successfully with the migrated tenth design.
Its current total-to-static efficiency is 75.1331%; the former saved efficiency
belonged to an older backend and is not retained as a reference result.
The optimization example converged in 19 iterations to 85.7563% efficiency,
with both inlet/outlet hub-to-tip ratios at 0.90000000 and both row flaring
angles within 7e-12 degrees of zero. Maximum normalized constraint violation
was 1.31e-12 and KKT optimality was 2.28e-7. Numerical results may vary slightly
with solver/platform versions.
