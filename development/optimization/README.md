# Turbine optimization integration validation

This directory keeps the turbine-specific integration matrix, fixtures, and
historical evidence after moving numerical optimization to PySolverView.

PySolverView owns SLSQP, the SciPy bridge, finite differences, domain recovery,
KKT checks, convergence plots, and optimization reports. Turbodash retains its
[physical problem adapter](../../turbodash/optimization.py).
User examples live in [demos/optimization](../../demos/optimization/README.md).

## Why retain this directory

The 52-case matrix validates real turbine starts, axial and radial geometries,
physical selectors and normalization, equalities and inequalities, impossible
constraints, tolerance profiles, and finite-difference settings. It checks
physical results and callback/export alignment against the frontend results.
PySolverView's bundled analytical studies cannot test those turbine boundaries.

The former local KKT reconstruction, finite-difference audit, custom convergence
plots, and optimization report renderer have been removed. Each case calls
`result.optimizer.plot_convergence_history`, `print_convergence_history`, and
`print_optimization_report` directly. KKT status and residuals in the matrix
data come from PySolverView without recalculation.

| File | Purpose |
| --- | --- |
| [verify_kkt.py](verify_kkt.py) | Real turbine integration matrix, filtering, workers, and resumption. |
| [case_poor_start.yaml](case_poor_start.yaml) | Physically evaluable turbine stress fixture. |
| [kkt_validation.md](kkt_validation.md) | Case definitions, tolerance profiles, and validation interpretation. |
| [historical_benchmark.json](historical_benchmark.json) | Previous Turbodash-driver evidence, not a current acceptance oracle. |

## Running the checks

Run from the repository root after `poetry install` has installed the required
PySolverView PyPI dependency, version `>=0.7.2,<0.8` (Poetry `^0.7.2`):

```powershell
poetry run python -B -m pytest tests/test_optimization_api.py tests/test_kkt_history_alignment.py tests/test_kkt_reporting.py -q
poetry run python -B development/optimization/verify_kkt.py --preflight
poetry run python -B development/optimization/verify_kkt.py --filter axial_nominal__flaring__standard --workers 1
poetry run python -B development/optimization/verify_kkt.py --workers 2
```

The full matrix is a longer integration study. A rejected optimization can pass
its physical integration checks. Generated exports, PySolverView reports and
plots, and JSON/CSV matrix summaries go to ignored `output/`. Use a fresh output
directory after changing the solver, inputs, or settings. Resume only unchanged
runs; summaries from the old independent audit are automatically rerun.

Generic solver/KKT tests now live in the PySolverView repository under
`tests/optimization/solvers/` and `tests/diagnostics/`. Turbodash's regular tests
retain physical-model, scaling, domain-error translation, callback, and export
coverage. The app and demos use the same public `pysolver_view.OptimizationSolver`
frontend with its `pysolver` SLSQP backend through the turbine adapter. The
frontend owns the optimization report and convergence plots. Numerical defaults and validation come
from `pysolver_view.SLSQPOptions`; SciPy 1.17.1 compatibility is owned there.
